import numpy as np, argparse, time, pickle, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import MELDRobertaCometDataset
from model import MaskedNLLLoss
from commonsense_model import CommonsenseGRUModel
from sklearn.metrics import f1_score, accuracy_score
import csv

def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_MELD_loaders(batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = MELDRobertaCometDataset('train', classify)
    validset = MELDRobertaCometDataset('valid', classify)
    testset = MELDRobertaCometDataset('test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []
    seq = []
    arr = {}

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for data in dataloader:
        seq += data[-1]
        if train:
            optimizer.zero_grad()

        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        
        log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, att2=False)

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        arr[data[-1][0]] = [labels_.data.cpu().numpy(), pred_.data.cpu().numpy()]

        if train:
            total_loss = loss
            total_loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_sense_loss = round(np.sum(losses_sense)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], [alphas, alphas_f, alphas_b, vids], seq, arr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.5, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='simple', help='Attention type in context GRU')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--mode1', type=int, default=2, help='Roberta features to use')
    parser.add_argument('--seed', type=int, default=100, metavar='seed', help='seed')
    parser.add_argument('--norm', type=int, default=0, help='normalization strategy')
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    parser.add_argument('--classify', default='emotion')
    parser.add_argument('--residual', action='store_true', default=False, help='use residual connection')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    emo_gru = True
    if args.classify == 'emotion':
        n_classes  = 7
    elif args.classify == 'sentiment':
        n_classes  = 3
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    global  D_s

    D_m = 1024
    D_s = 768
    D_g = 150
    D_p = 150
    D_r = 150
    D_i = 150
    D_h = 100
    D_a = 100

    D_e = D_p + D_r + D_i

    global seed
    seed = args.seed
    # seed_everything(seed)
    
    model = CommonsenseGRUModel(D_m, D_s, D_g, D_p, D_r, D_i, D_e, D_h, D_a,
                                n_classes=n_classes,
                                listener_state=args.active_listener,
                                context_attention=args.attention,
                                dropout_rec=args.rec_dropout,
                                dropout=args.dropout,
                                emo_gru=emo_gru,
                                mode1=args.mode1,
                                norm=args.norm,
                                residual=args.residual)

    print ('MELD COSMIC Model.')


    if cuda:
        model.cuda()

    if args.classify == 'emotion':
        if args.class_weight:
            if args.mu > 0:
                loss_weights = torch.FloatTensor(create_class_weight(args.mu))
            else:   
                loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 
                0.84847735, 5.42461417, 1.21859721])
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss()
            
    else:
        loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    if args.classify == 'emotion':
        lf = open('logs/cosmic_meld_emotion_logs.txt', 'a')
    elif args.classify == 'sentiment':
        lf = open('logs/cosmic_meld_sentiment_logs.txt', 'a')

    train_loader, valid_loader, test_loader = get_MELD_loaders(batch_size=batch_size, 
                                                               classify=args.classify,
                                                               num_workers=0)

    # added code

    # load the saved model
    bytesform = open('trained_model.pkl', 'rb')
    model = pickle.load(bytesform)

    # open predictions file
    predictions = open('predictions.csv', 'w')
    csv_writer = csv.writer(predictions)
    csv_writer.writerow(['Utterance', 'Speaker', 'Emotion', 'Pred_Emotion', 'Sentiment', 'Dialogue_ID', 'Utterance_ID', 'Season', 'Episode', 'StartTime', 'EndTime'])

    valid_loss, valid_acc, _, valid_labels, valid_pred, valid_fscore, _, seq, arr = train_or_eval_model(model, loss_function, valid_loader, 1)

    # x = 'valid_loss: {}, acc: {}, fscore: {}'.format(valid_loss, valid_acc, valid_fscore)
    # print(x)

    actual = open('dev_sent_emo.csv')
    csv_reader = csv.reader(actual)

    # dialogue_id: [all other info]
    info = {}

    # dialogue_id: [emotion labels]
    actual_labels = {}

    idx = 0
    curr_dia_id = -1
    emo_labels = []
    other_info = []

    for row in csv_reader:
        if idx == 0:
            idx += 1
        else:
            label = 0
            if row[3] == 'neutral':
                label = 0
            elif row[3] == 'surprise':
                label = 1
            elif row[3] == 'fear':
                label = 2
            elif row[3] == 'sadness':
                label = 3
            elif row[3] == 'joy':
                label = 4
            elif row[3] == 'disgust':
                label = 5
            elif row[3] == 'anger':
                label = 6

            if row[5] == curr_dia_id:
                emo_labels.append(label)
                other_info.append(row[1:])
            else:
                if (curr_dia_id != -1):
                    info[str(int(row[5]) - 1)] = other_info
                    actual_labels[str(int(row[5]) - 1)] = emo_labels

                emo_labels = [label]
                other_info = [row[1:]]
                curr_dia_id = row[5]

    info[row[5]] = other_info
    actual_labels[row[5]] = emo_labels

    new_label_preds = {}

    def listtostr(ls):
        fstr = ""
        for val in ls:
            fstr += str(val)
        return fstr

    # -1039
    for key in arr:
        # 8 dialogues
        if key != 1151:
            labels_now = arr[key][0]
            preds_now = arr[key][1]
            str_labels_now = listtostr(labels_now)

            for i in range(1039, 1031, -1):
                if ((int(key) - i) != 35 and (int(key) - i) != 41 and (int(key) - i) != 43 and (int(key) - i) != 59 and (int(key) - i) != 75 and (int(key) - i) != 86 and (int(key) - i) != 100 and (int(key) - i) != 102):
                    old_labels = actual_labels[str(int(key) - i)]

                    str_old_labels = listtostr(old_labels)
                    
                    new_idx = str_labels_now.find(str_old_labels)
                    new_label_preds[str(int(key) - i)] = preds_now[new_idx:(new_idx+len(old_labels))]

                    str_labels_now = str_labels_now[new_idx+len(old_labels):]
                    preds_now = preds_now[new_idx+len(old_labels):]

            
    for key in new_label_preds:
        new_info = info[key]
        for i in range(len(new_label_preds[key])):
            label = ""
            if new_label_preds[key][i] == 0:
                label = 'neutral'
            elif new_label_preds[key][i] == 1:
                label = 'surprise'
            elif new_label_preds[key][i] == 2:
                label = 'fear'
            elif new_label_preds[key][i] == 3:
                label = 'sadness'
            elif new_label_preds[key][i] == 4:
                label = 'joy'
            elif new_label_preds[key][i] == 5:
                label = 'disgust'
            elif new_label_preds[key][i] == 6:
                label = 'anger'

            new_row = new_info[i][0:3] + [label] + new_info[i][3:]
            csv_writer.writerow(new_row)
    
    # 35 (1), 41 (3), 43(2), 59(5), 75(1), 86(4), 100(1), 102(2)
    # take out these ones