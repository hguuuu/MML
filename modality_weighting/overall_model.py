from trimodal_dataset import TrimodalDataset
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle
import sklearn.metrics as metrics
from unimodal_audio_model import UnimodalAudioModel
from unimodal_text_model import UnimodalTextModel
from unimodal_visual_model import UnimodalVisualModel
import sklearn.preprocessing as preprocessing
from collections import defaultdict
import csv

# https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
# https://pytorch.org/docs/stable/notes/faq.html#my-model-reports-cuda-runtime-error-2-out-of-memory
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# https://github.com/declare-lab/conv-emotion/blob/master/COSMIC

class FinalModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim1, hidden_dim2, num_layers, dropout):
        super(FinalModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim1//2, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.linear1 = nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_dim2, out_features=7)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return torch.flatten(output, 1, 2)

def get_emotion_weight(emotion, embedding, modality, text_model, audio_model, visual_model):
    # emotion is the index to look at
    vector = None
    new_embed = None
    if modality == "text":
        vector = text_model.double().get_prob(embedding[0][0][0:2304].view(1, 1, -1))
        new_embed = text_model.double().get_embedding(embedding[0][0][0:2304].view(1, 1, -1))
    elif modality == "audio":
        vector = audio_model.double().get_prob(embedding[0][0][2304:7137].view(1, 1, -1))
        new_embed = audio_model.double().get_embedding(embedding[0][0][2304:7137].view(1, 1, -1))
    else:
        vector = visual_model.double().get_prob(embedding[0][0][7137:].view(1, 1, -1))
        new_embed = visual_model.double().get_embedding(embedding[0][0][7137:].view(1, 1, -1))

    # softmax
    softmax = nn.Softmax(dim=1)
    vector2 = softmax(vector).tolist()[0]
    
    emotion = emotion.tolist()[0]
    return vector2[emotion], new_embed

def train(model, epochs, batch_size, optimizer, text_model, audio_model, visual_model):
    data = TrimodalDataset('unimodal_text_train_embeddings.pkl', 'unimodal_audio_train_embeddings.pkl', 'unimodal_visual_train_embeddings.pkl')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = 0

        for batch in dataloader:

            optimizer.zero_grad()
            
            embedding = batch["embedding"].view(batch_size, 1, -1)

            # passing the weights of each emotion for negative reinforcement (7 epochs, 35.13% accuracy, bimodal)
            # ls_embed = []
            # for emo in range(7):
            #     text_weight = get_emotion_weight(torch.from_numpy(np.asarray([emo])), batch["embedding"].view(batch_size, 1, -1), "text", text_model, audio_model)
            #     audio_weight = get_emotion_weight(torch.from_numpy(np.asarray([emo])), batch["embedding"].view(batch_size, 1, -1), "audio", text_model, audio_model)

            #     weights = preprocessing.normalize(np.array([text_weight, audio_weight]).reshape(1, -1))[0]
            #     if torch.cuda.is_available():
            #         embed = torch.cat([torch.mul(embedding[0][0][0:2304].view(1, 1, -1), text_weight), torch.mul(embedding[0][0][2304:].view(1, 1, -1), audio_weight)], dim=2).cuda()

            #     ls_embed.append(embed)

            # final_embed = torch.cat(ls_embed)
            # preds = model(final_embed)
            # true = torch.cat([batch["emotion"], batch["emotion"], batch["emotion"], batch["emotion"], batch["emotion"], batch["emotion"], batch["emotion"]]).cuda()
                
            # text_weight = non-normalized weight, text_embed1 = embeddings from unimodal
            text_weight, text_embed1 = get_emotion_weight(batch["emotion"], batch["embedding"].view(batch_size, 1, -1), "text", text_model, audio_model, visual_model)
            audio_weight, audio_embed1 = get_emotion_weight(batch["emotion"], batch["embedding"].view(batch_size, 1, -1), "audio", text_model, audio_model, visual_model)
            visual_weight, visual_embed1 = get_emotion_weight(batch["emotion"], batch["embedding"].view(batch_size, 1, -1), "visual", text_model, audio_model, visual_model)

            # normalized weights
            # weights = preprocessing.normalize(np.array([text_weight, audio_weight, visual_weight]).reshape(1, -1))[0]

            # using embeddings from unimodal and normalizing them before weight multiplication
            text_embed = torch.from_numpy(preprocessing.normalize(np.array(text_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            audio_embed = torch.from_numpy(preprocessing.normalize(np.array(audio_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            visual_embed = torch.from_numpy(preprocessing.normalize(np.array(visual_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)

            # using embeddings from original extracted embeddings and normalizing them before weight multiplication
            # text_embed = torch.from_numpy(preprocessing.normalize(np.array(embedding[0][0][0:2304].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            # audio_embed = torch.from_numpy(preprocessing.normalize(np.array(embedding[0][0][2304:7137].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            # visual_embed = torch.from_numpy(preprocessing.normalize(np.array(embedding[0][0][7137:].tolist()).reshape(1, -1))[0]).view(1, 1, -1)

            if torch.cuda.is_available():
                # non-normalized weights
                embed = torch.cat([torch.mul(text_embed, text_weight), torch.mul(audio_embed, audio_weight), torch.mul(visual_embed, visual_weight)], dim=2).cuda()
                # normalized weights
                # embed = torch.cat([torch.mul(text_embed, weights[0]), torch.mul(audio_embed, weights[1]), torch.mul(visual_embed, weights[2])], dim=2).cuda()
                # inverse non-normalized weights
                # embed = torch.cat([torch.mul(text_embed, 1 - text_weight), torch.mul(audio_embed, 1 - audio_weight), torch.mul(visual_embed, 1 - visual_weight)], dim=2).cuda()

                preds = model(embed)
                true = batch["emotion"].cuda()
            else:
                # non-normalized weights
                embed = torch.cat([torch.mul(text_embed, text_weight), torch.mul(audio_embed, audio_weight), torch.mul(visual_embed, visual_weight)], dim=2)
                # normalized weights
                # embed = torch.cat([torch.mul(text_embed, weights[0]), torch.mul(audio_embed, weights[1]), torch.mul(visual_embed, weights[2])], dim=2)
                # inverse non-normalized weights
                # embed = torch.cat([torch.mul(text_embed, 1 - text_weight), torch.mul(audio_embed, 1 - audio_weight), torch.mul(visual_embed, 1 - visual_weight)], dim=2)

                preds = model(embed)
                true = batch["emotion"]

            calc_loss = loss_func(preds, true)
            loss += float(calc_loss)
            
            calc_loss.backward()
            optimizer.step()

        torch.save(model, 'models/overall_model_trimodal1_' + str(epoch) + 'epochs.pt')

        print("Epoch: ", epoch, ", Loss: ", loss/len(data))

def evaluate(model, batch_size=1):
    model.eval()

    data = TrimodalDataset('unimodal_text_dev_embeddings.pkl', 'unimodal_audio_dev_embeddings.pkl', 'unimodal_visual_dev_embeddings.pkl')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    all_preds = []
    all_true = []

    for batch in dataloader:
        # WE DONT HAVE THE LABELS AT EVAL TIME
        embedding = batch["embedding"].view(batch_size, 1, -1)

        emotions_pred = []

        for i in range(7):
            # text_weight = non-normalized weight, text_embed1 = embeddings from unimodal
            text_weight, text_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "text", text_model, audio_model, visual_model)
            audio_weight, audio_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "audio", text_model, audio_model, visual_model)
            visual_weight, visual_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "visual", text_model, audio_model, visual_model)

            # normalized weights
            # weights = preprocessing.normalize(np.array([text_weight, audio_weight, visual_weight]).reshape(1, -1))[0]

            # using embeddings from unimodal and normalizing them before weight multiplication
            text_embed = torch.from_numpy(preprocessing.normalize(np.array(text_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            audio_embed = torch.from_numpy(preprocessing.normalize(np.array(audio_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            visual_embed = torch.from_numpy(preprocessing.normalize(np.array(visual_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)

            # using embeddings from original extracted embeddings and normalizing them before weight multiplication
            # text_embed = torch.from_numpy(preprocessing.normalize(np.array(embedding[0][0][0:2304].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            # audio_embed = torch.from_numpy(preprocessing.normalize(np.array(embedding[0][0][2304:7137].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            # visual_embed = torch.from_numpy(preprocessing.normalize(np.array(embedding[0][0][7137:].tolist()).reshape(1, -1))[0]).view(1, 1, -1)

            # non-normalized weights
            embed = torch.cat([torch.mul(text_embed, text_weight), torch.mul(audio_embed, audio_weight), torch.mul(visual_embed, visual_weight)], dim=2)
            # normalized weights
            # embed = torch.cat([torch.mul(text_embed, weights[0]), torch.mul(audio_embed, weights[1]), torch.mul(visual_embed, weights[2])], dim=2)
            # inverse non-normalized weights
            # embed = torch.cat([torch.mul(text_embed, 1 - text_weight), torch.mul(audio_embed, 1 - audio_weight), torch.mul(visual_embed, 1 - visual_weight)], dim=2)

            preds = model(embed)

            softmax = nn.Softmax(dim=1)
            preds = softmax(preds)[0]
            
            emotions_pred.append(preds[i].tolist())

        preds = np.argmax(emotions_pred)
        true = batch["emotion"].tolist()
        
        all_preds.append(preds)
        all_true += true

    return metrics.f1_score(all_true, all_preds, average='weighted')

def eval_emotion(model, batch_size=1):
    model.eval()

    data = TrimodalDataset('unimodal_text_test_embeddings.pkl', 'unimodal_audio_test_embeddings.pkl', 'unimodal_visual_test_embeddings.pkl')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    all_preds = []
    all_true = []

    count = 0

    emo_weights = {}
    # true emotion first one, guessing emotion second one
    for j in range(7):
        x = {}
        for k in range(7):
            x[str(k)] = {'text': 0, 'audio': 0, 'visual': 0, 'ct': 0}
        emo_weights[str(j)] = x

    for batch in dataloader:
        if count % 100 == 0:
            print(count)
        count += 1

        embedding = batch["embedding"].view(batch_size, 1, -1)
        true = batch["emotion"].tolist()

        for i in range(7):
            text_weight, text_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "text", text_model, audio_model, visual_model)
            audio_weight, audio_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "audio", text_model, audio_model, visual_model)
            visual_weight, visual_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "visual", text_model, audio_model, visual_model)

            emo_weights[str(true[0])][str(i)]['text'] += text_weight
            emo_weights[str(true[0])][str(i)]['audio'] += audio_weight
            emo_weights[str(true[0])][str(i)]['visual'] += visual_weight
            emo_weights[str(true[0])][str(i)]['ct'] += 1

    for key in emo_weights:
        for key2 in emo_weights[key]:
            emo_weights[key][key2]['text'] = emo_weights[key][key2]['text']/emo_weights[key][key2]['ct']
            emo_weights[key][key2]['audio'] = emo_weights[key][key2]['audio']/emo_weights[key][key2]['ct']
            emo_weights[key][key2]['visual'] = emo_weights[key][key2]['visual']/emo_weights[key][key2]['ct']

    print('Neutral: ', emo_weights['0'])
    print('Surprise: ', emo_weights['1'])
    print('Fear: ', emo_weights['2'])
    print('Sadness: ', emo_weights['3'])
    print('Joy: ', emo_weights['4'])
    print('Disgust: ', emo_weights['5'])
    print('Anger: ', emo_weights['6'])

# used for error analysis on the best model (trimodal1)
def evaluate_for_analysis(model, batch_size=1):
    model.eval()

    data = BimodalDataset('unimodal_text_dev_embeddings.pkl', 'unimodal_audio_dev_embeddings.pkl', 'unimodal_visual_dev_embeddings.pkl')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    all_preds = {}
    all_true = {}

    for batch in dataloader:
        # WE DONT HAVE THE LABELS AT EVAL TIME
        embedding = batch["embedding"].view(batch_size, 1, -1)

        emotions_pred = []

        for i in range(7):
            text_weight, text_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "text", text_model, audio_model, visual_model)
            audio_weight, audio_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "audio", text_model, audio_model, visual_model)
            visual_weight, visual_embed1 = get_emotion_weight(torch.IntTensor([i]), batch["embedding"].view(batch_size, 1, -1), "visual", text_model, audio_model, visual_model)

            text_embed = torch.from_numpy(preprocessing.normalize(np.array(text_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            audio_embed = torch.from_numpy(preprocessing.normalize(np.array(audio_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)
            visual_embed = torch.from_numpy(preprocessing.normalize(np.array(visual_embed1[0][0].tolist()).reshape(1, -1))[0]).view(1, 1, -1)

            embed = torch.cat([torch.mul(text_embed, text_weight), torch.mul(audio_embed, audio_weight), torch.mul(visual_embed, visual_weight)], dim=2)
            preds = model(embed)

            softmax = nn.Softmax(dim=1)
            preds = softmax(preds)[0]
            
            emotions_pred.append(preds[i].tolist())

        preds = np.argmax(emotions_pred)
        true = batch["emotion"].tolist()
        
        all_preds[batch["dialogue_id"][0] + "_" + batch["utterance_id"][0]] = preds
        all_true[batch["dialogue_id"][0] + "_" + batch["utterance_id"][0]] = true

    return all_preds, all_true
        

if __name__ == "__main__":
    torch.manual_seed(25)

    # import unimodal models
    text_model = torch.load('models/text_model.pt')
    audio_model = torch.load('models/audio_model.pt')
    visual_model = torch.load('models/visual_model.pt')

    # with original embeddings
    # model = FinalModel(embedding_dim=10602, hidden_dim1=4096, hidden_dim2=512, num_layers=1, dropout=0.2).double()

    # with embeddings from unimodal models
    # model = FinalModel(embedding_dim=640, hidden_dim1=256, hidden_dim2=128, num_layers=1, dropout=0.2).double()
    
    # if torch.cuda.is_available():
    #     model.cuda()
        # text_model.cuda()
        # audio_model.cuda()
        # visual_model.cuda()

    # train(model=model, epochs=5, batch_size=1, optimizer=optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003), text_model=text_model, audio_model=audio_model, visual_model=visual_model)

    # for i in range(0, 5):
    #     print('epoch: ', i+1)
    #     model = torch.load('models/overall_model_trimodal1_' + str(i) + 'epochs.pt')

    #     print(evaluate(model=model.cpu(), batch_size=1))

    # weights for each emotion (dev)
    model = torch.load('../cloud/models/overall_model_trimodal1_2epochs.pt')

    # open predictions file
    predictions = open('predictions.csv', 'w')
    csv_writer = csv.writer(predictions)
    csv_writer.writerow(['Utterance', 'Emotion', 'Pred_Emotion', 'Dialogue_ID', 'Utterance_ID'])

    # preds, true = evaluate_for_analysis(model=model.cpu(), batch_size=1)
    # pickle.dump(preds, open('preds.pkl', 'wb'))
    # pickle.dump(true, open('true.pkl', 'wb'))

    preds = pickle.load(open('preds.pkl', 'rb'))
    true = pickle.load(open('true.pkl', 'rb'))

    actual = open('../midterm/conv-emotion/COSMIC/erc-training/dev_sent_emo.csv')
    csv_reader = csv.reader(actual)

    # dialogue_id: [all other info]
    info = {}

    idx = 0

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

        info[row[5] + "_" + row[6]] = [row[1], row[5], row[6]]

    for key in preds:
        new_info = info[key]
        label_pred = ""
        label_true = ""
        if preds[key] == 0:
            label_pred = 'neutral'
        elif preds[key] == 1:
            label_pred = 'surprise'
        elif preds[key] == 2:
            label_pred = 'fear'
        elif preds[key] == 3:
            label_pred = 'sadness'
        elif preds[key] == 4:
            label_pred = 'joy'
        elif preds[key] == 5:
            label_pred = 'disgust'
        elif preds[key] == 6:
            label_pred = 'anger'

        if true[key][0] == 0:
            label_true = 'neutral'
        elif true[key][0] == 1:
            label_true = 'surprise'
        elif true[key][0] == 2:
            label_true = 'fear'
        elif true[key][0] == 3:
            label_true = 'sadness'
        elif true[key][0] == 4:
            label_true = 'joy'
        elif true[key][0] == 5:
            label_true = 'disgust'
        elif true[key][0] == 6:
            label_true = 'anger'

        new_row = [new_info[0]] + [label_true, label_pred] + new_info[1:]
        csv_writer.writerow(new_row)