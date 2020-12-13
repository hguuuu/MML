from unimodal_text_dataset import UnimodalTextDataset
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle
import sklearn.metrics as metrics

class UnimodalAudioModel(nn.Module):
    def __init__(self, word_embedding_dim, hidden_dim1, hidden_dim2, num_layers, dropout):
        super(UnimodalAudioModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=word_embedding_dim, hidden_size=hidden_dim1//2, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.linear1 = nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_dim2, out_features=7)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return torch.flatten(output, 1, 2)

    def forward2(self, x):
        output, _ = self.lstm(x)
        output = self.linear1(output)

        return output

    def get_prob(self, embedding):
        probs = self.forward(embedding)
        return probs

    def get_embedding(self, embedding):
        embed = self.forward2(embedding)
        return embed

def train(model, epochs, batch_size, optimizer):
    data = UnimodalTextDataset('unimodal_audio_train_embeddings.pkl')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = 0
        losses = []

        for batch in dataloader:
            optimizer.zero_grad()

            if torch.cuda.is_available():
                preds = model(batch["embedding"].view(batch_size, 1, -1).cuda())
                true = batch["emotion"].cuda()
            else:
                preds = model(batch["embedding"].view(batch_size, 1, -1))
                true = batch["emotion"]

            calc_loss = loss_func(preds, true)
            loss += calc_loss.item()
            losses.append(calc_loss.item())
            
            calc_loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("Epoch: ", epoch, ", Loss: ", loss/len(data))

def evaluate(model, batch_size):
    model.eval()

    data = UnimodalTextDataset('unimodal_audio_dev_embeddings.pkl')
    dataloader = DataLoader(data, batch_size=batch_size, drop_last=True)

    all_preds = []
    all_true = []

    for batch in dataloader:
        preds = model(batch["embedding"].view(batch_size, 1, -1))
        true = batch["emotion"].tolist()

        softmax = nn.Softmax(dim=1)
        preds = softmax(preds)
        preds = torch.argmax(preds, dim=1).tolist()
        
        all_preds += preds
        all_true += true

    return metrics.f1_score(all_true, all_preds, average='weighted')


if __name__=="__main__":
    torch.manual_seed(25)

    # hidden_d1 = [2048, 1024, 512]
    # hidden_d2 = [256, 128, 64]
    # dropout = [0.3, 0.4]
    # epochs = [20]
    # layers = [1]
    # lr = [0.0001, 0.001, 0.01]

    hidden_d1 = [512]
    hidden_d2 = [128]
    dropout = [0.4]
    epochs = [20]
    layers = [1]
    lr = [0.0001]

    best_score = 0
    best_metrics = []

    for a in hidden_d1:
        for b in hidden_d2:
            for c in dropout:
                for d in epochs:
                    for e in layers:
                        for f in lr:
                            print(", hidden d1: ", a, ", hidden d2: ", b, ", dropout: ", c, ", epochs: ", d, ", lstm layers: ", e, ", lr: ", f)
                            model = UnimodalAudioModel(1611*3, a, b, e, c).double()
                            if torch.cuda.is_available():
                                model.cuda()

                            train(model, d, 100, optim.Adam(model.parameters(), lr=f, weight_decay=0.0003))

                            score = evaluate(model.cpu(), 100)

                            print(score)

                            if score > best_score:
                                torch.save(model, 'models/audio_model.pt')
                                best_metrics = [a, b, c, d, e, f]
                                best_score = score

    print(best_score)
    print(best_metrics)

    # model = UnimodalTextModel(768*3, 768, 300, 1, 0.2)
    # if torch.cuda.is_available():
    #     model.cuda()
    # train(20, 2, optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003))

    # pickle.dump(model, open('text_model.pkl', 'wb'))

    # model = pickle.load(open('text_model.pkl', 'rb'))

    # evaluate(model.cpu(), 2)
