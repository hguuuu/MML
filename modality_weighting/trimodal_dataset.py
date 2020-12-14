import torch
import numpy as numpy
from torch.utils.data import Dataset, DataLoader
import pickle

# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

class TrimodalDataset(Dataset):
    def __init__(self, text, audio, visual):
        self.text_embeddings = pickle.load(open(text, 'rb'))
        self.audio_embeddings = pickle.load(open(audio, 'rb'))
        self.visual_embeddings = pickle.load(open(visual, 'rb'))
    
    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = {}
        for key in self.visual_embeddings[idx]:
            if key == "embedding":
                item['embedding'] = torch.cat([self.text_embeddings[idx]['bert_embedding'], self.audio_embeddings[idx][key], self.visual_embeddings[idx][key].flatten(0, 1)])
            else:
                item[key] = self.text_embeddings[idx][key]

        return item