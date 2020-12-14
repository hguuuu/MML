# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import torch
import numpy as numpy
from torch.utils.data import Dataset, DataLoader
import pickle

class UnimodalDataset(Dataset):
    def __init__(self, split):
        self.embeddings = pickle.load(open(split, 'rb'))
    
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.embeddings[idx]
