import torch
import os
from torch.utils import data
class Dataset(data.Dataset):
    def __init__(self, path, transform):
        self.ids = []
        self.label = path
        self.transform = transform
        for f in os.listdir(path):            
            if not f.startswith('.'):
                self.ids.append(f)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        y, sr = librosa.load(label + "/" + self.ids[index])
        X = librosa.feature.melspectrogram(y=y, sr=sr)
        if self.transform is not None:
            X = self.transform(X)
        label = self.label
        return X, label
        
