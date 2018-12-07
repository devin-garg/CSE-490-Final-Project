import torch
import os
from torch.utils import data
from torchvision import transforms
import PIL
import multiprocessing
import librosa
import numpy as np
####INPUTS To Change####
trainPath = 'nsynth-valid/audio/'
testPath = 'nsynth-test/audio/'
####

class Dataset(data.Dataset):
    def __init__(self, path, transform):
        self.ids = []
        self.path = path
        self.transform = transform
        self.labelDict = {'keyboard electronic': 0, 'flute synthetic': 1, 'organ electronic': 2, 'keyboard synthetic': 3, 'bass synthetic': 4, 'string acoustic': 5, 'mallet acoustic': 6, 'guitar acoustic': 7, 'guitar electronic': 8, 'brass acoustic': 9, 'reed acoustic': 10, 'bass electronic': 11, 'vocal synthetic': 12, 'keyboard acoustic': 13, 'vocal acoustic': 14, 'flute acoustic': 15}
        for f in os.listdir(path):
            if not f.startswith('.'):
                self.ids.append(f)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        y, sr = librosa.load(self.path + self.ids[index])
        X = librosa.feature.melspectrogram(y=y, sr=sr)
        #print(X.shape)
        X = X/np.max(X)*255
        #print(X.shape)
        X = np.expand_dims(X, axis=-1).astype(np.float32)
        #print(X.shape)
        if self.transform:
            X = self.transform(X)

        label = ' '.join(self.ids[index].split("_")[0:2])
        label = self.labelDict[label]
        return (X, label)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])
data_train = Dataset(trainPath, transform=train_transforms)
data_test = Dataset(testPath, transform=test_transforms)
BATCH_SIZE = 128
TEST_BATCH_SIZE = 10
USE_CUDA = True
use_cuda = USE_CUDA and torch.cuda.is_available()
kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                           shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                          shuffle=False, **kwargs)
