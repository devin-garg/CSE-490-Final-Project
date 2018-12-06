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
        for f in os.listdir(path):
            if not f.startswith('.'):
                self.ids.append(f)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        y, sr = librosa.load(self.path + self.ids[index])
        X = librosa.feature.melspectrogram(y=y, sr=sr)
        X = X/np.max(X)*255
        X = np.expand_dims(X, axis=-1).astype(np.float32)
        print(X.shape)
        if self.transform:
            X = self.transform(X)

        label = ' '.join(self.ids[index].split("_")[0:2])
        return X, label
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
