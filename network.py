import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim


best_acc1=0
#num_of_classes=200

class InstrumentNet(nn.Module):
    def __init__(self):
        super(InstrumentNet, self).__init__()
        # TODO define the layers
        #raise NotImplementedError('Need to define the layers for your network')
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64,64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256,512, kernel_size=3, padding=1)
        self.bn45 = nn.BatchNorm2d(512)

        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(1024)

        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4096, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048,1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_of_classes)

    def forward(self, x):
        # TODO define the forward pass
        #raise NotImplementedError('Need to define the forward pass')

        x = self.bn1(F.relu(self.conv1_1(x)))

        x = self.bn2(F.max_pool2d(F.relu(self.conv1_2(x)),2,2))
        #print(x.shape)
        #print(x.shape)
        x = self.bn3(F.max_pool2d(F.relu(self.conv2_1(x)),2,2))

        #print(x.shape)
        x = self.bn4(F.max_pool2d(F.relu(self.conv3_1(x)),2,2))
        #print(x.shape)

        x = self.bn45(F.max_pool2d(F.relu(self.conv4(x)),2,2))
        x = self.bn5(F.max_pool2d(F.relu(self.conv4_1(x)),2,2))
        x = self.bn7(F.max_pool2d(F.relu(self.conv5_1(x)),2,2))

        #print(x.shape)

        x = x.view(-1,4096)
        #print(x.shape)
        x = self.bn6(F.relu(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.bn8(F.relu(self.fc2(x)))
        x= self.fc3(x)
        #print(x.shape)

        return x

    def loss(self, prediction, label, reduction='elementwise_mean'):

        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        global best_acc1

        if (accuracy > best_acc1):
          best_acc1= accuracy
          pt_util.save(self,file_path,num_to_keep)
        # TODO save the model if it is the best
        #raise NotImplementedError('Need to implement save_best_model')

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)
