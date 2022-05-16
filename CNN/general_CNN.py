"""
created on:2022/4/14 23:02
@author:caijianfeng
"""
import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(9, 16, kernel_size=3, padding=2, dtype=torch.float32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,padding=2, dtype=torch.float32)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=6, dtype=torch.float32)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=5, dtype=torch.float32)
        # self.conv5 = nn.Conv2d(128, 10, kernel_size=3)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(288, 10)
    
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.conv2(x)))
        # x = self.mp(F.relu(self.conv3(x)))
        # x = self.mp(F.relu(self.conv4(x)))
        # x = self.mp(F.relu(self.conv5(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
