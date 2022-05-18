"""
created on:2022/5/17 23:08
@author:caijianfeng
"""
import numpy
import torch
from torch import nn
import torch.nn.functional as F

'''双通道 CNN'''


class Double_CNN(nn.Module):
    def __init__(self):
        super(Double_CNN, self).__init__()
        self.conv1 = nn.Conv2d(9, 16, kernel_size=3, padding=2, dtype=torch.float32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=2, dtype=torch.float32)
        self.conv3 = nn.Conv2d(9, 16, kernel_size=3, padding=2, dtype=torch.float32)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=2, dtype=torch.float32)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(576, 5)
    
    def forward(self, x):
        in_size = x.size(0)
        x1 = self.mp(F.relu(self.conv1(x[:, 0:9])))
        x1 = self.mp(F.relu(self.conv2(x1)))
        x2 = self.mp(F.relu(self.conv3(x[:, 9:18])))
        x2 = self.mp(F.relu(self.conv4(x2)))
        x = torch.cat((x1.view(in_size, -1), x2.view(in_size, -1)), 1)
        x = self.fc(x)
        return x
