"""
created on:2022/4/15 11:46
@author:caijianfeng
"""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from data_process.data_preprocess import Data


class PolSARDataset(Dataset):
    def __init__(self, data_path=None, label_path=None, mode='train', transform=None):
        super(PolSARDataset, self).__init__()
        self.data = Data()
        self.data_sets = self.data.get_data_list(data_path=data_path)
        self.labels = self.data.get_label_list(label_path=label_path)
        self.transform = transform
    
    def __getitem__(self, index):
        pix = []
        rows = self.data_sets.shape[1]
        columns = self.data_sets.shape[2]
        for i in range(self.data_sets.shape[0]):
            pix.append(self.data_sets[i][(index - 1) // columns][(index - 1) % columns])
            label = self.labels[(index - 1) // columns][(index - 1) % columns]
            pix = np.array(pix)
            pix = self.transform(pix)
        return pix, label
    
    def __len__(self):
        # print(self.labels.shape)
        return self.labels.shape[0] * self.labels.shape[1]
