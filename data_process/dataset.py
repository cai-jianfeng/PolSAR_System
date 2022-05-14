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
        self.data_sets = self.data.get_data_list(data_path=data_path)  # 三维数据(channel, row, column)
        self.labels = self.data.get_label_list(label_path=label_path)
        self.transform = transform
    
    def __getitem__(self, index):
        pixs = []
        rows = self.data_sets.shape[1]
        columns = self.data_sets.shape[2]
        row = (index - 1) // columns
        column = (index - 1) % columns
        for p in range(row - 2, row + 3):
            for q in range(column - 2, column + 3):
                pix = []
                for i in range(self.data_sets.shape[0]):
                    if 1400 > p >= 0 and 1200 > q >= 0:
                        pix.append(self.data_sets[i][p][q])
                    else:
                        pix.append(0)
                pix = np.array(pix)  # pix.shape: (9,)
                print('pix.shape:', pix.shape)
                pixs.append(pix)
        pixs = np.array(pixs).astype('float32')
        print('pixs.shape:', pixs.shape)
        pixs = pixs.reshape((5, 5, 9))
        print('pixs.shape:', pixs.shape)
        # pixs = torch.tensor(pixs, dtype=torch.float64)
        # pixs = np.array(pixs)
        pixs = self.transform(pixs)  # pixs_size: (9, 5, 5)
        label = self.labels[row][column]
        label = np.array([label], dtype="int64")
        print('label.shape:', label.shape)
        label = torch.from_numpy(label)
        return pixs, label
    
    def __len__(self):
        # print(self.labels.shape)
        return self.labels.shape[0] * self.labels.shape[1]
