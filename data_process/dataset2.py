"""
created on:2022/6/2 23:29
@author:caijianfeng
"""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from data_process.data_preprocess import Data


class PolSARDataset(Dataset):
    def __init__(self, data_path=None, mode='train', transform=None):
        super(PolSARDataset, self).__init__()
        self.data = Data()
        self.data_path = data_path
        self.data_paths = []
        self.labels = []
        self.transform = transform
        if mode == 'train':
            with open(os.path.join(self.data_path, 'train.txt'), 'r', encoding='utf-8') as f:
                self.info = f.readlines()
            for data_info in self.info:
                data_T_path, label = data_info.strip().split('\t')
                self.data_paths.append(data_T_path)
                self.labels.append(int(label))
        elif mode == 'eval':
            with open(os.path.join(self.data_path, 'eval.txt'), 'r', encoding='utf-8') as f:
                self.info = f.readlines()
            for data_info in self.info:
                data_T_path, label = data_info.strip().split('\t')
                self.data_paths.append(data_T_path)
                self.labels.append(int(label))
    
    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data = self.data.get_data_list(data_path=data_path)
        data = self.data.data_dim_change(data)
        data = np.array(data).astype('float32')
        data = self.transform(data)
        label = self.labels[index] - 1
        label = torch.tensor(label, dtype=torch.int64)
        return data, label
    
    def __len__(self):
        return len(self.data_paths)
