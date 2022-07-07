"""
created on:2022/5/22 12:27
@author:caijianfeng
"""
from data_process.dataset_info import data_set_info

dataset_info = data_set_info(data_path='./data/prepro_flevoland4/pre_data/label.xlsx')

radio = dataset_info.data_ratio()

print(radio)
