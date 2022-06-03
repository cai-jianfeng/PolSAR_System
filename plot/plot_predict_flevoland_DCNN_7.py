"""
created on:2022/6/2 18:23
@author:caijianfeng
"""
from plot.plot_mode import plot_mode
from eval_target.results import result
import json

plot = plot_mode(mode='predict')
plot.plot_labels(label_path='../data/prepro_flevoland4/predict_labels_DCNN_7.xlsx',
                 color_path='../data/prepro_flevoland4/pre_data/color.xlsx',
                 label_pic_name='flevoland_DCNN_10%_10_7x7')
readme_json = '../data_patch/T_RI_7/readme.json'
with open(readme_json, 'r') as f:
    data_info = json.load(f)
dim = data_info['dim']
target = result()
acc = target.accuracy_predict_txt(label_path='../data_patch/T_RI_7/predict.txt',
                                  predict_path='../data/prepro_flevoland4/predict_labels_DCNN_7.xlsx',
                                  dim=dim)
print('acc:', acc)
