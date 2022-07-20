"""
created on:2022/7/20 14:50
@author:caijianfeng
"""
from plot.plot_mode import plot_mode
from eval_target.results import result
import json

plot = plot_mode(mode='predict')
plot.plot_labels(label_path='../data/prepro_flevoland4/predict_labels_CNN_1%_10_7x7.xlsx',
                 color_path='../data/prepro_flevoland4/pre_data/color.xlsx',
                 label_pic_name='flevoland_CNN_1%_10_7x7')
readme_json = '../data_patch/TR/readme.json'
with open(readme_json, 'r') as f:
    data_info = json.load(f)
dim = data_info['dim']
target = result()
acc = target.accuracy_predict_txt(label_path='../data_patch/TR/predict.txt',
                                  predict_path='../data/prepro_flevoland4/predict_labels_CNN_1%_10_7x7.xlsx',
                                  dim=dim)
print('acc:', acc)
