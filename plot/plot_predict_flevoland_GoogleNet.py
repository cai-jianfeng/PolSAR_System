"""
created on:2022/5/30 18:33
@author:caijianfeng
"""
from plot.plot_mode import plot_mode
from eval_target.results import result
import json

plot = plot_mode(mode='predict')
plot.plot_labels(label_path='../data/prepro_flevoland4/predict_labels_GoogleNet.xlsx',
                 color_path='../data/prepro_flevoland4/pre_data/color.xlsx',
                 label_pic_name='flevoland_GoogleNet_10%_10')
readme_json = '../data_patch/T_R/readme.json'
with open(readme_json, 'r') as f:
    data_info = json.load(f)
dim = data_info['dim']
target = result()
acc = target.accuracy_predict_txt(label_path='../data_patch/T_R/predict.txt',
                                  predict_path='../data/prepro_flevoland4/predict_labels_GoogleNet.xlsx',
                                  dim=dim)
print('acc:', acc)
