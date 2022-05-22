"""
created on:2022/5/10 14:47
@author:caijianfeng
"""
from plot.plot_mode import plot_mode

plots = plot_mode(mode='ground')
plots.plot_labels(laebl_path='../data/prepro_flevoland_farm/pre_data/label.xlsx',
                  color_path='../data/prepro_flevoland_farm/pre_data/color.xlsx',
                  label_pic_name='Flevoland_farm_label')
