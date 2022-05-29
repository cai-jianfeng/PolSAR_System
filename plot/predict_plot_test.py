"""
created on:2022/5/27 20:20
@author:caijianfeng
"""
from plot.plot_mode import plot_mode
import xlrd
import numpy as np
import cv2

'''
绘制预测图
'''
# plot = plot_mode(mode='predict')
label_predcit_path = '../data/prepro_flevoland4/predict_labels_DCNN.xlsx'
label_path = '../data/prepro_flevoland4/pre_data/label.xlsx'
color_path = '../data/prepro_flevoland4/pre_data/color.xlsx'
label_pic = 'flevoland_predict_CNN_compared2'
mode = 'predict'
# plot.plot_labels(label_path=label_predcit_path,
#                  color_path=color_path,
#                  label_pic_name=label_pic)
colors = []
color_sheets = xlrd.open_workbook(color_path)
color_sheet = color_sheets.sheet_by_index(0)
rows = color_sheet.nrows
cols = color_sheet.ncols
for i in range(rows):
    color = [color_sheet.cell_value(i, 0), color_sheet.cell_value(i, 1), color_sheet.cell_value(i, 2)]
    colors.append(color)
print('颜色矩阵读取完成')
labels_predcit_sheets = xlrd.open_workbook(label_predcit_path)
labels_predcit_sheet = labels_predcit_sheets.sheet_by_index(0)
label_sheets = xlrd.open_workbook(label_path)
label_sheet = label_sheets.sheet_by_index(0)
rows = labels_predcit_sheet.nrows
cols = labels_predcit_sheet.ncols
num, nums = 0, 0
R = np.zeros((rows, cols), dtype='uint8')
G = np.zeros((rows, cols), dtype='uint8')
B = np.zeros((rows, cols), dtype='uint8')
for row in range(rows):
    for col in range(cols):
        label_predict = labels_predcit_sheet.cell_value(row, col)
        label = label_sheet.cell_value(row, col)
        # if label != 0:
        #     num += 1 if label_predict == label else 0
        #     nums += 1
        num += 1 if label_predict == label else 0
        nums += 1
        # label_predict = label_predict if label_predict != 0 else label
        label_predict = int(label_predict)
        label = int(label)
        if label_predict == 0:
            R[row][col] = colors[label][0] // 2
            G[row][col] = colors[label][1] // 2
            B[row][col] = colors[label][2] // 2
        else:
            R[row][col] = colors[label_predict][0]
            G[row][col] = colors[label_predict][1]
            B[row][col] = colors[label_predict][2]

acc = num / nums
print('acc:', acc)
label_map = cv2.merge([B, G, R])
cv2.imshow('label', label_map)
cv2.waitKey(0)
save_path = './' + mode + '/' + label_pic + '.jpg'
cv2.imwrite(save_path, label_map)
