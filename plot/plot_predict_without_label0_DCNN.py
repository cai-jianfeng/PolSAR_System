"""
created on:2022/5/30 20:58
@author:caijianfeng
"""
import numpy as np
import xlrd
import cv2

pre_path = '../data/prepro_flevoland4/predict_labels_DCNN.xlsx'
color_path = '../data/prepro_flevoland4/pre_data/color.xlsx'
label_path = '../data_patch/T_R/predict.txt'
label_pic_name = 'flevoland_DCNN_10%_10_without_label0'
mode = 'predict'
colors = []
if color_path:
    color_sheets = xlrd.open_workbook(color_path)
    color_sheet = color_sheets.sheet_by_index(0)
    rows = color_sheet.nrows
    cols = color_sheet.ncols
    for i in range(rows):
        color = [color_sheet.cell_value(i, 0), color_sheet.cell_value(i, 1), color_sheet.cell_value(i, 2)]
        colors.append(color)
    print('颜色矩阵读取完成')
with open(label_path, 'r') as f:
    info = f.readlines()
if pre_path:
    pre_sheets = xlrd.open_workbook(pre_path)
    pre_sheet = pre_sheets.sheet_by_index(0)
    rows = pre_sheet.nrows
    cols = pre_sheet.ncols
    print('label读取完成')
    R = np.zeros((rows, cols), dtype='uint8')
    G = np.zeros((rows, cols), dtype='uint8')
    B = np.zeros((rows, cols), dtype='uint8')
    num = 0
    accuarcy_num = 0
    nums = 0
    for i in range(rows):
        for j in range(cols):
            pre = int(pre_sheet.cell_value(i, j))
            label = int(info[num].strip().split('\t')[1])
            if label == 0:
                pre = 0
            else:
                accuarcy_num += 1 if pre == label else 0
                nums += 1
            R[i][j] = colors[pre][0]
            G[i][j] = colors[pre][1]
            B[i][j] = colors[pre][2]
            num += 1
    
    label_map = cv2.merge([B, G, R])
    cv2.imshow('label', label_map)
    cv2.waitKey(0)
    save_path = './' + mode + '/' + label_pic_name + '.jpg'
    cv2.imwrite(save_path, label_map)
    print('acc:', accuarcy_num / nums)