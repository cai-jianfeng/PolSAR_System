"""
created on:2022/5/26 22:35
@author:caijianfeng
"""
import xlrd
import numpy as np
import cv2

predict_path = '../data/prepro_flevoland4/predict_label.xlsx'
label_path = '../data/prepro_flevoland4/pre_data/label.xlsx'
color_path = '../data/prepro_flevoland4/pre_data/color.xlsx'
label_pic_name = 'flevoland4_CNN'
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
if predict_path:
    predict_sheets = xlrd.open_workbook(predict_path)
    labels_sheets = xlrd.open_workbook(label_path)
    predict_sheet = predict_sheets.sheet_by_index(0)
    labels_sheet = labels_sheets.sheet_by_index(0)
    rows = labels_sheet.nrows
    cols = labels_sheet.ncols
    print('label读取完成')
    num = 0
    R = np.zeros((rows, cols), dtype='uint8')
    G = np.zeros((rows, cols), dtype='uint8')
    B = np.zeros((rows, cols), dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            label = int(predict_sheet.cell_value(i, j))
            R[i][j] = colors[label][0]
            G[i][j] = colors[label][1]
            B[i][j] = colors[label][2]
            num += 1 if label == labels_sheet.cell_value(i, j) else 0
    print('acc:', num / (rows * cols))
    label_map = cv2.merge([B, G, R])
    cv2.imshow('label', label_map)
    cv2.waitKey(0)
    save_path = './' + mode + '/' + label_pic_name + '.jpg'
    cv2.imwrite(save_path, label_map)
