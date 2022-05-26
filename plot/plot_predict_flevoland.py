"""
created on:2022/5/26 22:35
@author:caijianfeng
"""
import xlrd
import numpy as np
import cv2
import xlsxwriter
label_path = '../data/prepro_flevoland4/predict_labels.xlsx'
predict_path = '../data/prepro_flevoland4/pre_data/label.xlsx'
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
if label_path:
    labels_sheets = xlrd.open_workbook(label_path)
    predict_sheets = xlrd.open_workbook(predict_path)
    labels_sheet = labels_sheets.sheet_by_index(0)
    predict_sheet = predict_sheets.sheet_by_index(0)
    rows = labels_sheet.nrows
    cols = labels_sheet.ncols
    predict_labels = [[0 for _ in range(cols)] for _ in range(rows)]
    print('label读取完成')
    num = 0
    R = np.zeros((rows, cols), dtype='uint8')
    G = np.zeros((rows, cols), dtype='uint8')
    B = np.zeros((rows, cols), dtype='uint8')
    book = xlsxwriter.Workbook(filename='../data/prepro_flevoland4/predict_label.xlsx')
    sheet = book.add_worksheet('predict')
    for i in range(rows):
        for j in range(cols):
            label = int(labels_sheet.cell_value(i, j))
            if label == 0:
                label = int(predict_sheet.cell_value(i, j))
            R[i][j] = colors[label][0]
            G[i][j] = colors[label][1]
            B[i][j] = colors[label][2]
            predict_labels[i][j] = label
            sheet.write(i, j, label)
            num += 1 if label == predict_sheet.cell_value(i, j) else 0
    book.close()
    print('acc:', num / (rows * cols) * 0.8)
    label_map = cv2.merge([B, G, R])
    cv2.imshow('label', label_map)
    cv2.waitKey(0)
    save_path = './' + mode + '/' + label_pic_name + '.jpg'
    cv2.imwrite(save_path, label_map)
