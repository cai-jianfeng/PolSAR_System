"""
created on:2022/5/6 21:44
@author:caijianfeng
"""
import cv2
# import xlrd
import numpy as np
from data_process.data_preprocess import Data

# label_workbook = xlrd.open_workbook('../datas/label.xlsx')
# label_sheet = label_workbook.sheet_by_index(0)
label_path = '../Flevoland4_data/label.xlsx'
data_pro = Data()
labels = data_pro.get_label_list(label_path=label_path)
# rows = label_sheet.nrows
# columns = label_sheet.ncols
rows = len(labels)
columns = len(labels[0])
print('rows:', rows, 'columns:', columns)
R = np.zeros((rows, columns), dtype="uint8")
G = np.zeros((rows, columns), dtype="uint8")
B = np.zeros((rows, columns), dtype="uint8")
for i in range(rows):
    for j in range(columns):
        # label = label_sheet.cell(i, j).value
        label = labels[i][j]
        # print(label)
        if label == 1:
            R[i][j] = 255
        elif label == 3:
            G[i][j] = 255
        elif label == 2:
            B[i][j] = 255
        elif label == 4:
            R[i][j] = 255
            G[i][j] = 255
            B[i][j] = 255

colors = cv2.merge([B, G, R])
cv2.imshow('labels', colors)
cv2.waitKey(0)
cv2.imwrite('./ground/Flevoland4_label.jpg', colors)  # 保存图片
