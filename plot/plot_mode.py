"""
created on:2022/5/8 20:43
@author:caijianfeng
"""
import xlrd
import cv2
import numpy as np

colors = [[255, 255, 255]]


class plot_mode:
    def __init__(self, mode='ground'):
        """
        :param mode: ground: ground truth;
                     predict:predict.
        """
        self.mode = mode
    
    def plot_labels(self, laebl_path=None, color_path = None, label_pic_name = None):
        if color_path:
            color_sheets = xlrd.open_workbook(color_path)
            color_sheet = color_sheets.sheet_by_index(0)
            rows = color_sheet.nrows
            cols = color_sheet.ncols
            for i in range(rows):
                color = [color_sheet.cell_value(i, 0), color_sheet.cell_value(i, 1), color_sheet.cell_value(i, 2)]
                colors.append(color)
        if laebl_path:
            labels_sheets = xlrd.open_workbook(laebl_path)
            labels_sheet = labels_sheets.sheet_by_index(0)
            rows = labels_sheet.nrows
            cols = labels_sheet.ncols
            R = np.zeros((rows, cols), dtype='uint8')
            G = np.zeros((rows, cols), dtype='uint8')
            B = np.zeros((rows, cols), dtype='uint8')
            for i in range(rows):
                for j in range(cols):
                    label = int(labels_sheet.cell_value(i, j))
                    R[i][j] = colors[label][0]
                    G[i][j] = colors[label][1]
                    B[i][j] = colors[label][2]
            
            label_map = cv2.merge([B, G, R])
            cv2.imshow('label', label_map)
            cv2.waitKey(0)
            save_path = './' + self.mode + '/' + label_pic_name +'.jpg'
            cv2.imwrite(save_path, label_map)
