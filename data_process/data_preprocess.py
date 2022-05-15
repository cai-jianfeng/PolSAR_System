"""
created on:2022/4/15 11:51
@author:caijianfeng
"""
import os
import random
import json
import xlrd
import numpy as np
import xlsxwriter


class Data:
    def __int__(self):
        pass
    
    def get_data_list(self, data_path):
        sheets = xlrd.open_workbook(data_path)
        sheets_num = sheets.nsheets
        # sheets_names = sheets.sheet_names() # 获取所有sheet的名字
        data_sets = []  # 三维数据
        for num in range(sheets_num):
            sheet = sheets.sheet_by_index(num)
            rows = sheet.nrows  # 获取sheet页的行数
            columns = sheet.ncols  # 获取sheet页的列数
            data_set = [[0 for _ in range(columns)] for _ in range(rows)]  # 二维数据
            for row in range(rows):
                for column in range(columns):
                    # cell = sheet.cell(row, column)  # 获取单元格
                    # data = cell.value  # 获取单元格数据
                    data = sheet.cell_value(row, column)  # 获取单元格数据
                    data_set[row][column] = data
            data_sets.append(data_set)
        data_sets = np.array(data_sets)
        # dim = data_sets.shape[0]
        # SPAN = np.array([[0 for _ in range(columns)] for _ in range(rows)])
        # for i in [0, 4, 8]:
        #     SPAN = SPAN + data_sets[i]  # SPAN = T11 + T22 + T33
        # print(data_sets.shape)  # 9 * 1400 * 1200
        return data_sets
    
    def save_data_segmentation(self, data_path, patch_size):
        """
        数据集切分（对一整张图像的数据进行切分成patch并保存）
        :param data_path: 原始数据(整张图像)的路径
        :param patch_size: 切分的数据块的大小([row, column])
        :return: None
        """
        data_sets = self.get_data_list(data_path=data_path)  # 获取数据
        dim = data_sets.shape  # 数据维度:(channel, row, column)
        num = 0
        for channel in range(dim[0]):
            for i in range(0, dim[1] - patch_size[0]):
                for j in range(0, dim[2] - patch_size[1]):
                    book = xlsxwriter.Workbook(filename='./data_patch/data_TR' + str(num) + '.xlsx')
                    sheet = book.add_worksheet()
                    for row in range(patch_size[0]):
                        for column in range(patch_size[1]):
                            sheet.write(row, column, data_sets[channel][i + row][j + column])
                    book.close()
                    num += 1
    
    def get_label_list(self, label_path):
        sheets = xlrd.open_workbook(label_path)
        sheet = sheets.sheet_by_index(0)
        rows = sheet.nrows  # 获取sheet页的行数
        columns = sheet.ncols  # 获取sheet页的列数
        label_set = [[0 for _ in range(columns)] for _ in range(rows)]
        for row in range(rows):
            for column in range(columns):
                cell = sheet.cell(row, column)  # 获取单元格
                label = cell.value  # 获取单元格数据
                label_set[row][column] = label
        
        label_set = np.array(label_set)
        # print(label_set.shape)  # 1400 * 1200
        return label_set
