"""
created on:2022/4/15 11:51
@author:caijianfeng
"""
import os
import random
import json
import xlrd
import numpy as np


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
                    cell = sheet.cell(row, column)  # 获取单元格
                    data = cell.value  # 获取单元格数据
                    data_set[row][column] = data
            data_sets.append(data_set)
        data_sets = np.array(data_sets)
        # dim = data_sets.shape[0]
        SPAN = np.array([[0 for _ in range(columns)] for _ in range(rows)])
        for i in [0, 4, 8]:
            SPAN = SPAN + data_sets[i]  # SPAN = T11 + T22 + T33
        print(data_sets.shape)  # 9 * 1400 * 1200
        return data_sets
    
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
        
        
        