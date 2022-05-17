"""
created on:2022/4/15 11:51
@author:caijianfeng
"""
import os
import random
import json

import numpy
import xlrd
import numpy as np
import xlsxwriter


class Data:
    def __int__(self):
        self.data_sets = None
        self.label_set = None
    
    def get_data_list(self, data_path):
        """
        获取指定path的数据
        :param data_path: 数据路径 --> str
        :return: 数据(三维:[channel, row, column]) --> numpy array
        """
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
        # self.data_sets = data_sets
        return data_sets
    
    def get_label_list(self, label_path):
        """
        获取指定路径的标签集
        :param label_path: 标签的path --> str
        :return: 标签集(二维数据:[row, column]) --> numpy array
        """
        sheets = xlrd.open_workbook(label_path)
        sheet = sheets.sheet_by_index(0)
        rows = sheet.nrows  # 获取sheet页的行数
        columns = sheet.ncols  # 获取sheet页的列数
        label_set = [[0 for _ in range(columns)] for _ in range(rows)]
        for row in range(rows):
            for column in range(columns):
                # cell = sheet.cell(row, column)  # 获取单元格
                # label = cell.value  # 获取单元格数据
                label = sheet.cell_value(row, column)  # 获取单元格数据
                label_set[row][column] = label
        
        label_set = np.array(label_set)
        # print(label_set.shape)  # 1400 * 1200
        # self.label_set = label_set
        return label_set

    def save_data_label_segmentation(self, data_path, label_path, target_path, train_list_path, eval_list_path, patch_size):
        """
        数据集切分（对一整张图像的数据进行切分成patch大小并保存）
        :param data_path: 原始数据(整张图像)的路径 --> str
        :param label_path: 原始标签(整张图像)的路径 --> str
        :param target_path: 分割后的数据集保存位置(文件夹) --> str
        :param train_list_path: 训练集说明文件 --> str
        :param eval_list_path: 测试集说明文件 --> str
        :param patch_size: 切分的数据块的大小([row, column]) --> tuple
        :return: None
        """
        if os.path.getsize(target_path) != 0:
            return True
        # if self.data_sets:  # 获取数据
        #     data_sets = self.data_sets
        # else:
        #     data_sets = self.get_data_list(data_path=data_path)
        data_sets = self.get_data_list(data_path=data_path)
        # if self.label_set:  # 获取标签
        #     label_set = self.label_set
        # else:
        #     label_set = self.get_label_list(label_path=label_path)
        label_set = self.get_label_list(label_path=label_path)
        dim = data_sets.shape  # 数据维度:(channel, row, column)
        num = 0
        train_list = []
        eval_list = []
        flag = 0
        for i in range(0, dim[1] - patch_size[0]):
            for j in range(0, dim[2] - patch_size[1]):
                target_paths = os.path.join(target_path, 'TR' + str(num) + '.xlsx')
                book = xlsxwriter.Workbook(filename=target_paths)
                for channel in range(dim[0]):
                    sheet = book.add_worksheet('sheet' + str(channel))
                    for row in range(patch_size[0]):
                        for column in range(patch_size[1]):
                            sheet.write(row, column, data_sets[channel][i + row][j + column])
                book.close()
                label = label_set[i + patch_size[0] // 2][j + patch_size[1] // 2]
                # print(num, ':', label)
                if num % 20 == 0:
                    eval_list.append(target_paths + '\t%d' % label + '\n')
                else:
                    train_list.append(target_paths + '\t%d' % label + '\n')
                num += 1
                if num == 100000:
                    flag = 1
                    break
            if flag == 1:
                break
        random.shuffle(eval_list)  # 打乱测试集
        with open(eval_list_path, 'a') as f:
            for eval_data in eval_list:
                f.write(eval_data)
        random.shuffle(train_list)  # 打乱训练集
        with open(train_list_path, 'a') as f:
            for train_data in train_list:
                f.write(train_data)
        print('生成数据列表完成！')
        return True
        
    def data_dim_change(self, data):
        dim = data.shape
        new_data = numpy.zeros((dim[1], dim[2], dim[0]))
        for channel in range(dim[0]):
            for row in range(dim[1]):
                for column in range(dim[2]):
                    new_data[row][column][channel] = data[channel][row][column]
        return new_data
        

