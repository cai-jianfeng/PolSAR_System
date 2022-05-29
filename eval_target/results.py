"""
created on:2022/5/29 20:45
@author:caijianfeng
"""
import xlrd

class result:
    def __init__(self):
        pass
    
    def accuracy_predict_txt(self, label_path, predict_path, dim):
        with open(label_path, 'r') as f:
            info = f.readlines()
        predict_book = xlrd.open_workbook(predict_path)
        predict_sheet = predict_book.sheet_by_index(0)
        rows = predict_sheet.nrows
        cols = predict_sheet.ncols
        num ,totoal = 0, 0
        nums = 0
        for row in range(rows):
            for col in range(cols):
                label = int(info[nums].strip().split('\t')[1])
                predict = predict_sheet.cell_value(row, col)
                # if label != 0:
                #     num += 1 if label == predict else 0
                #     totoal += 1
                num += 1 if label == predict else 0
                nums += 1
        acc = num / nums
        # acc = num / totoal
        return acc
    
    def accuracy_readme_json(self, label_path, predict_path, dim):
        label_book = xlrd.open_workbook(label_path)
        predict_book = xlrd.open_workbook(predict_path)
        label_sheet = label_book.sheet_by_index(0)
        predict_sheet = predict_book.sheet_by_index(0)
        rows = predict_sheet.nrows
        cols = predict_sheet.ncols
        num = 0
        for row in range(rows):
            for col in range(cols):
                label = label_sheet.cell_value(row + dim[0] // 2, col + dim[1] // 2)
                predict = predict_sheet.cell_value(row, col)
                num += 1 if label == predict else 0
        acc = num / (rows * cols)
        return acc
        