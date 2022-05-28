"""
created on:2022/5/28 14:40
@author:caijianfeng
"""
import xlrd

book = xlrd.open_workbook('../data/prepro_flevoland4/pre_data/label.xlsx')
sheet = book.sheet_by_index(0)
rows = sheet.nrows
cols = sheet.ncols
label_dict = dict()
for row in range(rows):
    for col in range(cols):
        label = sheet.cell_value(row, col)
        label_dict.setdefault(str(label), 0)
        label_dict[str(label)] += 1
print(label_dict)