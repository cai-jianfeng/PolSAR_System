"""
created on:2022/5/28 14:40
@author:caijianfeng
"""
import xlrd


class data_set_info:
    def __init__(self, data_path):
        self.book = xlrd.open_workbook(data_path)
        self.sheet = self.book.sheet_by_index(0)
        self.rows = self.sheet.nrows
        self.cols = self.sheet.ncols
        self.kind_num = None
        self.ratio = None
        self.kinds = None
    
    def data_ratio(self):
        """
        统计各个类别数据的比例
        :return: dict
        """
        if self.ratio:
            return self.ratio
        else:
            self.ratio = dict()
            self.kind_num = self.data_kind_num()
            self.kinds = len(self.kind_num)
            for kind in self.kind_num:
                self.ratio.setdefault(kind, 0)
                self.ratio[kind] = (self.kind_num[kind] / (self.rows * self.cols))
            return self.ratio
    
    def data_kind_num(self):
        """
        统计各个类别数据的个数
        :return: dict
        """
        if self.kind_num:
            return self.kind_num
        else:
            self.kind_num = dict()
            for row in range(self.rows):
                for col in range(self.cols):
                    label = self.sheet.cell_value(row, col)
                    self.kind_num.setdefault(str(int(label)), 0)
                    self.kind_num[str(int(label))] += 1
            return self.kind_num
    
    def kinds_num(self):
        """
        统计dataset的分类数
        :return: int
        """
        if self.kinds:
            return self.kinds
        else:
            return len(self.data_kind_num())
