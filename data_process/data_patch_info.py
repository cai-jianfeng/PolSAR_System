"""
created on:2022/5/28 14:47
@author:caijianfeng
"""
import json


class data_batch_infos:
    def __init__(self):
        pass
    
    def info_each_class_num_rate(self, path, readme_json):
        with open(path, 'r') as f:
            infos = f.readlines()
        with open(readme_json, 'r') as f:
            data_info = json.load(f)
        dim = data_info['dim']
        nums = dim[0] * dim[1]
        label_dict = dict()
        num = 0
        for info in infos:
            _, label = info.strip().split('\t')
            label_dict.setdefault(str(label), 0)
            label_dict[str(label)] += 1
            num += 1
        print(label_dict, '\n', 'num:', num)  # {'3': 2622, '4': 2865, '2': 2515, '1': 1310}
        rate = num / nums
        print('train_data rate:', rate)  # 0.005603397179888185
        return label_dict, num, rate
