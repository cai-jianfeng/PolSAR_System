"""
created on:2022/5/28 14:47
@author:caijianfeng
"""
train_path = '../data_patch/T_R/eval.txt'
with open(train_path, 'r') as f:
    infos = f.readlines()
label_dict = dict()
num = 0
for info in infos:
    _, label = info.strip().split('\t')
    label_dict.setdefault(str(label), 0)
    label_dict[str(label)] += 1
    num += 1
print(label_dict, '\n', 'num:', num)
