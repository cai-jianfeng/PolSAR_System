"""
created on:2022/4/17 15:36
@author:caijianfeng
"""
import os.path
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from data_process.dataset import PolSARDataset
from CNN import general_CNN
from CNN import genernal_CNN_mode
from data_process.data_preprocess import Data

train_parameters = {
    "input_size": [9, 9, 9],  # 输入的shape
    "class_dim": 5,  # 分类数
    "data_path": '../data/prepro_flevoland4/pre_data/TR.xlsx',
    "label_path": '../data/prepro_flevoland4/pre_data/label.xlsx',
    "target_path": '../data_patch/T_R',  # 数据集的路径
    "num_epochs": 20,  # 训练轮数
    "train_batch_size": 1,  # 批次的大小
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.005  # 超参数学习率
    }
}
# 参数初始化
batch_size = train_parameters['train_batch_size']
data_path = train_parameters['data_path']
label_path = train_parameters['label_path']
target_path = train_parameters['target_path']
train_list_path = '../data_patch/T_R/train.txt'
eval_list_path = '../data_patch/T_R/eval.txt'
predict_list_path = '../data_patch/T_R/predict.txt'
patch_size = train_parameters['input_size'][1:3]

'''
划分训练集和验证集, 乱序, 生成数据列表
'''
if os.path.getsize(target_path) == 0:
    print('------------------数据生成开始------------------')
    # 每次生成数据列表前, 首先清空train.txt和eval.txt
    with open(train_list_path, 'w') as f:
        f.seek(0)  # 将当前文件的当前位置设置为偏移量
        f.truncate()
    with open(eval_list_path, 'w') as f:
        f.seek(0)
        f.truncate()  # 从当前位置截断
    
    # 分割数据集, 生成数据列表
    data = Data()
    data.save_data_label_segmentation_T_R(data_path=data_path,
                                          label_path=label_path,
                                          target_path=target_path,
                                          train_list_path=train_list_path,
                                          eval_list_path=eval_list_path,
                                          patch_size=patch_size,
                                          predict_list_path=predict_list_path)
print('--------------------train---------------------------')

# batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = PolSARDataset(data_path='../data_patch/T_R',
                              mode='train',
                              transform=transform)
test_dataset = PolSARDataset(data_path='../data_patch/T_R',
                             mode='eval',
                             transform=transform)
# print(type(train_dataset))
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

CNN_model = general_CNN.CNN()
device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
CNN_model.to(device=device)

# criterion = torch.nn.MultiLabelSoftMarginLoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(params=CNN_model.parameters(), lr=train_parameters['learning_strategy']['lr'], momentum=0.5)
CNN_train = genernal_CNN_mode.CNN_train()
CNN_test = genernal_CNN_mode.CNN_test()

if __name__ == '__main__':
    cost = []
    accuracy = []
    for epoch in range(train_parameters['num_epochs']):
        CNN_train.train(model=CNN_model,
                        epoch=epoch,
                        train_loader=train_loader,
                        device=device,
                        optimizer=optimizer,
                        criterion=criterion,
                        cost=cost)
        CNN_test.test(model=CNN_model,
                      test_loader=test_loader,
                      device=device,
                      accuracy=accuracy)
    # 保存模型参数
    torch.save(CNN_model.state_dict(), "./CNN_model_parameter2.pkl")
    
    plt.plot(list(range(len(cost))), cost, 'r', label='CNN')
    plt.ylabel('loss for whole dataset')
    plt.xlabel('num_data / batch_size * epoch')
    plt.grid()
    plt.show()
    plt.savefig('../plot/loss/loss_whole_Flevoland4_CNN.png')
    
    plt.plot(list(range(len(accuracy))), accuracy, 'r', label='CNN')
    plt.ylabel('accuracy for CNN_test dataset')
    plt.xlabel('epoch')
    plt.grid()
    plt.show()
    plt.savefig('../plot/accuracy/accuracy_whole_Flevoland4_CNN.png')
