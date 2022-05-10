"""
created on:2022/4/17 15:36
@author:caijianfeng
"""
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from data_process.dataset import PolSARDataset
from CNN import general_CNN
from CNN import genernal_CNN_mode

train_parameters = {
    "input_size": [1, 64, 64],  # 输入的shape
    "class_dim": 10,  # 分类数
    "target_path": "datas/",  # 数据集的路径
    "num_epochs": 50,  # 训练轮数
    "train_batch_size": 64,  # 批次的大小
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.001  # 超参数学习率
    }
}
# 参数初始化
target_path = train_parameters['target_path']
batch_size = train_parameters['train_batch_size']

print('--------------------train---------------------------')

# batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = PolSARDataset(data_path='../datas/TR.xlsx', label_path='../datas/label.xlsx', transform=transform)
print(type(train_dataset))
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

CNN_model = general_CNN.CNN()
device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
CNN_model.to(device=device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(params=CNN_model.parameters(), lr=train_parameters['learning_strategy']['lr'], momentum=0.5)
CNN_train = genernal_CNN_mode.CNN_train()
CNN_test = genernal_CNN_mode.CNN_test()

if __name__ == '__main__':
    cost = []
    accuracy = []
    for epoch in range(50):
        CNN_train.train(model=CNN_model,
                        epoch=epoch,
                        train_loader=train_loader,
                        device=device,
                        optimizer=optimizer,
                        criterion=criterion,
                        cost=cost)

    plt.plot(list(range(len(cost))), cost, 'r', label='CNN')
    plt.ylabel('loss for whole dataset')
    plt.xlabel('epoch')
    plt.grid()
    plt.show()

    # plt.plot(list(range(len(accuracy))), accuracy, 'r', label='CNN')
    # plt.ylabel('accuracy for CNN_test dataset')
    # plt.xlabel('epoch')
    # plt.grid()
    # plt.show()
