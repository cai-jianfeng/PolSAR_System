import json
from torchvision import transforms
import torch
from CNN.genernal_CNN_mode import CNN_predict
from CNN.general_CNN_2 import CNN
from data_process.data_preprocess import Data
import os
CNN_model = CNN()
CNN_parameters = torch.load('../CNN/CNN_model_parameter_whole_dataset_1%_10_7x7.pkl')  # 加载训练好的模型参数
CNN_model.load_state_dict(CNN_parameters)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CNN_model.to(device=device)
data_path = '../data/prepro_flevoland4/pre_data/TR.xlsx'
data_paths = []
data = Data()
target_path = '../data_patch/T_R_7'
if os.path.getsize(target_path) == 0:
    print('------------------数据生成开始------------------')
    label_path = '../data/prepro_flevoland4/pre_data/label.xlsx'
    train_list_path = '../data_patch/T_R_7/train.txt'
    eval_list_path = '../data_patch/T_R_7/eval.txt'
    predict_list_path = '../data_patch/T_R_7/predict.txt'
    # 每次生成数据列表前, 首先清空train.txt和eval.txt
    with open(train_list_path, 'w') as f:
        f.seek(0)  # 将当前文件的当前位置设置为偏移量
        f.truncate()
    with open(eval_list_path, 'w') as f:
        f.seek(0)
        f.truncate()  # 从当前位置截断
    data.save_data_label_segmentation_T_R_without_label0(data_path=data_path,
                                          label_path=label_path,
                                          target_path=target_path,
                                          train_list_path=train_list_path,
                                          eval_list_path=eval_list_path,
                                          patch_size=(7, 7),
                                          predict_list_path=predict_list_path)
with open(os.path.join(target_path, 'predict.txt'), 'r', encoding='utf-8') as f:
    info = f.readlines()
for data_info in info:
    data_T_path, _ = data_info.strip().split('\t')
    data_paths.append(data_T_path)
print('数据路径读取完毕')
readme_json = '../data_patch/T_R_7/readme.json'
with open(readme_json, 'r') as f:
    data_info = json.load(f)
dim = data_info['dim']
predict_datas = data.get_data(data_paths=data_paths, dim=dim)
print('数据读取完毕')
target_path = '../data/prepro_flevoland4'
color_path = '../data/prepro_flevoland4/pre_data/color.xlsx'
label_pic_name = 'flevoland4_CNN_1%_10_7x7'
transform = transforms.Compose([
    transforms.ToTensor(),
])
predict_model = CNN_predict()
predict_model.predict(model=CNN_model,
                      predict_datas=predict_datas,
                      target_path=target_path,
                      color_path=color_path,
                      label_pic_name=label_pic_name,
                      transform=transform,
                      device=device)