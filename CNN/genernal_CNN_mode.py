"""
created on:2022/4/15 11:25
@author:caijianfeng
"""
import torch
from plot.plot_mode import plot_mode
import xlrd
import xlsxwriter
import os
import numpy as np


class CNN_train:
    def __init__(self):
        pass
    
    def train(self, model, epoch, train_loader, device, optimizer, criterion, cost):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            # print('target.shape:', target.shape)
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            # running_loss += loss.item()
            running_loss += loss.item()
            if batch_idx % 30 == 29:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 30))
                cost.append(running_loss / 30)
                running_loss = 0.0
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss))
            # cost.append(running_loss)
            # running_loss = 0.0


class CNN_test:
    def test(self, model, test_loader, device, accuracy):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                print('img_shape:', images.shape)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print('Acuuracy on NN_test set: %d %%' % (100 * correct / total))
            accuracy.append(100 * correct / total)


class CNN_predict:
    def predict(self, model, predict_datas, target_path, color_path, label_pic_name, transform, device):
        rows = predict_datas.shape[0]
        columns = predict_datas.shape[1]
        predict_labels = [[0 for _ in range(columns)] for _ in range(rows)]
        target_path = os.path.join(target_path, 'predict_labels' + '.xlsx')
        book = xlsxwriter.Workbook(filename=target_path)
        sheet = book.add_worksheet('sheet')
        for row in range(rows):
            for column in range(columns):
                predict_data = predict_datas[row][column]
                # predict_data = transform(predict_data)
                predict_data = np.expand_dims(predict_data, 0)
                predict_data = torch.tensor(predict_data, dtype=torch.float32)
                predict_data = predict_data.to(device)
                predict_label = model(predict_data)
                predict_label = torch.max(predict_label.data, dim=1)
                print('predict_label:', predict_label)
                predict_labels[row][column] = predict_label
                sheet.write(row, column, predict_label.values)
        book.close()
        plot_mod = plot_mode(mode='predict')
        plot_mod.plot_labels(label_path=target_path,
                             color_path=color_path,
                             label_pic_name=label_pic_name)
                
        
        
        