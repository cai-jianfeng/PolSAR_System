"""
created on:2022/4/15 11:25
@author:caijianfeng
"""
import torch


class CNN_train:
    def __init__(self):
        pass
    
    def train(self, model, epoch, train_loader, device, optimizer, criterion, cost):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            print('target.shape:', target.shape)
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # if batch_idx % 30 == 29:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 30))
            cost.append(running_loss)
            running_loss = 0.0


class CNN_test:
    def test(self, model, test_loader, device, accuracy):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print('Acuuracy on CNN_test set: %d %%' % (100 * correct / total))
            accuracy.append(100 * correct / total)
