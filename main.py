import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


BATCH_SIZE = 100        
EPOCHS = 10                   # 总共训练批次
Lr = 0.005

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

# MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True)

test_dataset = torchvision.datasets.MNIST(root='data/',
    train=False,
    transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False)


class Detector(nn.Module):
   def __init__(self):
        super(Detector, self).__init__()
        self.layer1 = nn.Sequential(
            #卷积
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            #激活函数
            nn.ReLU(),
            #池化层
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            #卷积
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            #激活函数
            nn.ReLU(),
            #池化层
            nn.MaxPool2d(kernel_size=2, stride=2))
        #全连接层
        self.fc = nn.Linear(7 * 7 * 32, 10)
    
    
   def forward(self, x):
       #第一层
        out = self.layer1(x)
        #第二层
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #全连接
        out = self.fc(out)
        return out


model = Detector()
#选择损失函数
criterion = nn.CrossEntropyLoss()
#选择优化器
optimizer = optim.SGD(model.parameters(), lr = Lr)

#训练函数
def train(epoch):
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data

        outputs = model(inputs)
        #计算损失函数
        loss = criterion(outputs, target)

        #手动清空梯度
        optimizer.zero_grad()

        #反向传播计算梯度
        loss.backward()
        optimizer.step()

#测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)

            _, predicted = torch.max(outputs.data, dim = 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the  test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    i = 0
    for epoch in range(EPOCHS):
        train(epoch)
        test()
        i += 1
        print("当前任务进度为" , i)
    #保存模型参数
    torch.save(model.state_dict(), "/home/rosen/桌面/Rosen/Pytorch/model.pkl")







