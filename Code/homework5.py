import torch
import torch.nn as nn
import torchvision
import PIL.Image as Image
import math  # 主要做最后输出楼层向上取整
import random
import pandas
from torch.utils.data import Dataset
import numpy


class MyDataset(Dataset):
    def __init__(self, file):
        # 读取csv文件，头设置为空，df像一个二维数组,
        self.df = pandas.read_csv(file, header=None)
        # 因为在getitem和len都要用，需要做指定成类级别的变量
        pass

    def __getitem__(self, index):  # 取i行的数据
        data = self.df.iloc[index]  # 指定行数对于内容
        lables = data[0]  # 第0列为标签
        values = data[1:]  # 从第1列开始都是数值
        valtensorf = torch.tensor(values.values, dtype=torch.float32) / 255.0  # 数据的数值改为tensorfloat32并归一化
        target = torch.zeros((10))  # 生成一维的10个0
        target[lables] = 1.0  # lables位置为1.0
        return lables, valtensorf, target

    def __len__(self):  # 数据集的长度，
        return len(self.df)


class ResBlock(nn.Module):  # 定义残差块
    def __init__(self, n_channels):
        super().__init__()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(n_channels)

    def forward(self, x):  # 残差块实现过程
        x1 = self.conv2(x)
        r1 = torch.relu(self.bn(x1))
        x = x + r1
        return x


class Myconv(nn.Module):
    def __init__(self):
        super().__init__()  # 初始化
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(16)
        self.resBlocks = nn.Sequential(  # 序列形式执行
            *([ResBlock(16)] * 5)  # 变成列表乘以残差块个数
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out = nn.Linear(16 * 7 * 7, 10)  # 输入，输出

        self.loss = nn.MSELoss()  # MSEloss就是前面的误差(误差平方），self.loss现在是函数形式
        self.optmiser = torch.optim.SGD(self.parameters(), lr=0.05)  # SGD优化器，学习率选择为0.05
        self.count = 0  # 训练计数
        self.progress = []  # 进度表示需要存储的内容，用列表来表示

    def forward(self, x):  # 将输入矩阵转换为输出
        x = self.conv1(x)  #
        x = self.pool(x)
        x = self.resBlocks(x)
        x = self.pool2(x)
        x = x.view((16 * 7 * 7))  # 变成一纬然后送到fc层
        x = torch.sigmoid(self.out(x))
        # x = torch.relu(self.out(x))
        # x = torch.sigmoid(self.out(x))
        return x

    def train(self, input, target):  # 定义训练方法
        output = self.forward(input)
        myloss = self.loss(output, target)  # output,target为矩阵可能包含很多数据

        self.optmiser.zero_grad()  # 梯度清零
        myloss.backward()  # 用误差做反向传播计算
        self.optmiser.step()  # 调用优化器step方法进行梯度更新
        if self.count % 1000 == 0:
            self.progress.append(myloss.item())  # 每隔1000次记录一次误差，并添加到progress列表里
        self.count = self.count + 1


net = Myconv()  # 定义对象net
net.load_state_dict("d:/dataset/deeplearningclass/modeltrain.pth")  # 加载以前的模型

traindataset = MyDataset("d:/dataset/deeplearningclass/mnist_train.csv")  # 包含很多行的训练数据
testdataset = MyDataset("d:/dataset/deeplearningclass/mnist_test.csv")
count = 0
item = 0

# 训练并查看结果
for epoch in range(0, 3):
    for lables, valtensorf, target in traindataset:  # 每次都执行一次getitem
        valtensorf = valtensorf.view(1, 1, 28, 28)
        net.train(valtensorf, target)
        if net.count % 1000 == 0:
            print(net.progress[-1])
        if net.count % 10000 == 0:
            torch.save(net.state_dict(), "d:/dataset/deeplearningclass/modeltrain.pth")  # 每一万次保存一轮次

    # torch.save(net.state_dict(),"d:/dataset/deeplearningclass/model.pt")
# 测试集测试准确率
for lables, data, target in testdataset:
    output = net.forward(data.view((1, 1, 28, 28))).detach().numpy()
    if output.argmax() == lables:  # argmax数字最大的位置
        count = count + 1
    item = item + 1
print(count / item)  # 正确率
