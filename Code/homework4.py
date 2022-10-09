import torch
import torch.nn as nn
import torchvision
import PIL.Image as Image
import math  # 主要做最后输出楼层向上取整
import random
data = []
flag = []  # 期望值标签列表组成的矩阵
with open("d:/dataset/deeplearningclass/class.txt") as f:
    for line in f:  # 循环line次
        words = line.split(",")  # words为列表
        img = Image.open("d:/dataset/deeplearningclass/" + words[0])  # 打开图片装载进内存，这里第0项是文件名
        img = img.convert("L")  # convert 把原来的图像做转换，L为二值图像只有黑色白色 tips：原图片像素为三原色
        imdl = list(img.getdata())  # 转换成列表  getdata从图里面把每一个像素的值取出
        data.append([imdl])
        i = int(words[1])  # 取标签
        d = [0.0] * 10  # 初始化标签列表
        d[i - 1] = 1.0
        flag.append([d])
# print(data)
dt = torch.tensor(data)  # 列表转换为张量
print(dt.size())
x = dt.view((30, 1, 20, 20))  # view改变维度
X = x / 255.0  # 转换为0-1之间浮点数
S = torch.tensor(flag)  # 标签列表转换成tensor
# print(X)
# print(len(X))


class Myconv(nn.Module):
    def __init__(self):
        super().__init__()  # 初始化
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out = nn.Linear(16 * 5 * 5, 10)  # 输入，输出

        self.loss = nn.MSELoss()  # MSEloss就是前面的误差(误差平方），self.loss现在是函数形式
        self.optmiser = torch.optim.SGD(self.parameters(), lr=0.05)  # SGD优化器，学习率选择为0.05
        self.count = 0  # 训练计数
        self.progress = []  # 进度表示需要存储的内容，用列表来表示

    def forward(self, x):  # 将输入矩阵转换为输出
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view((16 * 5 * 5))  # 变成一纬然后送到fc层
        x = torch.sigmoid(self.out(x))
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


image = Image.open("d:/dataset/deeplearningclass/0-1.png")  # 20*20
toPilImage = torchvision.transforms.ToPILImage()  # topilimage处理图片
image = image.convert("L")     # l 转换为灰度图像
data = list(image.getdata())     # getdata获取图像数据，按行读取，像素
print(len(data))
d = torch.tensor(data)    # 列表转换为张量
x = d.view((1, 1, 20, 20))  # view改变维度
x = x / 255.0   # 转换为0-1之间浮点数


net = Myconv()  # 定义对象net
epoch = 100000
for j in range(epoch):  # 循环epoch次，这里为100000次
    i = random.randint(0, 29)
    net.train(X[i], S[i])
    ot = net.forward(X[i])
    loss = net.progress[-1]
    tag = math.ceil((i + 1) / 3)  # 输出的时候对应的标签值，为了方便观察输出结果
    if j % 1000 == 0:
        print('i=', i, 'tag：', tag, 'loss=', loss, 'final output is:', ot)
