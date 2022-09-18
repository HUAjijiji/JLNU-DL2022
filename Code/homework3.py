import random
import torch
import torch.nn as nn
import PIL.Image as Image  # Image 可以从图像读取数据

data = []
flag = []  # 期望值标签列表组成的矩阵
with open("d:/dataset/deeplearningclass/class.txt") as f:
    for line in f:  # 循环line次
        words = line.split(",")  # words为列表
        img = Image.open("d:/dataset/deeplearningclass/" + words[0])  # 打开图片装载进内存，这里第0项是文件名
        img = img.convert("L")  # convert 把原来的图像做转换，L为二值图像只有黑色白色 tips：原图片像素为三原色
        iml = list(img.getdata())  # 转换成列表  getdata从图里面把每一个像素的值取出
        data.append([iml])
        i = int(words[1])  # 取标签
        d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 初始化标签列表
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        d[i - 1] = 1.0
        flag.append([d])
# print(flag)
X = torch.tensor(data) / 255.0  # 把data数据转换成tensor，并转换成浮点数做归一化
# print(X)
S = torch.tensor(flag)  # 标签列表转换成tensor

'''X = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],  # 这个代表-1楼以此类推
                  [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                  [[1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]],
                  [[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]],
                  [[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
                  [[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]],
                  [[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]],
                  [[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]],
                  [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                  [[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]]])
# 以下是期望输出值，用二进制数表示
S = torch.tensor([[[0.0, 0.0, 0.0, 0.0]],
                  [[0.0, 0.0, 0.0, 1.0]],
                  [[0.0, 0.0, 1.0, 0.0]],
                  [[0.0, 0.0, 1.0, 1.0]],
                  [[0.0, 1.0, 0.0, 0.0]],
                  [[0.0, 1.0, 0.0, 1.0]],
                  [[0.0, 1.0, 1.0, 0.0]],
                  [[0.0, 1.0, 1.0, 1.0]],
                  [[1.0, 0.0, 0.0, 0.0]],
                  [[1.0, 0.0, 0.0, 1.0]]])

'''


class MyClass(nn.Module):  # 继承nn.moudle
    def __init__(self):
        super().__init__()  # super指代父类，这指示nn.Moudle   这里主要做初始化
        self.modle = nn.Sequential(  # Sequential可以按序执行
            nn.Linear(400, 50),
            nn.Sigmoid(),
            nn.Linear(50, 30),
            nn.Sigmoid()
        )
        self.loss = nn.MSELoss()  # MSEloss就是前面的误差(误差平方），self.loss现在是函数形式
        self.optmiser = torch.optim.SGD(self.parameters(), lr=0.05)  # SGD优化器，学习率选择为0.05
        self.count = 0  # 训练计数
        self.progress = []  # 进度表示需要存储的内容，用列表来表示

    def forward(self, input):  # 定义网络计算方法
        return self.modle(input)

    def train(self, input, target):  # 定义训练方法
        output = self.forward(input)
        myloss = self.loss(output, target)  # output,target为矩阵可能包含很多数据

        self.optmiser.zero_grad()  # 梯度清零
        myloss.backward()  # 用误差做反向传播计算
        self.optmiser.step()  # 调用优化器step方法进行梯度更新
        if self.count % 1000 == 0:
            self.progress.append(myloss.item())  # 每隔1000次记录一次误差，并添加到progress列表里
        self.count = self.count + 1


net = MyClass()  # 定义对象net
epoch = 600000
for j in range(epoch):  # 循环epoch次，这里为10000次
    i = random.randint(0, 29)
    net.train(X[i], S[i])
    ot = net.forward(X[i])
    loss = net.progress[-1]
    if j % 1000 == 0:
        # print(net.progress[-1])  # 输出损失列表的最后一项
        # print(net.forward(X[i]))
        print('i=', i, 'loss=', loss, 'final output is:', ot)

