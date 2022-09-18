import random
import torch

# 10个输入,用七位数字代表一个楼层号码
X = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],  # 这个代表-1楼以此类推
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

lr = 0.01
W = torch.rand((7, 7), requires_grad=True)  # 7：输入数量，第二个7：中间节点的数量
V = torch.rand((7, 4), requires_grad=True)  # 7：输入数量，9：输出数量
epoch = 1000000  # 循环遍历一百万次
for j in range(0, epoch):
    i = random.randint(0, 9)  # 从0-9索引

    Y = torch.sigmoid(torch.mm(X[i], W))  # mm是矩阵乘以矩阵 ，m=matrix
    Z = torch.sigmoid(torch.mm(Y, V))
    e = (S[i] - Z) * (S[i] - Z)
    e = e[0][0] * 4 + e[0][1] * 2 + e[0][2] * 1 + e[0][3] * 1.6  # 确保误差为标量
    if j % 10000 == 0 and i != 0 or j > epoch - 10:
        print(f'This is the {i}th floor,now error is {e:.3f},the expected output is:{S[i]},the actual output is:{Z}.')
    elif j % 10000 == 0 and i == 0 or j > epoch - 10:
        print(
            f'This is the {i - 1}th floor,now error is {e:.3f},the expected output is:{S[i]},the actual output is:{Z}.')
    e.backward()  # 反向传播
    W.data = W.data - lr * W.grad  # 更新权重
    V.data = V.data - lr * V.grad
    W.grad.zero_()  # 梯度清零
    V.grad.zero_()
