# !usr/bin/python
# -*_ coding:utf-8 -*-

"""A line regress implement by pytorch high api"""

__author__ = "huluwa-2020-04-09"

import torch
import numpy as np
import matplotlib.pyplot as plt


def make_dataset(samples_num, weights, bias):
    """
    制作数据集
    :param samples_num:
    :param weights:
    :param bias:
    :return:
    """
    features = torch.from_numpy(np.random.normal(0, 1, (samples_num, len(weights))).astype(np.float32))
    labels = torch.zeros(samples_num)

    for i, w in enumerate(weights):
        labels += w * features[:, i]

    labels += bias  # 广播机制
    labels += torch.from_numpy(np.random.normal(0, 0.01, samples_num))  # 添加噪声

    return features, labels


# 生产数据集
sample_nums = 1000
weights_true = [2, -3.4]
bias_true = 4.2

features, labels = make_dataset(sample_nums, weights_true, bias_true)

##################################################################

# 初始化超参数
lr = 0.03
epoch_nums = 10
batch_size = 32

# 定义数据读取迭代
dataset = torch.utils.data.TensorDataset(features, labels)  # 组合特征和标签
read_data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)  # 小批量读取随机打乱的数据

# 定义模型
net = torch.nn.Sequential(
    torch.nn.Linear(len(weights_true), 1)
    )

# 初始化模型参数
torch.nn.init.normal_(net[0].weight, mean=0, std=0.01)
torch.nn.init.constant_(net[0].bias, val=0)
# net[0].bias.data.fill_(0)

# 定义损失函数
mseloss = torch.nn.MSELoss()

# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
print(optimizer)

# 训练模型
epoch_loss = []
for epoch in range(epoch_nums):
    for x, y in read_data_iter:
        out = net(x)            # 前项计算
        loss = mseloss(out, y.view(-1, 1))  # 计算均方差损失
        optimizer.zero_grad()   # 清空梯度，net.zero_grad()
        loss.backward()         # 反向传播计算梯度
        optimizer.step()        # 更新参数

    # 打印每个epoch的训练结果
    print("epoch %d, loss: %.6f" % (epoch+1, loss.item()))
    epoch_loss.append(loss.item())

print(weights_true, net[0].weight)
print(bias_true, net[0].bias)

plt.plot([e+1 for e in range(epoch_nums)], epoch_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
