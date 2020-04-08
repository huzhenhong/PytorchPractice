# !usr/bin/python
# -*_ coding:utf-8 -*-

"""A line regress implement by hands"""

__author__ = "huluwa-2020-04-08"

import torch
import numpy as np


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


def read_dataset(batch_size, features, labels):
    """
    批量读取数据
    :param batch_size:
    :param features:
    :param labels:
    :return:
    """
    samples_nums = len(features)
    index_list = list(range(samples_nums))  # 创建一张索引表
    np.random.shuffle(index_list)  # 随机打乱

    for i in range(0, samples_nums, batch_size):
        index = index_list[i: min(i+batch_size, samples_nums)]  # 最后一次可能不足batch_size
        j = torch.LongTensor(index)
        yield features.index_select(0, j), labels.index_select(0, j)


def calclate(x, w, b):
    """
    前向计算
    :param x:
    :param w:
    :param b:
    :return:
    """
    return torch.mm(x, w) + b


def squared_loss(prid, label):
    """
    求均方误差
    :param prid:
    :param label:
    :return:
    """
    return (prid - label.view(prid.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    随机梯度下降
    :param params:
    :param lr:
    :param batch_size:
    :return:
    """
    for param in params:
        param.data -= lr * param.grad / batch_size  # 批量对每一个参数进行更新


# 生产数据集
sample_nums = 1000
weights_true = [2, -3.4]
bias_true = 4.2

features, labels = make_dataset(sample_nums, weights_true, bias_true)


# 初始化超参数
lr = 0.03
epoch_nums = 10
batch_size = 32

# 初始化权重和偏置
weights = torch.tensor(np.random.normal(0, 0.01, (len(weights_true), 1)), dtype=torch.float32)
bias = torch.zeros(1, dtype=torch.float32)

# 监测梯度，以便训练
weights.requires_grad_(requires_grad=True)
bias.requires_grad_(requires_grad=True)


epoch_loss = []     # 用来统计每个epoch的loss
for epoch in range(epoch_nums):
    for x, y in read_dataset(batch_size, features, labels):
        prid = calclate(x, weights, bias)       # 前向计算
        loss = squared_loss(prid, y).sum()      # 求损失和
        loss.backward()                         # 反向传播求梯度
        sgd([weights, bias], lr, batch_size)    # 小批量随机梯度下降更新模型

        # 清空梯度信息
        weights.grad.data.zero_()
        bias.grad.data.zero_()

        epoch_loss.append(float(loss))  # 记录当前loss

    # 每个epoch结束时打印统计训练信息
    print('epoch {}, loss {}'.format(epoch+1, np.mean(epoch_loss)))

print(weights_true, weights)
print(bias_true, bias)
