# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : main logic
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2020-11-30 21:37:50
 LastEditors  : huzhenhong
 LastEditTime : 2020-12-01 16:25:20
 FilePath     : \\PytorchPractice\\01-LineRegress\\main.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from line_regress_manual import LineRegressManual
from line_regress_laconical import LineRegressLaconical


def draw_dataset_scatter(features, labels):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(features[:, 0], features[:, 1], labels, c=labels)
    ax.set_xlabel('features[0]')
    ax.set_ylabel('features[1]')
    ax.set_zlabel('lable')
    plt.show()


def draw_train_loss(loss):
    plt.figure()
    plt.plot([e+1 for e in range(len(loss))], loss, color='b', marker='.', label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def make_dataset(sample_num, weight_true, bias_true):
    # 标准正态分布，标准差为0，方差为1（高斯白噪声）
    features = torch.randn(sample_num, len(weight_true))
    labels = torch.zeros(sample_num)

    for i, w in enumerate(weight_true):
        labels += w * features[:, i]

    # 添加偏置，运用了广播机制
    labels += bias_true
    # 添加噪声
    # numpy默认dtype为np.float64, torch默认dtype为torch.float32
    # 此情况下dtype为torch.float64
    labels += torch.tensor(np.random.normal(0, 0.01, sample_num))

    return features, labels


if __name__ == '__main__':
    # 是否手动实现
    is_manual = True

    # 设置超参数
    sample_num = 1000
    weight_true = [1.2, -2.3]
    bias_true = 4.5
    lr = 0.03
    epoch_num = 10
    batch_size = 32

    # 制作数据集
    features, labels = make_dataset(sample_num, weight_true, bias_true)
    draw_dataset_scatter(features, labels)

    if is_manual:
        obj = LineRegressManual(sample_num, weight_true, bias_true, epoch_num, batch_size, lr)
    else:
        obj = LineRegressLaconical(sample_num, weight_true, bias_true, epoch_num, batch_size, lr)

    train_loss, weight_pred, bias_pred = obj.train(features, labels)

    draw_train_loss(train_loss)
    print('weight_pred : {},\nbias_pred : {}'.format(weight_pred.detach().numpy(), bias_pred.detach().numpy()))
    # print(f'weight_pred : {weight_pred.detach().numpy():<.8f},\nbias_pred : {bias_pred.detach().numpy():<.8f}')
