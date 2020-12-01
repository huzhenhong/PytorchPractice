# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2020-12-01 17:30:48
 LastEditors  : huzhenhong
 LastEditTime : 2020-12-01 17:43:50
 FilePath     : \\PytorchPractice\\Util\\base_function.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt


def make_dataset(sample_num, weight_true, bias_true, is_add_nose=True):
    # 标准正态分布，标准差为0，方差为1（高斯白噪声）
    features = torch.randn(sample_num, len(weight_true))
    labels = torch.zeros(sample_num)

    for i, w in enumerate(weight_true):
        labels += w * features[:, i]

    # 添加偏置，运用了广播机制
    labels += bias_true
    # 添加噪声

    if is_add_nose:
        labels += torch.tensor(np.random.normal(0, 0.01, sample_num))

    return features, labels


def draw_dataset_scatter3d(features, labels):
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