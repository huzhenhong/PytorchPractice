# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : A line regress implement by hands
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2020-11-21 16:53:18
 LastEditors  : huzhenhong
 LastEditTime : 2020-12-01 00:09:51
 FilePath     : \\PytorchPractice\\01-LineRegress\\line_regress_manual.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''

import torch
import time
import numpy as np


class LineRegressManual:
    def __init__(self, sample_num, weight_true, bias_true, epoch_num, batch_size, lr):
        self._sample_num = sample_num
        self._weight_true = weight_true
        self._bias_true = bias_true
        self._epoch_num = epoch_num
        self._batch_size = batch_size
        self._lr = lr

        self.__init_params()

    def __init_params(self):
        self._weight_pred = torch.tensor(np.random.normal(0, 0.01, (len(self._weight_true), 1)), dtype=torch.float32)
        self._bias_pred = torch.zeros(1)

        # 参数默认为True
        self._weight_pred.requires_grad_()
        self._bias_pred.requires_grad_()

    def read_dataset(self, features, labels):
        # 创建一张索引表，并将其随机打乱
        index_list = list(range(self._sample_num))
        np.random.shuffle(index_list)

        for i in range(0, self._sample_num, self._batch_size):
            # 最后一次可能不足batch_size
            index = index_list[i:min(i + self._batch_size, self._sample_num)]
            # list转tensor
            j = torch.tensor(index)
            yield features.index_select(0, j), labels.index_select(0, j)

    def calclate(self, X, W, b):
        return torch.mm(X, W) + b

    def squared_loss(self, pred, label):
        # 除以2是为了简化求导公式
        return (pred - label.view(pred.size())) ** 2 / 2

    def sgd(self, params):
        for param in params:
            # 因为损失是求的累加，反向传播求导的结果也是累加的
            # 每一个参数进行更新时需要先求下平均梯度
            param.data -= self._lr * param.grad / self._batch_size

    def train(self, features, labels):
        epoch_loss = []
        for epoch in range(self._epoch_num):
            start_time = time.time()
            batch_avg_loss = []

            for X, Y in self.read_dataset(features, labels):
                pred = self.calclate(X, self._weight_pred, self._bias_pred)
                # 求损失累和
                loss_sum = self.squared_loss(pred, Y).sum()
                # 反向传播求梯度，PyTorch会自动根据输入和前向计算构建计算图
                # 也就是说所有的计算都会被track，要是某个tensor的requires_grad_置为True
                # 就表示backward的时对其进行求导
                # 这里的损失是累加的（求损失是计算、累加也是计算，最后都会被计算图所记录），所以对应的weight梯度也是累加的
                # 所以更新梯度的时候需要先求下平均
                loss_sum.backward()
                # 小批量随机梯度下降更新模型
                self.sgd([self._weight_pred, self._bias_pred])

                # 清空梯度信息
                self._weight_pred.grad.data.zero_()
                self._bias_pred.grad.data.zero_()

                # 记录当前batch的平均loss
                batch_avg_loss.append(loss_sum.item() / self._batch_size)

            mean_loss = float(np.mean(batch_avg_loss))
            print('epoch {}, cost time {}, loss {}'.format(epoch+1, time.time() - start_time, mean_loss))
            epoch_loss.append(mean_loss)

        return epoch_loss, self._weight_pred, self._bias_pred
