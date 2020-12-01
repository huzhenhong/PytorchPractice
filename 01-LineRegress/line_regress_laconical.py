# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : A line regress implement by pytorch high api
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2020-11-21 16:53:18
 LastEditors  : huzhenhong
 LastEditTime : 2020-12-01 16:40:42
 FilePath     : \\PytorchPractice\\01-LineRegress\\line_regress_laconical.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''

import torch
import time
import numpy as np
from torch.utils import data


class LineRegressLaconical():
    def __init__(self, sample_num, weight_true, bias_true, epoch_num, batch_size, lr):
        self._sample_num = sample_num
        self._weight_true = weight_true
        self._bias_true = bias_true
        self._epoch_num = epoch_num
        self._batch_size = batch_size
        self._lr = lr

        self.__init_net(len(self._weight_true))

    def __init_net(self, input_size):
        self._net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1)
            )

        # 框架已默认进行了初始化
        # # 初始化模型参数
        # torch.nn.init.normal_(self._net[0].weight, mean=0, std=0.01)
        # torch.nn.init.constant_(self._net[0].bias, val=0)
        # # self._net[0].bias.data.fill_(0)

        # 定义损失函数
        self._mseloss = torch.nn.MSELoss()

        # 定义优化算法
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self._lr)

    def train(self, features, labels):
        # 组合特征和标签
        dataset = data.TensorDataset(features, labels)
        # 小批量读取随机打乱的数据
        read_data_iter = data.DataLoader(dataset, self._batch_size, shuffle=True)

        epoch_loss = []

        for epoch in range(self._epoch_num):
            start_time = time.time()

            batch_loss = []
            for X, Y in read_data_iter:
                pred = self._net(X)
                loss = self._mseloss(pred, Y.view(-1, 1))
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                batch_loss.append(loss.item())

            mean_loss = float(np.mean(batch_loss))
            print('epoch {}, cost time {}, loss {}'.format(epoch+1, time.time() - start_time, mean_loss))
            epoch_loss.append(mean_loss)

        return epoch_loss, self._net[0].weight, self._net[0].bias
