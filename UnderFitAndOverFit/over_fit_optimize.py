# !usr/bin/python
# -*- coding:utf-8 -*-

"""A weight dropout demo"""

__author__ = 'huzhenhong 2020-04-18'

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt


class PloyFitExperiment:
    def __init__(self, num_imputs, batch_size, lr, drop_rate=0.0):
        self.num_imputs = num_imputs
        self.batch_size = batch_size
        # 定义模型
        self.loss = torch.nn.MSELoss()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_imputs, 20),
            torch.nn.Dropout(drop_rate),
            torch.nn.Linear(20, 1),
        )

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

    def make_data_iter(self, n_train, n_valid):

        # weight_true = torch.randn(( self.num_imputs, 1))
        weight_true = torch.ones(self.num_imputs, 1) * 0.01
        bias_true = 0.5

        features = torch.randn((n_train + n_valid,  self.num_imputs))
        labels = torch.matmul(features, weight_true) + bias_true
        labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

        train_data = torch.utils.data.TensorDataset(features[:n_train], labels[:n_train])
        valid_data = torch.utils.data.TensorDataset(features[n_train:], labels[n_train:])

        train_iter = torch.utils.data.DataLoader(train_data, self.batch_size, shuffle=True)
        valid_iter = torch.utils.data.DataLoader(valid_data, self.batch_size, shuffle=False)

        return train_iter, valid_iter

    def l2_penalty(self, weights):
        """
        L2范数权重惩罚
        :param weights:
        :return:
        """
        return torch.pow((weights ** 2).sum(), 0.5)

    def fit(self, train_iter, valid_iter, num_epochs, weight_decay=0.0):
        """

        :param train_iter:
        :param valid_iter:
        :param num_epochs:
        :param weight_decay:
        :return:
        """
        epoch_trian_loss = []
        epoch_valid_loss = []

        for epoch in range(num_epochs):
            step_loss = []
            # 训练
            for features, labels in train_iter:
                l = self.loss(self.net(features), labels.view(-1, 1)).sum() + \
                    weight_decay * self.l2_penalty(self.net[0].weight.data)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                step_loss.append(l.item())

            # 训练损失
            train_loss = np.mean(step_loss)
            epoch_trian_loss.append(train_loss)
            # 验证损失
            valid_loss = self.validate(valid_iter)
            epoch_valid_loss.append(valid_loss)

            print(f'epoch {epoch} trian loss: {train_loss:<.4f} valid loss: {valid_loss:<.4f}')

        # print(f'weight: {self.net.weight.data}, bias: {self.net.bias.data}')

        return epoch_trian_loss, epoch_valid_loss

    def validate(self, valid_iter):
        """
        验证当前模型的损失
        :param valid_iter:
        :return:
        """
        iter_loss = []

        with torch.no_grad():
            for features, labels in valid_iter:
                l = self.loss(self.net(features), labels.view(-1, 1)).sum()
                iter_loss.append(l.item())

        return np.mean(iter_loss)

    def draw_reslut(self, trian_loss, valid_loss):
        """

        :param trian_loss:
        :param valid_loss:
        :return:
        """
        epochs = [e for e in range(len(trian_loss))]
        plt.plot(epochs, trian_loss, color='b', label='trian_loss')
        plt.plot(epochs, valid_loss, color='r', label='valid_loss')
        plt.ylim(0, 0.5)
        plt.legend()
        plt.show()


ploy_fit = PloyFitExperiment(num_imputs=200, batch_size=16, lr=0.01, drop_rate=0.0)
train_iter, valid_iter = ploy_fit.make_data_iter(20, 100)
train_losses, valid_losses = ploy_fit.fit(train_iter, valid_iter, num_epochs=20, weight_decay=0.1)
ploy_fit.draw_reslut(train_losses, valid_losses)
