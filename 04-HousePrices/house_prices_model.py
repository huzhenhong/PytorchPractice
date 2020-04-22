# !usr/bin/python
# -*- coding:utf-8 -*-

"""A house prices model class"""

__author__ = "huzhenhong 2020-04-19"


import time
import torch
import torch.utils.data


class HousePrices:
    def __init__(self, num_epochs, batch_size, input_size, lr, weight_decay):
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1),
            # torch.nn.Dropout(0.1),
            # torch.nn.Linear(5, 1),
            # torch.nn.Dropout(0.1),
            # torch.nn.Linear(10, 1)
        )

        for param in self.net.parameters():
            # torch.nn.init.normal_(param, mean=0, std=0.01)
            param.requires_grad_()

        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr, weight_decay=weight_decay)

        self.loss = torch.nn.MSELoss()

    def log_rmse(self, features, labels):
        with torch.no_grad():
            # 将小于1的数设为1，这样预测值就不会为负数，毕竟房价没有负数
            cliped_pred = torch.max(self.net(features), torch.tensor(1.0))
            rmse = torch.sqrt(self.loss(cliped_pred.log(), labels.log()))

        return rmse.item()

    def train(self, x_trian, y_train, x_valid, y_valid):

        train_iter = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_trian, y_train),
                                                 batch_size=self.batch_size,
                                                 shuffle=True)

        epoch_train_loss = []
        epoch_valid_loss = []

        # 这里为什么要训练多个epoch，调参的时只训练一个epoch可能看不出效果
        for epoch in range(self.num_epochs):
            start = time.time()

            for features, labels in train_iter:
                predict = self.net(features)
                los = self.loss(predict, labels).sum()
                # 清空梯度
                self.optimizer.zero_grad()
                # 反向传播
                los.backward()
                # 更新梯度
                self.optimizer.step()

            # 再用当前epoch训练好的模型验证下官方损失函数
            train_loss = self.log_rmse(x_trian, y_train)
            epoch_train_loss.append(train_loss)

            if x_valid is not None:
                valid_loss = self.log_rmse(x_valid, y_valid)
                epoch_valid_loss.append(valid_loss)
                print(f'epoch {epoch + 1}, cost time {time.time() - start:<.4f}, '
                      f'train loss {train_loss:<.4f} valid loss {valid_loss:<.4f}')
            else:
                print(f'epoch {epoch + 1}, cost time {time.time() - start:<.4f}, train loss {train_loss:<.4f}')


        return epoch_train_loss, epoch_valid_loss
