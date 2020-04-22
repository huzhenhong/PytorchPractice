# !usr/bin/python
# -*- coding:utf-8 -*-

"""A house prices k fold train"""

__author__ = "huzhenhong 2020-04-19"


from house_prices_model import HousePrices
import torch
import torch.utils.data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class KFold:
    def __init__(self):
        pass

    def make_dataset(self):
        train_data = pd.read_csv('dataset/train.csv')
        test_data = pd.read_csv('dataset/test.csv')

        all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 第一个为ID，最后一个为标签，都去掉

        # 获取数字特征索引，去掉object型别的特征，比如字符串后剩下的就是数字了
        numeric_features_index = all_features.dtypes[all_features.dtypes != 'object'].index

        # 对数字特征进行标准化
        all_features[numeric_features_index] = all_features[numeric_features_index] \
            .apply(lambda x: (x - x.mean()) / x.std())
        # 标准化后，所有数值特征的均值为0，所以可以用0来替换确实缺失的值
        all_features[numeric_features_index] = all_features[numeric_features_index].fillna(0)

        # 将非数字特征转化为数字特征，比如MSZoning可是RL或者RM，那么就对MSZoning进行one-hot编码,这一步会导致特征数增加
        all_features = pd.get_dummies(all_features, dummy_na=True)  # 将缺失值也当做一种情况进行one-hot编码

        # 构建数据集
        n_train = train_data.shape[0]
        train_features = torch.tensor(all_features[: n_train].values, dtype=torch.float)
        test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
        train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float).view(-1, 1)
        # train_labels1 = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

        return train_features, train_labels, test_features, test_data


    def get_k_fold_data(self, k, i, train_features, train_labels):
        assert k > 1
        fold_size = train_features.shape[0] // k

        x_trian, y_train, x_valid, y_valid = None, None, None, None

        for j in range(k):
            idx = slice(j * fold_size, (j+1) * fold_size)
            x_part, y_part = train_features[idx, :], train_labels[idx]

            if j == i:
                x_valid, y_valid = x_part, y_part
            elif x_trian is None:
                x_trian, y_train = x_part, y_part
            else:
                x_trian = torch.cat((x_trian, x_part), dim=0)
                y_train = torch.cat((y_train, y_part), dim=0)

        return x_trian, y_train, x_valid, y_valid


    def k_fold_train(self, k, num_epochs, batch_size, train_features, train_labels, lr, weight_decay):
        k_train_loss, k_valid_loss = {}, {}
        final_train_loss, final_valid_loss = [], []

        for i in range(k):
            house_prices = HousePrices(num_epochs, batch_size, train_features.shape[-1], lr, weight_decay)

            # 获取数据
            data = self.get_k_fold_data(k, i, train_features, train_labels)

            # K折训练
            train_loss, valid_loss = house_prices.train(*data)

            k_train_loss[i] = train_loss
            k_valid_loss[i] = valid_loss

            # 其实只关心最后一次的loss就好，不能平均loss，平均没有意义，关心的是训练结束后的性能
            final_train_loss.append(train_loss[-1])
            final_valid_loss.append(valid_loss[-1])

        print(f'final train loss [{np.mean(final_train_loss):<.4f}] valid loss [{np.mean(final_valid_loss):<.4f}]')

        return k_train_loss, k_valid_loss


    def draw_k_train_result(self, k, num_epochs, k_train_loss, k_valid_loss):
        # 绘制最后训练结果
        plt.figure(figsize=(25, 5))
        for i in range(k):
            # 绘制每一折里num_epochs训练结果
            xs = [x for x in range(num_epochs)]
            plt.subplot(1, k, i+1)
            plt.plot(xs, k_train_loss[i], color='b', label='k_train_loss')
            plt.plot(xs, k_valid_loss[i], color='r', label='k_valid_loss')
            plt.ylim(0, 0.5)
            plt.xlabel('k')
            plt.ylabel('loss')
            plt.title(f'k={i+1}')
            plt.legend()

        plt.show()


k_fold = KFold()

# 步骤一：获取数据集
train_features, train_labels, test_features, test_data = k_fold.make_dataset()

# 步骤二：超参数设定并训练
k = 5
num_epochs = 150
batch_size = 32
lr = 10
weight_decay = 0.01

train_loss, valid_loss = k_fold.k_fold_train(k, num_epochs, batch_size, train_features,
                                      train_labels, lr=lr, weight_decay=weight_decay)

k_fold.draw_k_train_result(k, num_epochs, train_loss, valid_loss)

# 步骤三
# 步骤四
# 进行调参

# 步骤五：选定模型后充分训练一遍
house_prices = HousePrices(num_epochs, batch_size, train_features.shape[-1], lr, weight_decay)
loss, _ = house_prices.train(train_features, train_labels, None, None)

plt.plot([x+1 for x in range(num_epochs)], loss, marker='.', color='red', label='train_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(0, 0.5)
plt.legend()
plt.show()

# 保存模型
params = house_prices.net.named_parameters()

for i, param in enumerate(params):
    if 'weight' in param[0]:
        torch.save(param[1], "weight.pt")
    if 'bias' in param[0]:
        torch.save(param[1], "bias.pt")
