# !usr/bin/python
# -*- coding:utf-8 -*-

"""A house prices predict demo"""

__author__ = "huzhenhong 2020-04-19"


import time
import torch
import torch.utils.data
import pandas as pd
import matplotlib.pyplot as plt


def log_rmse(predict, labels):
    # 将小于1的数设为1，这样预测值就不会为负数，毕竟房价没有负数
    cliped_pred = torch.max(predict, torch.tensor(1.0).requires_grad_())
    rmse = torch.sqrt(2 * loss(cliped_pred.log(), labels.log()).mean())

    return rmse

def make_dataset():
    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 第一个为ID，最后一个为标签，都去掉

    # 获取数字特征索引，去掉object型别的特征，比如字符串后剩下的就是数字了
    numeric_features_index = all_features.dtypes[all_features.dtypes != 'object'].index

    # 对数字特征进行标准化
    all_features[numeric_features_index] = all_features[numeric_features_index]\
        .apply(lambda x: (x - x.mean()) / x.std())
    # 标准化后，所有数值特征的均值为0，所以可以用0来替换确实缺失的值
    all_features[numeric_features_index] = all_features[numeric_features_index].fillna(0)

    # 将非数字特征转化为数字特征，比如MSZoning可是RL或者RM，那么就对MSZoning进行one-hot编码,这一步会导致特征数增加
    all_features = pd.get_dummies(all_features, dummy_na=True)  # 将缺失值也当做一种情况进行one-hot编码

    # 构建数据集
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[: n_train].values, dtype=torch.float)
    test_features = torch.tensor(all_features[n_train: ].values, dtype=torch.float)
    train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float).view(-1, 1)

    return train_features, train_labels, test_features


def validate(valid_iter):
    acc_sum, loss_sum = 0, 0

    # 验证时不需要计算梯度
    with torch.no_grad():
        for features, labels in valid_iter:
            predict = net(features)
            loss_sum += log_rmse(predict, labels).sum().item()
            acc_sum += (predict.argmax(dim=1) == labels).float().sum().item()

    return acc_sum / valid_iter.sampler.num_samples, loss_sum / valid_iter.sampler.num_samples


def train(train_iter, valid_iter, num_epochs):
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_valid_loss = []
    epoch_valid_acc = []

    for epoch in range(num_epochs):
        train_acc_count = 0
        train_loss_sum = 0

        start = time.time()

        for features, labels in train_iter:
            predict = net(features)
            los = log_rmse(predict, labels).sum()
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            los.backward()
            # 更新梯度
            optimizer.step()

            # 统计训练信息
            train_acc_count += (predict.argmax(dim=1) == labels).sum().item()
            train_loss_sum += los.item()

        train_acc = train_acc_count / train_iter.sampler.num_samples
        train_loss = train_loss_sum / train_iter.sampler.num_samples
        # 验证准确率
        valid_acc, valid_loss = validate(valid_iter)

        print(f'epoch {epoch + 1}, cost time {time.time() - start:<.4f}, '
              f'train loss {train_loss:<.4f} accuracy {train_acc:<.4f}, '
              f'valid loss {valid_loss:<.4f} accuracy {valid_acc:<.4f}')

        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_valid_loss.append(valid_loss)
        epoch_valid_acc.append(valid_acc)

    return epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc


def get_k_fold_data_iter(k, i, train_features, train_labels, batch_size):
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

    train_dataset = torch.utils.data.TensorDataset(x_trian, y_train)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)

    # 不打乱没有.sampler.num_samples
    return train_iter, valid_iter


def k_fold_trian(k, train_features, train_labels, num_epochs, batch_size):
    k_train_loss, k_valid_loss = [], []
    k_train_acc, k_valid_acc = [], []

    for i in range(k):
        train_iter, valid_iter = get_k_fold_data_iter(k, i, train_features, train_labels, batch_size)

        train_loss, train_acc, valid_loss, valid_acc = train(train_iter, valid_iter, num_epochs)
        k_train_loss.append(train_loss[-1])
        k_train_acc.append(train_acc[-1])
        k_valid_loss.append(valid_loss[-1])
        k_valid_acc.append(valid_acc[-1])

    xs = [i for i in range(k)]
    plt.plot(xs, k_train_loss, color='b', label='k_train_loss')
    plt.plot(xs, k_valid_loss, color='r', label='k_valid_loss')

    # plt.plot(xs+0.25, k_train_acc, color='g', label='k_train_acc')
    # plt.plot(xs+0.75, k_valid_acc, color='c', label='k_valid_acc')
    plt.legend()
    plt.show()



lr = 0.01
weight_decay = 0.1
num_epochs = 10
batch_size = 64

train_features, train_labels, test_features = make_dataset()
loss = torch.nn.MSELoss()
net = torch.nn.Linear(train_features.shape[-1], 1)
for param in net.parameters():
    param.requires_grad_()

optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)


k_fold_trian(10, train_features, train_labels, num_epochs, batch_size)