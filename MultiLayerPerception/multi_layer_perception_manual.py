# !usr/bin/python
# -*- coding:utf-8 -*-

"""multi layer perception implement by hands"""

__author__ = "huzhenhong 2020-04-15"


import sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def create_dataset_iter(batch_size):
    """
    创建数据集迭代器
    :param batch_size:
    :return:
    """
    # ToTensor默认图像数据类型是uint8，会自动将像素值缩放到(0.0, 1.0),图片shape为CxHxW
    fashion_mnist_trian = torchvision.datasets.FashionMNIST(root='dataset',
                                                            train=True,
                                                            download=True,
                                                            transform=torchvision.transforms.ToTensor())

    fashion_mnist_test = torchvision.datasets.FashionMNIST(root='dataset',
                                                           train=False,
                                                           download=True,
                                                           transform=torchvision.transforms.ToTensor())

    if sys.platform.startswith('win'):
        num_workers = 0  # 不适用额外进程读取数据
    else:
        num_workers = 0

    train_iter = torch.utils.data.DataLoader(fashion_mnist_trian,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

    test_iter = torch.utils.data.DataLoader(fashion_mnist_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers)

    return train_iter, test_iter


def relu(x):
    """
    自定义relu
    :param x:
    :return:
    """
    return torch.max(x, other=torch.tensor(0.0))


def forward(x):
    """
    自定义前馈计算
    :param x:
    :return:
    """
    x = x.view((-1, num_inputs))
    h = relu(torch.matmul(x, weight_1) + bias_1)
    return torch.matmul(h, weight_2) + bias_2   # 输出层不激活


def evaluate_accuracy(test_iter):
    """

    :param test_iter:
    :return:
    """
    acc_sum, loss_sum = 0, 0

    for features, labels in test_iter:
        predict = forward(features)
        loss_sum += loss(predict, labels).sum().item()
        acc_sum += (predict.argmax(dim=1) == labels).float().sum().item()
    return acc_sum / test_iter.sampler.num_samples, loss_sum / test_iter.sampler.num_samples


def train(train_iter, test_iter, num_epochs):
    """

    :param train_iter:
    :param test_iter:
    :param num_epochs:
    :return:
    """
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_valid_loss = []
    epoch_valid_acc = []

    for epoch in range(num_epochs):
        train_acc_count = 0
        train_loss_sum = 0

        for features, labels in train_iter:
            predict = forward(features)
            l = loss(predict, labels).sum()
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            # 更新梯度
            optimizer.step()

            # 统计训练信息
            train_acc_count += (predict.argmax(dim=1) == labels).sum().item()
            train_loss_sum += l.item()

        train_acc = train_acc_count / train_iter.sampler.num_samples
        train_loss = train_loss_sum / train_iter.sampler.num_samples
        # 验证准确率
        valid_acc, valid_loss = evaluate_accuracy(test_iter)

        print('epoch %d, train loss %.4f accuracy %.4f, valid loss %.4f accuracy %.4f' %
              (epoch+1, train_loss, train_acc, valid_loss, valid_acc))

        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_valid_loss.append(valid_loss)
        epoch_valid_acc.append(valid_acc)

    return (epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc)


# 超参数设定
num_inputs, num_outputs, num_hiddens, num_epochs, batch_size, lr = 784, 10, 256, 10, 128, 0.01

weight_1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
bias_1 = torch.zeros(1, dtype=torch.float)

weight_2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
bias_2 = torch.zeros(num_outputs, dtype=torch.float)

# 设定参数需要记录梯度
params = [weight_1, bias_1, weight_2, bias_2]
for param in params:
    param.requires_grad_()

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params, lr=lr)

train_iter, test_iter = create_dataset_iter(batch_size)

train_loss, train_acc, valid_loss, valid_acc = train(train_iter, test_iter, num_epochs)

epochs = [e+1 for e in range(num_epochs)]
plt.figure(figsize=(12, 12))

plt.subplot(121)
plt.plot(epochs, train_loss, color='b', marker='.', label='train_loss')
plt.plot(epochs, train_acc, color='r', marker='.', label='train_acc')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim([0, 1])

plt.subplot(122)
plt.plot(epochs, valid_loss, color='b', marker='.', label='valid_loss')
plt.plot(epochs, valid_acc, color='r', marker='.', label='valid_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim([0, 1])
plt.show()