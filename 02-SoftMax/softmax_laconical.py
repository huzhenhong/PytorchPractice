# !usr/bin/python
# -*_ coding:utf-8 -*-

"""A softmax implement by hands"""

__author__ = "huluwa-2020-04-12"

import sys
import torch
import torchvision
from torch.utils import data
import matplotlib.pyplot as plt
from collections import OrderedDict


def create_dataset_iter(batch_size):
    """
    创建数据集迭代器
    :param batch_size:
    :return:
    """
    # ToTensor默认图像数据类型是uint8，会自动将像素值缩放到(0.0, 1.0),图片shape为CxHxW
    fashion_mnist_trian = torchvision.datasets.FashionMNIST(root='../dataset/fashion-mnist',
                                                            train=True,
                                                            download=True,
                                                            transform=torchvision.transforms.ToTensor())

    fashion_mnist_test = torchvision.datasets.FashionMNIST(root='../dataset/fashion-mnist',
                                                           train=False,
                                                           download=True,
                                                           transform=torchvision.transforms.ToTensor())

    if sys.platform.startswith('win'):
        num_workers = 0  # 不适用额外进程读取数据
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(fashion_mnist_trian,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

    test_iter = torch.utils.data.DataLoader(fashion_mnist_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers)

    return train_iter, test_iter


def evaluate_accuracy(net, test_iter):
    """

    :param net:
    :param test_iter:
    :return:
    """
    acc_sum = 0
    loss_sum = 0
    for features, labels in test_iter:
        predict = net(features)

        acc_sum += (predict.argmax(dim=1) == labels).float().sum().item()
        loss_sum += loss(predict, labels).sum().item()

    return acc_sum / test_iter.sampler.num_samples, loss_sum / test_iter.sampler.num_samples


def train(net, train_iter, test_iter, num_epochs, loss, optimizer):
    """

    :param net:
    :param train_iter:
    :param test_iter:
    :param num_epochs:
    :param loss:
    :param optimizer:
    :return:
    """
    for epoch in range(num_epochs):
        train_acc_count = 0
        train_loss_sum = 0

        for features, labels in train_iter:
            predict = net(features)
            los = loss(predict, labels).sum()

            optimizer.zero_grad()
            # 反向传播
            los.backward()
            # 更新梯度
            optimizer.step()

            # 统计训练信息
            train_acc_count += (predict.argmax(dim=1) == labels).float().sum().item()
            train_loss_sum += los.item()

        train_acc = train_acc_count / train_iter.sampler.num_samples
        train_loss = train_loss_sum / train_iter.sampler.num_samples
        # 验证准确率
        valid_acc, valid_loss = evaluate_accuracy(net, test_iter)

        print('epoch %d, train loss %.4f accuracy %.4f, valid loss %.4f accuracy %.4f' %
              (epoch+1, train_loss, train_acc, valid_loss, valid_acc))


batch_size = 256
num_epochs = 10
lr = 0.1
num_inputs = 784
num_outputs = 10


# class LinearNet(torch.nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = torch.nn.Linear(num_inputs, num_outputs)
#
#     def forward(self, x):
#         # x 的 shape 是 (batch, 1, 28, 28)
#         return self.linear(x.view(x.shape[0], -1))
#
# net = LinearNet(num_inputs, num_outputs)

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

# net = torch.nn.Sequential(
#     FlattenLayer(),
#     torch.nn.Linear(num_inputs, num_outputs)
# )


net = torch.nn.Sequential(
    OrderedDict([
            ('faltten', FlattenLayer()),
            ('linear', torch.nn.Linear(num_inputs, num_outputs))])
)

# 初始化参数
torch.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
torch.nn.init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 定义优化函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 创建数据集迭代器
train_iter, test_iter = create_dataset_iter(batch_size)

train(net, train_iter, test_iter, num_epochs, loss, optimizer)


def get_fashion_mnist_text_labels(labels):
    """
    通过标签索引对应的名称
    :param labels:
    :return:
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']

    return [text_labels[i] for i in labels]


def show_fashion_mnist(images, labels):
    """
    显示标签对应的图片
    :param images:
    :param labels:
    :return:
    """
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for fig, img, label in zip(figs, images, labels):
        fig.imshow(img.view((28, 28)).numpy())
        fig.set_title(label)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()


def test(net, test_iter):
    """

    :param net:
    :param test_iter:
    :return:
    """
    features, labels = next(iter(test_iter))
    features = features[: 10]
    labels = labels[: 10]
    true_text_labels = get_fashion_mnist_text_labels(labels)
    pred_text_lables = get_fashion_mnist_text_labels(net(features).argmax(dim=1).numpy())

    titles = [true + '\n' + pred for true, pred in zip(true_text_labels, pred_text_lables)]

    show_fashion_mnist(features, titles)


test(net, test_iter)
