# !usr/bin/python
# -*_ coding:utf-8 -*-

"""A softmax implement by hands"""

__author__ = "huluwa-2020-04-12"

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


def softmax(output):
    """

    :param output:
    :return:
    """
    output_exp = output.exp()
    line_sum = output_exp.sum(dim=1, keepdim=True)  # 行上求和
    return output_exp / line_sum


def calculate(x):
    """

    :param x:
    :return:
    """
    return softmax(torch.mm(x.view((-1, num_inputs)), weights) + bias)


def cross_entropy(predict, label):
    """
    loss = -log(y)，y:正确类别的预测概率
    :param predict:
    :param label:
    :return:
    """

    return -torch.log(predict.gather(1, label.view(-1, 1)))


def sgd(params, lr, batch_size):
    """
    随机梯度下降
    :param params:
    :param lr:
    :param batch_size:
    :return:
    """
    for param in params:
        param.data -= lr * param.grad / batch_size  # 批量对每一个参数进行更新


def evaluate_accuracy(net, test_iter):
    """

    :param net:
    :param test_iter:
    :return:
    """
    acc_sum, loss_sum = 0, 0

    for features, labels in test_iter:
        predict = net(features)
        loss_sum += cross_entropy(predict, labels).sum().item()
        acc_sum += (predict.argmax(dim=1) == labels).float().sum().item()
    return acc_sum / test_iter.sampler.num_samples, loss_sum / test_iter.sampler.num_samples


def train(train_iter, test_iter, num_epochs, batch_size, lr, params):
    """

    :param train_iter:
    :param test_iter:
    :param num_epochs:
    :param batch_size:
    :param lr:
    :param params:
    :return:
    """
    for epoch in range(num_epochs):
        train_acc_count = 0
        train_loss_sum = 0

        for features, labels in train_iter:
            predict = calculate(features)
            loss = cross_entropy(predict, labels).sum()

            # 反向传播
            loss.backward()
            # 更新梯度
            sgd(params, lr, batch_size)
            # 清空梯度
            for param in params:
                param.grad.data.zero_()

            # 统计训练信息
            train_acc_count += (predict.argmax(dim=1) == labels).sum().item()
            train_loss_sum += loss.item()

        train_acc = train_acc_count / train_iter.sampler.num_samples
        train_loss = train_loss_sum / train_iter.sampler.num_samples
        # 验证准确率
        valid_acc, valid_loss = evaluate_accuracy(calculate, test_iter)

        print('epoch %d, train loss %.4f accuracy %.4f, valid loss %.4f accuracy %.4f' %
              (epoch+1, train_loss, train_acc, valid_loss, valid_acc))



batch_size = 256
num_epochs = 10
lr = 0.05
num_inputs = 784
num_outputs = 10

weights = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float32)
bias = torch.zeros(num_outputs, dtype=torch.float32)

# 记录权重和偏置的梯度
weights.requires_grad_(requires_grad=True)
bias.requires_grad_(requires_grad=True)

train_iter, test_iter = create_dataset_iter(batch_size)

train(train_iter, test_iter, num_epochs, batch_size, lr, [weights, bias])


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


def test(test_iter):
    """

    :param test_iter:
    :return:
    """
    features, labels = iter(test_iter).next()
    features = features[: 10]
    labels = labels[: 10]
    true_text_labels = get_fashion_mnist_text_labels(labels)
    pred_text_lables = get_fashion_mnist_text_labels(calculate(features).argmax(dim=1).numpy())

    titles = [true + '\n' + pred for true, pred in zip(true_text_labels, pred_text_lables)]

    show_fashion_mnist(features, titles)

test(test_iter)
