# !usr/bin/python
# -*- coding:utf-8 -*-

"""multi layer perception implement by hands"""

__author__ = "huzhenhong 2020-04-15"


import sys
import time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


class MyMultiLayerPerception:
    """
    自定义一个多层感知机实现类
    """
    def __init__(self, num_inputs, num_outputs, num_hiddens, num_epochs, batch_size, lr):
        # 超参数设定
        self.num_inputs, self.num_outputs, self.num_hiddens, self.num_epochs, self.batch_size, self.lr = \
            num_inputs, num_outputs, num_hiddens, num_epochs, batch_size, lr

        # 初始化参数
        self.weight_1 = torch.tensor(np.random.normal(0, 0.01, (self.num_inputs, self.num_hiddens)),
                                     dtype=torch.float).cuda()

        self.bias_1 = torch.zeros(1, dtype=torch.float).cuda()

        self.weight_2 = torch.tensor(np.random.normal(0, 0.01, (self.num_hiddens, self.num_outputs)),
                                     dtype=torch.float).cuda()

        self.bias_2 = torch.zeros(self.num_outputs, dtype=torch.float).cuda()

        # 设定参数需要记录梯度
        self.params = [self.weight_1, self.bias_1, self.weight_2, self.bias_2]
        for param in self.params:
            param.requires_grad_()

        # 创建损失函数和优化器
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr)

    def create_dataset_iter(self):
        """
        创建数据集迭代器
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
            num_workers = 4  # 0表示不适用额外进程读取数据
        else:
            num_workers = 4

        train_iter = torch.utils.data.DataLoader(fashion_mnist_trian,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers)

        test_iter = torch.utils.data.DataLoader(fashion_mnist_test,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)

        return train_iter, test_iter

    def relu(self, x):
        """
        自定义relu
        :param x:
        :return:
        """
        return torch.max(x, other=torch.tensor(0.0).cuda())

    def forward(self, x):
        """
        自定义前馈计算
        :param x:
        :return:
        """
        x = x.view((-1, self.num_inputs))
        h = self.relu(torch.matmul(x, self.weight_1) + self.bias_1)
        return torch.matmul(h, self.weight_2) + self.bias_2   # 输出层不激活

    def evaluate_accuracy(self, test_iter):
        """

        :param test_iter:
        :return:
        """
        acc_sum, loss_sum = 0, 0

        for features, labels in test_iter:
            features = features.cuda()
            labels = labels.cuda()

            predict = self.forward(features)
            loss_sum += self.loss(predict, labels).sum().item()
            acc_sum += (predict.argmax(dim=1) == labels).float().sum().item()

        return acc_sum / test_iter.sampler.num_samples, loss_sum / test_iter.sampler.num_samples

    def train(self, train_iter, test_iter):
        """

        :param train_iter:
        :param test_iter:
        :return:
        """
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_valid_loss = []
        epoch_valid_acc = []

        for epoch in range(self.num_epochs):
            train_acc_count = 0
            train_loss_sum = 0

            start = time.time()

            for features, labels in train_iter:
                features = features.cuda()
                labels = labels.cuda()

                predict = self.forward(features)
                los = self.loss(predict, labels).sum()
                # 清空梯度
                self.optimizer.zero_grad()
                # 反向传播
                los.backward()
                # 更新梯度
                self.optimizer.step()

                # 统计训练信息
                train_acc_count += (predict.argmax(dim=1) == labels).sum().item()
                train_loss_sum += los.item()

            train_acc = train_acc_count / train_iter.sampler.num_samples
            train_loss = train_loss_sum / train_iter.sampler.num_samples
            # 验证准确率
            valid_acc, valid_loss = self.evaluate_accuracy(test_iter)

            print(f'epoch {epoch+1}, cost time {time.time() - start:<.4f}, '
                  f'train loss {train_loss:<.4f} accuracy {train_acc:<.4f}, '
                  f'valid loss {valid_loss:<.4f} accuracy {valid_acc:<.4f}')

            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
            epoch_valid_loss.append(valid_loss)
            epoch_valid_acc.append(valid_acc)

        return epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc

    def draw_result(self, train_loss, train_acc, valid_loss, valid_acc):
        epochs = [e + 1 for e in range(self.num_epochs)]

        plt.figure(figsize=(24, 12))

        plt.subplot(121)
        plt.title('loss')
        plt.plot(epochs, train_loss, color='b', marker='.', label='train_loss')
        plt.plot(epochs, valid_loss, color='r', marker='.', label='valid_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.ylim([0, 1])
        plt.legend()

        plt.subplot(122)
        plt.title('accuracy')
        plt.plot(epochs, train_acc, color='b', marker='.', label='train_acc')
        plt.plot(epochs, valid_acc, color='r', marker='.', label='valid_acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.ylim([0, 1])
        plt.legend()

        plt.show()


if __name__ == '__main__':
    # 主逻辑
    multiLayerPerception = MyMultiLayerPerception(784, 10, 1024, 10, 64, 0.05)

    train_iter, test_iter = multiLayerPerception.create_dataset_iter()

    train_loss, train_acc, valid_loss, valid_acc = multiLayerPerception.train(train_iter, test_iter)

    multiLayerPerception.draw_result(train_loss, train_acc, valid_loss, valid_acc)
