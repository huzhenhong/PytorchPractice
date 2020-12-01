# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : main logic
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2020-11-30 21:37:50
 LastEditors  : huzhenhong
 LastEditTime : 2020-12-01 17:44:25
 FilePath     : \\PytorchPractice\\01-LineRegress\\main.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''
import path_magic
import base_function as bf
from line_regress_manual import LineRegressManual
from line_regress_laconical import LineRegressLaconical

path_magic.print_cwd()


if __name__ == '__main__':
    # 是否手动实现求loss和sgd
    is_manual = True

    # 设置超参数
    sample_num = 1000
    weight_true = [1.2, -2.3]
    bias_true = 4.5
    lr = 0.03
    epoch_num = 10
    batch_size = 32

    # 制作数据集
    features, labels = bf.make_dataset(sample_num, weight_true, bias_true)
    bf.draw_dataset_scatter3d(features, labels)

    if is_manual:
        obj = LineRegressManual(sample_num, weight_true, bias_true, epoch_num, batch_size, lr)
    else:
        obj = LineRegressLaconical(sample_num, weight_true, bias_true, epoch_num, batch_size, lr)

    train_loss, weight_pred, bias_pred = obj.train(features, labels)

    bf.draw_train_loss(train_loss)
    print('weight_pred : {},\nbias_pred : {}'.format(weight_pred.detach().numpy(), bias_pred.detach().numpy()))
