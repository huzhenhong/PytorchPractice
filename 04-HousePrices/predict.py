# !usr/bin/python
# -*- coding:utf-8 -*-

"""predict"""

__author__ = "huzhenhong 2020-04-19"




import torch
import pandas as pd

# test_data = pd.read_csv('dataset/test.csv')
#
# features = test_data.iloc[:, 1:]  # 第一个为ID去掉
#
# # 获取数字特征索引，去掉object型别的特征，比如字符串后剩下的就是数字了
# numeric_features_index = features.dtypes[features.dtypes != 'object'].index
#
# # 对数字特征进行标准化
# features[numeric_features_index] = features[numeric_features_index] \
#     .apply(lambda x: (x - x.mean()) / x.std())
# # 标准化后，所有数值特征的均值为0，所以可以用0来替换确实缺失的值
# features[numeric_features_index] = features[numeric_features_index].fillna(0)
#
# # 将非数字特征转化为数字特征，比如MSZoning可是RL或者RM，那么就对MSZoning进行one-hot编码,这一步会导致特征数增加
# features = pd.get_dummies(features, dummy_na=True)  # 将缺失值也当做一种情况进行one-hot编码
#
# test_features = torch.tensor(features.values, dtype=torch.float)

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

# 步骤六：预测
net = torch.nn.Linear(test_features.shape[-1], 1)

for name, param in net.named_parameters():
    with torch.no_grad():
        if 'weight' in name:
            param.data = torch.load('weight.pt')
        elif 'bias' in name:
            param.data = torch.load('bias.pt')

for name, param in net.named_parameters():
    pass

ret = net(test_features)
ret = ret.detach()
ret = ret.numpy()


test_data['SalePrice'] = pd.Series(ret.reshape(1, -1)[0])
submission = pd.concat((test_data['Id'], test_data['SalePrice']), axis=1)
submission.to_csv('submission.csv', index=False)


result = pd.read_csv('submission.csv')
print(result.shape)
print(result.iloc[0:10, :])