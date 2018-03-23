# coding=utf-8
# user=hu

import os
import cv2
import numpy as np
import random

import torch
import torch.nn as nn
import torch.utils.data as dataf
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

# 训练参数
cuda = True
train_epoch = 20
train_lr = 0.01
batchsize = 5

# 训练集路径
test_path = 'D:\\mini_train\\mini_data\\test\\'
train_path = 'D:\\mini_train\\mini_data\\train\\'

# 路径数据
all_path = []


def load_data(data_path):
    signal = os.listdir(data_path)
    for fsingal in signal:
        filepath = data_path+fsingal

        filename = os.listdir(data_path)
    print(filename)
    for fname in filename:
        ffpath = data_path + fname
        # print(ffpath)
        path = [fname, ffpath]
        all_path.append(path)
    # 数据集大小
    count = len(all_path)
    data_x = np.empty((count, 1, 28, 28), dtype='float32')
    data_y = []

    # 打乱顺序
    random.shuffle(all_path)
    i = 0
    # 读取图片 训练集应该是灰度图 最终结果是i*i*i
    # 分别表示：batch大小，通道数，像素矩阵
    print(len(all_path))
    for item in all_path:
        # print(item[0], '\n', item[1])
        img = cv2.imread(item[1], 0)
        img = cv2.resize(img, (28, 28))
        arr = np.asarray(img, dtype='float32')
        data_x[i, :, :, :] = arr
        i += 1
        data_y.append(1)
    data_x = data_x / 255
    data_y = np.asarray(data_y)

    # lener = len(all_path)
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    dataset = dataf.TensorDataset(data_x, data_y)
    loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

    # print(data_y)
    return loader

# train_load = load_data(train_path)
test_load = load_data(test_path)

if __name__ == '__main__':
    pass





























