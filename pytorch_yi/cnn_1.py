# coding=utf-8
# user=hu

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision

torch.manual_seed(1)  # 随机种子

EPOCH = 1  # 训练整批数据多少次
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MINIST = False



if __name__ == '__main__':
    pass