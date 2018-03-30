# coding=utf-8
# user=hu

"""
test pytorch_gnn
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D



def np_and_torch():
    np_data = np.arange(100).reshape([10, 10])
    torch_data = torch.from_numpy(np_data)
    print(torch.Size(100,100))

    print(np_data)
    print(torch_data)


# out=start+(end-start)*weight
def lerp():
    """
    torch.floor(input, out=None) → Tensor
床函数: 返回一个新张量，包含输入input张量每个元素的floor，即不小于元素的最大整数。

参数：

input (Tensor) – 输入张量
out (Tensor, optional) – 输出张量
例子：

>>> a = torch.randn(4)
>>> a

 1.3869
 0.3912
-0.8634
-0.5468
[torch.FloatTensor of size 4]

>>> torch.floor(a)

 1
 0
-1
-1
[torch.FloatTensor of size 4]
    :return: 
    """
    star = torch.arange(1)
    end = torch.Tensor(1).fill_(5)
    print(star)
    print(end)
    resquest = torch.lerp(star, end, 2)
    print(resquest)


# 平方根的倒数
def rsqrt():
    """
    (2) torch.rsqrt(input)

返回平方根的倒数
    :return: 
    """
    x = torch.arange(1, 3)
    resquest = torch.rsqrt(x)
    print(resquest)


# 返回平均值
def mean():
    x = torch.arange(4)
    r = torch.mean(x)
    print(x)
    print('#'*100)
    print(r)


def prod():
    x = torch.arange(1, 3)
    r = torch.prod(x)
    print(x)
    print("#" * 100)
    print(r)


def activation_t():
    x = torch.linspace(-5, 5, 20)
    # print(x)
    x_variable = Variable(x)
    print(x_variable)


def regression():
    x = torch.linspace(-1, 1, 10)
    z = torch.rand([10, 10, 10])
    print(z)
    # x = torch.unsqueeze(x.size)
    # print(x)
    # for i in range(2):
    #     x = torch.unsqueeze(x * x, dim=1)
    # print(torch.unsqueeze(x), torch.unsqueeze(x))
    # print(torch.rand(x.size()))
    y = x.pow(2) + 0.2*torch.rand(x.size())
    pass


def plt_3d_show():
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(-4, 4, 0.25)
    y = np.arange(-4, 4, 0.24)
    print(x)
    print('#' * 10)
    print(y)
    print('#' * 10)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2, + y**2)
    z = np.sin(r)
    print(x)
    ax.plot_surface(x, y, z)

    plt.show()


if __name__ == '__main__':
    # np_and_torch()
    # lerp()
    # rsqrt()
    # mean()
    # prod()
    # activation_t()
    # regression()
    plt_3d_show()
    pass
