# coding=utf-8
# user=hu

import torch
from torch.autograd import Variable


def torch_y():
    tensor = torch.FloatTensor([[1, 2], [3, 4]])
    variable = Variable(tensor, requires_grad=True)
    print(tensor)
    print(variable)
    print(variable.data)
    t_out = torch.mean(tensor*tensor)
    v_out = torch.mean(variable*variable)
    print(t_out)
    print(v_out)
    print(variable.data.numpy())

if __name__ == '__main__':
    torch_y()
