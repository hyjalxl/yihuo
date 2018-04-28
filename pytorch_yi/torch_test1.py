# coding=utf-8
# user=hu

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import string

all_characters = string.printable
n_characters = len(all_characters)






def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = ord(string[c])
    return Variable(tensor)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # Max pooling over a(2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
    print(char_tensor('123abc!!!胡阳杰'))
    # print(ord('胡'))
    # print(chr(2))
