import torch
import torch.nn as nn
import torch.autograd as autograd


m = nn.Conv2d(16, 32, (3, 3)).float()
loss = nn.NLLLoss2d()
# input is size N x C x height x width
input = autograd.Variable(torch.randn(3, 16, 10, 10))
# each element in target has to have 0 <= value < c
target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
output = loss(m(input), target)
output.backward()
print('m:', m)
print('#########################################')
print('input:', input)
print('#########################################')
print('target:', target)
print('#########################################')
print('output:', output)
print('#########################################')
print('m.input', m(input))
print(m(input)[:, 2])