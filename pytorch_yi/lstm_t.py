# coding=utf-8
# user=hu
# 使用MNIST数据测试lstm循环神经网络
import torch
import time
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Hyper Parameters

EPOCH = 100
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

# Mnist digital dataset
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# Data Loader for easy mini-batch return in training
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]

# print(test_x[0])
# print(test_y[0])
print(train_data[0][0])
print(train_data[0][1])


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=4,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


def main():

    rnn = RNN().cuda()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    # training and testing
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, 28, 28)).cuda()
            b_y = Variable(y).cuda()

            ######################
            # 查看训练数据
            # print('x:\n', x)
            # print('y:\n', y)
            # print('b_x:\n', b_x)
            # print('b_y:\n', b_y)
            # time.sleep(5)
            ######################

            out = rnn(b_x)
            # print(out)
            # time.sleep(5)
            loss = loss_func(out, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1000 == 1:
                print('out:', torch.max(out, 1)[1].int().view(64, 1))
                # print('out:', torch.sum(out[: 1]))
                # print('#'*100)
                # print(out[1:])
                # print(out[:1])
                # print(out[2:])
                # print(out[:2])
                # print(out)
                print('b_y:', b_y.int())

                print(loss)
            # time.sleep(0.5)


if __name__ == '__main__':
    pass