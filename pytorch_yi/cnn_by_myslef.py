# coding=utf-8
# user=hu

import os
import torch
import numpy as np
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.autograd import Variable

# Hyper Parameters
EPOCH = 1  # Train the training data n times.
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),  # Converts a PLI.Image or numpy.ndarry to
                                                  # torch.FloatTensor of shape(C x H x W)and normalize in the range[0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example

print(train_data.train_data.size())               # (torch.Size([60000, 28, 28])
print(train_data.train_labels.size())
# print(train_data.train_data[1].numpy())
# for lable in train_data.train_labels:
#     print(lable)
# plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[2])
# plt.show()

# Data loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# print(train_loader)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000].cuda()/255.
test_y = test_data.test_labels[:2000].cuda()

# print(test_x.size())
# print(test_y[2])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,                      # input height
                out_channels=16,                    # n_filters 过滤器层数
                kernel_size=5,                      # filter size 过滤器尺寸
                stride=1,                           # filter movement/step 过滤器步长
                padding=2                           # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                                      # output shape(16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),            # choose max value in 2x2 area, output shape(16, 14, 14)
        )

        self.conv2 = nn.Sequential(                 # input shape(16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),             # output shape(32, 14, 14)
            nn.ReLU(),                              # activation
            nn.MaxPool2d(2),                        # output shape(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)    # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
# print(cnn)  # net architecture
cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
no_saved = True
# print(os.listdir('./model/'))
if os.listdir('./model/'):
    no_saved = False
if no_saved:
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x).cuda()   # batch x
            b_y = Variable(y).cuda()   # batch y

            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(step)
            if step % 50 == 0:
                # print()
                # print(torch.sum(x[0]), y[0], output)
                # print(b_y, output)
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                print('Epoch:', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
if not os.listdir('./model/'):
    torch.save(nn, './model/mnist_net.pkl')
if no_saved:
    test_output, _ = cnn(test_x[19:20])
else:
    load_cnn = torch.load('./model/mnist_net.pkl')
    test_output, _ = load_cnn(test_x[19:20])
img_data = test_x[19:20].data.cpu().numpy()
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
# print(img_data)
plt.imshow(int(img_data*255))
plt.show()
print(pred_y)
print(test_y[19:20])



if __name__ == "__main__":
    pass