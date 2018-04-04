# coding=utf-8
# user=hu
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset


import cv2 as cv


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 2, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x


def get_file_list():
    img_list = os.listdir('./cnn513_a')
    return img_list


if __name__ == '__main__':
    net = Net()
    print(net)
    file_list = get_file_list()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss = nn.NLLLoss2d()
    data_list = []
    for i in range(100):
        for img_name in file_list:
            # img = cv.imread('./cnn513_a/' + img_name)
            # img_tensor = torch.from_numpy(img).type(torch.FloatTensor)/255.
            # # img_tensor = img_tensor.view(1, 3, 512, 512)
            # # print(img_tensor)
            # print(img_tensor)
            # cv.imshow('in', img_tensor.int().numpy()*255)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # img_v = Variable(img_tensor)
            #
            # label_img = cv.imread('./cnn513_b/' + img_name, 0)
            # label_tensor = torch.from_numpy(label_img)/255
            #
            # label_v = Variable(label_tensor.long().view(1, 126, 126))
            # # print(label_v)
            # # print(img_v)
            a = cv.imread('./cnn513_a/' + img_name, 0)
            b = cv.imread('./cnn513_b/' + img_name, 0)
            a_tensor = torch.from_numpy(a).type(torch.FloatTensor)/255.
            # b_tensor = torch.from_numpy(b).type(torch.FloatTensor)/255.
            b_tensor = torch.from_numpy(b)
            a2 = a_tensor.view(1, 512, 512)
            b2 = b_tensor.view(1, 126, 126)
            torch_dataset = TensorDataset(a2, b2)
            data_list.append(torch_dataset)
            # print(a_tensor.size())
            # print(b_tensor.size())
        loader = DataLoader(torch_dataset)
        # print(loader)

        for a, b in loader:
            a = Variable(a, requires_grad=True)
            b = Variable(b)
            result = net(a.view(1, 1, 512, 512))
            # print(result)
            loss_result = loss(result, b.long()/255)
            loss_result.backward()
            # print(loss_result.data)
            optimizer.zero_grad()

            optimizer.step()
        # print(loss_result)
        # cv.imshow('result', result.data.long()[0][0].numpy()*255)
        # cv.imshow('result', result.data.long()[0][1].numpy()*255)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        print(result[0][0].data)
        print(result[0][1].data)

        print('################')
        print(loss_result)
