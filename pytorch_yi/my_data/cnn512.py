import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import time
import os
import cv2 as cv
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt


def default_loader(path):
    img = cv.imread(path)
    # img = Image.open(path).convert('RGB')
    return img


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)  # Input ->(1, 512, 512)
        self.conv2 = nn.Conv2d(6, 16, 5)  # ->(6, 256, 256)->(16, 128, 128)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)  # ->(16, 128, 128)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2x2) windows
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can specity a single number
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


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # input->(1, 512, 512)->(16, 256, 256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # input->(16, 256, 256)->(32, 126, 126)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # input->(32, 128, 128)->(64, 61, 61)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # input->(64, 64, 64)->(128, 28, 28)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(128 * 28 * 28, 1024)

    def forward(self, x):
        # self.num_flat_features可以查看每一层神经网络的size
        # self.num_flat_features(x)
        x = self.conv1(x)
        # self.num_flat_features(x)
        x = self.conv2(x)
        # self.num_flat_features(x)
        x = self.conv3(x)
        # self.num_flat_features(x)
        x = self.conv4(x)
        # self.num_flat_features(x)
        x = x.view(-1, self.num_flat_features(x))
        output = self.out(x)
        return output

    def num_flat_features(self, x):
        # size = x.size()[1:]
        size = x.size()
        # print(size)
        num_features = 1
        for s in size:
            num_features *= s
            # print(num_features)
        return num_features


class MyDataset(Dataset):

    def __init__(self, path, transform=None, loader=default_loader):
        self.train_img_path = path + 'my_data_A/'
        self.label_img_path = path + 'my_data_B/'
        # print(self.train_img_path)
        self.train_list = []
        # label_list = []
        for train_names in os.listdir(self.train_img_path):
            if train_names.find('png', len(train_names)-3, len(train_names)) != -1:

                self.train_list.append((self.train_img_path + train_names, self.label_img_path + train_names))
        # print(self.train_list)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        train, label = self.train_list[item]
        train_file = self.loader(train)
        # print(train_file)
        label_file = self.loader(label)
        if self.transform is not None:
            train_img = self.transform(train_file)
            label_img = self.transform(label_file)
        return train_img, label_img

    def __len__(self):
        return len(self.train_list)


if __name__ == '__main__':
    # img = cv.imread('./my_data_A/TB1..FLLXXXXXbCXpXXunYpLFXX.png', 0)
    # img_label = cv.imread('./my_data_B/TB1..FLLXXXXXbCXpXXunYpLFXX.png', 0)
    # img_label_tensor = Variable((torch.from_numpy(img_label).type(torch.FloatTensor)/255.).view(1, 1024))
    # print('img_label_tensor:', img_label_tensor)
    # # cv.imshow('t', img)
    # # cv.waitKey(0)
    # # cv.destroyAllWindows()
    # img_tensor = torch.from_numpy(img).type(torch.FloatTensor)/255.
    # img_tensor = img_tensor.view(1, 1, 512, 512)
    # img_variable = Variable(img_tensor, requires_grad=True)
    # # print(img_variable)
    #
    # net = Net()
    # # print(net)
    #
    # mynet = MyNet()
    # out = mynet(img_variable)
    # print(out)
    # mynet.zero_grad()
    # print('############################')
    # criterion = nn.MSELoss()
    #
    # loss = criterion(out, img_label_tensor)
    # print('loss:', loss)

    data = MyDataset(path = './', transform=transforms.ToTensor())
    # print(len(data))
    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    img = default_loader('./my_data_A/TB1.3pkLXXXXXXjaFXXunYpLFXX.png')
    # cv.imshow('i', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print(len(data_loader))
    net = MyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    loss_function2 = nn.CosineEmbeddingLoss()
    loss_function3 = nn.NLLLoss2d()
    loss_function4 = nn.CrossEntropyLoss()
    for epoch in range(5):
        for step, (train, label) in enumerate(data_loader):
            b_x = Variable(train)
            # print(label)
            b_y = Variable(label.byte().float()[:1, 2]).view(1, 1024)
            # cv.imshow('label', label)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            print('#####################')
            # for i in b_y.view(32, 32).data.numpy():
            #     print(i)


            print(b_x)
            # time.sleep(3)
            output = net(b_x)
            # print(output)
            loss = loss_function4(output, b_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                lang_data = output.data.view(32, 32)
                data_array = lang_data.numpy()
                # print(data_array)
                # cv.imshow('j', data_array*255)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                # for i in data_array:
                #     print(i + 1)

