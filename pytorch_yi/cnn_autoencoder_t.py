# coding=utf-8
# user=hu
import numpy as np
import torch
import time
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt

data_root = '/mnt/hu/machine_learning/tianchi2018/dataset_1'
data_root_ssd = '/mnt/samsung/train_data/dataset_1'

img_data = torchvision.datasets.ImageFolder(data_root_ssd, transform=transforms.Compose(
                                                [   # transforms.Scale(256),
                                                    # transforms.CenterCrop(224),
                                                    transforms.Grayscale(1),
                                                    transforms.ToTensor(),
                                                    # transforms.Resize(32*32)
                                                ]),)
print(len(img_data))
# print(len(img_data.imgs)
# print(img_data.size())


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('batch from data')

load_model = torch.load('./model/cnn_one_label.pkl')
time.sleep(10)

s = []
n = 0
num_to_label_dict = {}
for i in range(len(img_data)):
    # if img_data[i][1] not in s:
    #     s.append(img_data[i][1])
    # print(img_data[i][1])
    # time.sleep(0.01)
    if img_data[i][1] == n:
        num_to_label_dict[n] = img_data.imgs[i][0].split('/')[-2]
        print(img_data.imgs[i][0].split('/')[-2])
        # show_batch(img_data[i][0])
        # plt.title(img_data.imgs[i])
        # plt.show()
        n += 1
        # print(img_data[i][0])
with open('./model/num_to_label_dict.txt', 'w') as f:
    f.write(str(num_to_label_dict))
    print('Save num_to_label_dict.txt is ok!')
print(s)
time.sleep(6)
data_loader = torch.utils.data.DataLoader(img_data, batch_size=200, shuffle=True)
# test_data = img_data[:500]
# print(img_data[0][0])





# show_batch(img_data[0][0])
# plt.show()
print(len(data_loader))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(64*8*8, 653)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x)
        x = self.conv3(x)
        # print(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


cnn = CNN()
cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
# time.sleep(1)

t = int(time.time())
for epoch in range(5):
    for step, (x, y) in enumerate(data_loader):
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()
        # print(b_x)
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print('time:', int(time.time())-t)
            t = int(time.time())
            print('Epoch:', epoch, '| train loss: %.4f' % loss.data[0])
            print(y.numpy()[:36])
            # print(output)
            pred_y = torch.max(output, 1)[1].data.cpu().numpy().squeeze()
            print(pred_y[:36])
            # show_batch(b_x[0])
            plt.show()
            # input2 = input('input:')
torch.save(cnn, './model/cnn_one_label.pkl')




if __name__ == '__main__':
    pass