# coding=utf-8
# user=hu
import os
import time
import random
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transform

# 读取标签和数字对应的字典
with open('./model/num_to_label_dict.txt', 'r') as f:
    num_to_label_dict = eval(f.read())

root = '/mnt/samsung/train_data/dataset_1/'

transform1 = transform.Compose([transform.ToTensor(), transform.Resize(64, 64)])


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


# test_cnn = torch.load('./model/cnn_one_label.pkl', map_location=lambda storage, loc: storage.cuda())
test_cnn = torch.load('./model/cnn_one_label.pkl')
# print(test_cnn)
test_cnn.cpu()
dirs_list = os.listdir(root)
dirs_num = len(dirs_list)
print(dirs_num)
for random_num in range(dirs_num):
    random_dir = random.randint(0, dirs_num-1)

    img_dir = os.path.join(root, dirs_list[random_dir])
    print(img_dir)
    img_path_list = os.listdir(img_dir)
    img_path_random = os.path.join(img_dir, img_path_list[random.randint(0, len(img_path_list)-1)])
    print(img_path_random)
    img = Image.open(img_path_random)
    plt.imshow(img)
    img_tensor = transform1(img)
    # print(img_tensor.view(1, 1, 64, 64))

    result = test_cnn(Variable(img_tensor.view(1, 1, 64, 64)))[0]
    # print(result[1])
    # print(result)
    pred_y = torch.max(result, 1)[1].data.numpy().squeeze()
    # pred_y 是ndarray数据需要将她变为int类型才能作为索引
    # print(int(pred_y))
    # cnn网络输出层label概率最高那一位的数值
    n = float(result.data[0][int(pred_y)])
    print('概率最大那一位的数据n是：', n)
    pred_label = num_to_label_dict[int(pred_y)]
    result_sum = int(torch.sum(result).data.numpy()[0])/653
    print('概率最高位数值: %s | result的平均数: %s' % (n, result_sum))
    print('概率是：', n/result_sum)
    print('pred_label:', pred_label)
    plt.title(int(pred_y))
    plt.show()
    # for img_path in img_path_list:
    #     print(img_path)
    # print(len(os.listdir(os.path.join(root, dirs))))


if __name__ == '__main__':
    pass