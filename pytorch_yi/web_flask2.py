# coding=utf-8
# user=hu
import os
import time
import cv2 as cv
import numpy as np
import torch
from flask import Flask, request
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable

# 读取标签和数字对应的字典
with open('./model/num_to_label_dict.txt', 'r') as f:
    num_to_label_dict = eval(f.read())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './scan_img/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


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


scan_cnn = torch.load('./model/cnn_one_label.pkl')
scan_cnn.cpu()


# For a giver file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def hello_world():
    return 'hello'


@app.route('/upload', methods=['POST'])
def upload():
    upload_file = request.files['image01']
    if upload_file and allowed_file(upload_file.filename):
        img_pil = Image.open(upload_file)
        img_cv = cv.cvtColor(np.asarray(img_pil), cv.COLOR_RGB2GRAY)
        ret, thres = cv.threshold(img_cv, 145, 255, 0)
        _, contours, _ = cv.findContours(thres, 1, 2)
        str_result = ''
        for cnt in contours:
            area = cv.contourArea(cnt)
            if 50 < area < 600:
                x, y, w, h = cv.boundingRect(cnt)
                aspect_ratio = float(w)/h
                if 0.45 < aspect_ratio < 1.1:
                    crop_img = img_cv[y:y+h, x:x+w]
                    crop_img = cv.resize(crop_img, (64, 64))
                    img_tensor = torch.from_numpy(np.float32(crop_img)/255.)
                    result = scan_cnn(Variable(img_tensor.view(1, 1, 64, 64)))[0]
                    # pred_y 是预测的label在第几位
                    pred_y = int(torch.max(result, 1)[1].data.numpy().squeeze())
                    # n 是预计的label所在位置对应的数值
                    n = float(result.data[0][pred_y])
                    # pred_label是预测的标签
                    pred_label = num_to_label_dict[pred_y]
                    if n > -10:
                        print('概率是：%.2f | 标签是：%s' % (n, pred_label))
                        str_result = str_result + str(pred_label)
        # plt.imshow(img_cv, cmap='gray')
        # plt.title(upload_file.filename)
        # plt.show()
        print(str_result)
        return 'ok :' + str_result
    else:
        return 'no'


if __name__ == '__main__':
    app.run()
    pass