# coding=utf-8
# user=hu
import re
import os
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms
import pytesseract

# 读取标签和数字对应的字典
with open('./model/num_to_label_dict.txt', 'r') as f:
    num_to_label_dict = eval(f.read())

cut_img_root = './scan_img/'
other_img_root = './cut_img/l_cut1/'
other_img = '../orc_test/jpg/'

img_path_list = os.listdir(cut_img_root)
img_path_list2 = os.listdir(other_img_root)
print(len(img_path_list))
transform1 = transforms.Compose([transforms.ToTensor()])


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


def no_scan():
    for img_path in img_path_list2:
        if len(img_path.split('_')[0]) == 1:
            print(img_path)
            img_path_full = os.path.join(other_img_root, img_path)
            img = Image.open(img_path_full).convert('L')
            img_64 = img.resize((64, 64))
            img_64_tensor = transform1(img_64)
            result = scan_cnn(Variable(img_64_tensor.view(1, 1, 64, 64)))[0]
            pred_y = int(torch.max(result, 1)[1].data.numpy().squeeze())
            pred_label = num_to_label_dict[pred_y]
            print('pred_label is:', pred_label)
            plt.imshow(img_64)
            plt.title(pred_label)
            plt.show()


def scan_with_contours():
    for img_path in img_path_list:
        img_path = os.path.join(cut_img_root, img_path)
        print('Image path is:', img_path)
        img_pil = Image.open(img_path).convert('RGB')
        img_cv = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
        img_cv = cv.resize(img_cv, (384, 384))
        img_cv_gray = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(img_cv_gray, 127, 255, 0)
        im2, contours, hierarchy = cv.findContours(thresh, 1, 2)
        print('contours is:', contours)
        print('hierarchy is:', hierarchy)
        for cnt in contours:
            # cnt = contours[0]
            area = cv.contourArea(cnt)
            size = img_cv.shape
            full_area = size[0] * size[1]
            if 100 < area < 300:
                # 框出平直的矩形框
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # 框出最小矩形块
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                # cv.drawContours(img_cv, [box], 0, (0, 0, 255), 2)
        plt.imshow(img_cv)
        plt.show()
        # cv.imshow('img_cv', img_cv)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # time.sleep(5)


scan_cnn = torch.load('./model/cnn_one_label.pkl')
scan_cnn.cpu()


def scan_with_box():
    for img_path in img_path_list:
        img_path = os.path.join(cut_img_root, img_path)
        print('img_path is:', img_path)

        t = time.time()
        img_pil = Image.open(img_path).convert('L')
        img_pil.resize((512, 512))
        for y in range(1, 460, 8):
            for x in range(1, 460, 8):
                box = (x, y, x+16, y+16)
                scan_img = img_pil.crop(box)
                scan_img_resize = scan_img.resize((64, 64))
                scan_img_tensor = transform1(scan_img_resize)
                result = scan_cnn(Variable(scan_img_tensor.view(1, 1, 64, 64)))[0]
                # pred_y 是预测的label在第几位
                pred_y = int(torch.max(result, 1)[1].data.numpy().squeeze())
                # n 是预测的label所在位置的数值的float类型
                n = float(result.data[0][pred_y])
                # n2 是预测结果每一位的平均值
                n2 = int(torch.sum(result).data.numpy()[0])/653
                pred_label = num_to_label_dict[pred_y]
                if n/n2 < 0.2 and 64 < ord(pred_label) < 123:
                    pass
                    print('n is: %.2f | n2 is %.2f' % (n, n2))
                    print('概率为： %.2f' % (n/n2))
                    print('label is:', pred_label)
                    plt.imshow(scan_img_resize)
                    plt.title(pred_label)
                    plt.show()

                # time.sleep(1)
        print('# #  '*20)
        print('time:', time.time()-t)


        # plt.imshow(img_pil)
        # plt.show()
        # time.sleep(1)


def thresh_img():
    for img_path in img_path_list:
        img_full_path = os.path.join(cut_img_root, img_path)
        print('img_full_path is:', img_full_path)
        img = cv.imread(img_full_path, 0)
        for i in range(87, 168, 10):
            ret, thresh = cv.threshold(img, i, 255, 1)
            plt.subplot(3, 3, int(i/10-7))
            plt.imshow(thresh, cmap='gray')
            plt.title(i)
        plt.show()
        # cv.imshow(img_path, thresh)
        # cv.waitKey(0)
        # cv.destroyAllWindows()


def img_2_text():
    for name in img_path_list:
        img_full_name = os.path.join(cut_img_root, name)
        img = Image.open(img_full_name).convert('L')
        threshold = 150
        table2 = []
        table = []
        for j in range(256):
            if j < threshold:
                table.append(0)
                table2.append(1)
            else:
                table.append(1)
                table2.append(0)
        out = img.point(table, '1')
        out2 = img.point(table2, '1')
        # out2.show()
        # out.show()

        text = pytesseract.image_to_string(out, lang='chi_sim')
        text2 = pytesseract.image_to_string(out2)
        print('text:\n', text)
        print('text2:\n', text2)
        plt.imshow(img)
        plt.show()
        time.sleep(3)


def only_contours():
    img_list = os.listdir(other_img)
    for img_name in img_list:
        if img_name.split('.')[-1] in '.jpg, .png, .bmg, .jpeg':
            print(img_name)
            img_full_name = os.path.join(other_img, img_name)
            img = cv.imread(img_full_name)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, 145, 255, 0)
            _, contours, _ = cv.findContours(thresh, 1, 2)
            for cnt in contours:
                area = cv.contourArea(cnt)

                if 50 < area < 500:
                    x, y, w, h = cv.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    if 0.45 < aspect_ratio < 1.1:
                        print(aspect_ratio)
                        rect = cv.minAreaRect(cnt)
                        # cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        box = cv.boxPoints(rect)
                        box = np.int0(box)
                        crop_img = gray[y:y+h, x:x+w]
                        crop_img = cv.resize(crop_img, (64, 64))
                        img_tensor = torch.from_numpy(np.float32(crop_img)/255.)
                        # print(img_tensor)

                        ###################识别
                        # img_tensor = transform1(img_tensor)
                        result = scan_cnn(Variable(img_tensor.view(1, 1, 64, 64)))[0]
                        # pred_y 是预测的label在第几位
                        pred_y = int(torch.max(result, 1)[1].data.numpy().squeeze())
                        # n 是预测的label所在位置的数值的float类型
                        n = float(result.data[0][pred_y])
                        # n2 是预测结果每一位的平均值
                        n2 = int(torch.sum(result).data.numpy()[0]) / 653
                        pred_label = num_to_label_dict[pred_y]
                        if n / n2 < 0.2 and 64 < ord(pred_label) < 123:
                            pass
                        print('n is: %.2f | n2 is %.2f' % (n, n2))
                        print('概率为： %.2f' % (n / n2))
                        print('label is:', pred_label)
                        plt.imshow(crop_img)
                        plt.title(pred_label)
                        plt.show()

                        # cv.imshow('crop_img', crop_img)
                        # cv.waitKey(0)
                        # cv.destroyAllWindows()
                        # cv.drawContours(img, [box], 0, (0, 255, 0), 2)

            # cv.imshow(img_name, img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()


if __name__ == '__main__':
    # scan_with_box()
    # thresh_img()
    # img_2_text()
    # no_scan()
    only_contours()
