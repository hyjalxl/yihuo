# -*- coding: utf-8 -*-

from PIL import Image
import pytesseract
import cv2
import os

import matplotlib.pyplot as plt

def img2text(img):
    # im = cv2.imread(file_name)
    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('im2', im)
    # cv2.waitKey(0)
    
    text = pytesseract.image_to_string(img, lang='chi_sim')
    
    print(text)


def get_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        
        return files


def get_img(file_name):
    return Image.open('./jpg/' + file_name).convert('L')


def img_crop(img):
    print('img_crop is run')
    # box = (100, 100, 200, 200)
    # img.show()
    #
    # region = img.crop(box)
    # region.show()
    # print(img.size)
    # print(region.size)
    # print(img.split()) # 打印图像的各个通道

    img_x, img_y = img.size  # 获取图片的大小
    print(img_x, img_y)

    # 使用100X100滑块进行滑动
    for h in range(1, img_y, 20):
        for i in range(20, img_y-10, 20):
            print(i, h, i+100, h+100)
            box = (i, h, i+100, h+100)
            crop_img = img.crop(box)
            # crop_img.show()
            # crop_img.close()
            img2text(crop_img)



if __name__ == '__main__':
    print('hello')
    file_name_list = get_file_name('./jpg')

    img = get_img('j.jpg')
    img_crop(img)

    # for file_name in file_name_list:
    #     s = '.jpg'
    #     if s in file_name:
    #         print(file_name)
    #         img = get_img(file_name)
    #         img2text(img)
    #         print('>' * 50)
    # img = get_img('j.jpg')
    # img_crop(img)
