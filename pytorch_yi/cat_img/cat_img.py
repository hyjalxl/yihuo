"""
将图片剪切成编注区和非标注区两个部分图像大小暂且不同，
第一步将图片文字区域用方框画出，
第二部将文字区另存到1文件夹里面
"""
# coding=utf-8
# user=hu

import os
import cv2 as cv
import numpy as np


def draw_box(img_path):
    """
    
    :param img_path: 需要剪切的图片路径
    :return: 
    """
    img = cv.imread(img_path, 0)
    ret, thresh = cv.threshold(img, 127, 255, 0)
    img2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rect = cv.minAreaRect(contours[0])
    print('rect is: ', rect)
    box = cv.boxPoints(rect)
    print('before to numpy box is: ', box)
    box = np.int0(box)
    print('after to numpy bos is:', box)

    # 读取和照片同文件名的txt文件，将文本中的每行数据加到box_list列表
    with open(img_path+'.txt', 'r', encoding='UTF-8') as f:
        box_list = f.readlines()

    # box_list列表内每行数据
    for data_line in box_list:
        # print(data_line)
        data_list = data_line.split(',')

        # print(data_list)

    cv.imshow(img_path, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    pass

if __name__ == '__main__':
    img_path = 'T1_kR_XadkXXcDMjo8_100900.jpg'
    draw_box(img_path)
