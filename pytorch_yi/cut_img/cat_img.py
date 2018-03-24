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
    # 读入图像
    img = cv.imread(img_path, 1)

    # 读取和照片同文件名的txt文件，将文本中的每行数据加到box_list列表
    with open(img_path + '.txt', 'r', encoding='UTF-8') as f:
        box_list = f.readlines()
    # box_list列表内每行数据
    for data_line in box_list:
        # print(data_line)
        # data_list是方框顶点坐标和对应文本的列表
        # data_list的样子['6.02', '9.04', '4.02', '33.19', '73.82', '32.19', '74.82', '5.04', '聚源\n']
        data_list = data_line.split(',')
        # print(data_list)
        box = [[float(data_list[0]), float(data_list[1])], [float(data_list[2]), float(data_list[3])], [float(data_list[4]), float(data_list[5])], [float(data_list[6]), float(data_list[7])]]
        np_box = np.array(box, np.int32)
        print(np_box)
        pts = np_box.reshape((- 1, 1, 2))
        cv.polylines(img, [pts], True, (0, 0, 255), thickness=2)


    # 以下代码是找图片轮廓，此函数暂时用不到
    # ret, thresh = cv.threshold(img, 127, 255, 0)
    # img2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # rect = cv.minAreaRect(contours[0])
    # print('rect is: ', rect)
    # box2 = cv.boxPoints(rect)
    # print('before to numpy box is: ', box2)
    # box2 = np.int0(box2) # after to numpy bos is: <class 'numpy.ndarray'>
    # print('after to numpy bos is:', '\n', box2)
    # # cv.drawContours(img, [box2], 0, (0, 0, 255), 2)

    # 下面是图像显示部分
    cv.imshow(img_path, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    pass

if __name__ == '__main__':
    img_path = 'T1b7UvXohaXXXXXXXX_!!0-item_pic.jpg'
    draw_box(img_path)
