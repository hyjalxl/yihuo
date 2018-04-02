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
import time


def draw_box(img_name):
    """
    原计划将图像按照方框进行剪切，后更改方案为按照方框进行二值化显示
    :param img_name: 需要处理的图片路径
    :return: 
    """
    # 读入图像
    img = cv.imread(img_name, 1)
    img2 = cv.resize(img, (256, 256))
    cv.imwrite('./t_A/' + img_name[0: len(img_name) - 3] + 'png', img2)
    # 读取和照片同文件名的txt文件，将文本中的每行数据加到box_list列表
    with open(img_name + '.txt', 'r', encoding='UTF-8') as f:
        box_list = f.readlines()
    # box_list是列表内每行数据
    for data_line in box_list:
        # print(data_line)
        # data_list是方框顶点坐标和对应文本的列表
        # data_list的样子['6.02', '9.04', '4.02', '33.19', '73.82', '32.19', '74.82', '5.04', '聚源\n']
        data_list = data_line.split(',')
        # print(data_list)
        box = [[float(data_list[0]), float(data_list[1])], [float(data_list[2]), float(data_list[3])], [float(data_list[4]), float(data_list[5])], [float(data_list[6]), float(data_list[7])]]
        np_box = np.array(box, np.int32)
        # #############################上面是获取标注框，下面为处理图像部分#######################
        # print(np_box)
        # print('#'*50)
        pts = np_box.reshape((- 1, 1, 2))
        # print(pts)
        # cv.polylines 勾画多边形
        cv.polylines(img, [pts], True, (0, 0, 255), lineType=8, thickness=1)
        # cv.fillpoly 填充颜色
        cv.fillPoly(img, [pts], (0, 255, 0,))

        # 根据填充的颜色二值化图像
        # ret, thresh = cv.threshold(img, 254, 255, 0)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        l_blue = np.array([50, 254, 254])
        h_blue = np.array([60, 255, 255])

        mask = cv.inRange(hsv, l_blue, h_blue)
        # img2 = cv.cvtColor(mask, cv.COLOR_HSV2BGR)


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
    # print(len(mask))
    # 按位取反使图像泛白显示
    # res = cv.bitwise_not(mask)
    mask = cv.resize(mask, (256, 256))
    cv.imwrite('./t_B/' + img_name[0: len(img_name)-3] + 'png', mask)

    # cv.imshow(img_name, mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    pass


def get_img_name():
    # for name_lists in [files for _, _, files in os.walk('./')]:
    #
    #     for name in name_lists:
    #         print()
    img_name_list = []
    for name in os.listdir('./'):
        if name.find('jpg', len(name)-3, len(name)) != -1:
            img_name_list.append(name)
    return img_name_list

if __name__ == '__main__':

    img_name_list = get_img_name()
    for img_name in img_name_list:
        # img_name = 'T1A3mtFspcXXXXXXXX_!!0-item_pic.jpg'
        draw_box(img_name)
        time.sleep(0.5)
