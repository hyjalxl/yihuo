"""

"""
# coding=utf-8
# user=hu
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

txt_dir = 'D:\\machine_learning\\tianchi2018\\train_1000\\txt_1000'
imgs_dir = 'D:\\machine_learning\\tianchi2018\\train_1000\\image_1000\\'


def get_txt_data(t_dir):
    """
    获取txt文件夹下所有txt文件内的数据
    :param t_dir: 
    :return: data_list是所有标注框四个顶点坐标值的列表，
    """
    data_list = []  # data_list存储形式为[[一行坐标数据],[一行坐标数据],[一行坐标数据],[一行坐标数据],.....]

    # print(dir)
    for root, dirs, files in os.walk(t_dir):
        for file_name in files:
            s = t_dir + '\\' + file_name  # 构造文件绝对路径
            # print(s)
            with open(s, 'r', encoding='UTF-8') as f:
                line_list = f.readlines()  # line_list是每个文本文件的所有行数据列表
                # print(line_list)
                for lines_text in line_list:
                    for line in lines_text.splitlines():
                        text_list = line.split(',')
                        pop_data = text_list.pop()
                        if 'N7100' in pop_data:
                            print(file_name)
                        # print(data_list)
                        # data_list = line_text.split(',')
                        # data_list.pop()
                        data_list.append(text_list)
                        # print(data_list)
        print(len(data_list))
        return data_list, files


def get_img(img_path):
    """
    获取D:\machine_learning\tianchi2018\train\image_1000目录下的图片
    :param img_dir: 文件的目录
    :return: 
    """
    pass
    img = cv.imread(img_path, 0)
    # print(type(img))
    # print(img.size)
    ret, thresh = cv.threshold(img, 127, 255, 0)
    a, contour, b = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(contour[0])
    box_list = [[[67, 400]], [[37, 400]], [[200, 200]], [[300, 300]]]
    cv.drawContours(img, np.array(box_list), -1, (0, 255, 0), 4)
    cv.imshow('img_dir', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    # txt_dir是txt文件数据文件目录
    data_l, files = get_txt_data(txt_dir)  # data_l是所有文字框的一个列表形式是[[一行坐标数据],[一行坐标数据],[一行坐标数据],[一行坐标数据],.....]
    wide_dict = {}  # 创建文字框的宽的字典
    for d in data_l:  # d是每一个文字框的列表样式是[左上x,左上y,左下x,左下y,右上x,右上y,右下x,右下y]
        box_wide = int(((float(d[0])-float(d[2]))**2 + (float(d[1])-float(d[3]))**2) ** 0.5)
        if wide_dict.get(box_wide):
            wide_dict[box_wide] = wide_dict[box_wide] + 1
        else:
            wide_dict[box_wide] = 1

    # print(files)
    i = 1
    for img_name in files:
        if i > 2:
            break
        img_dir = imgs_dir + img_name[0:len(img_name) - 3] + 'jpg'
        print(img_dir)
        # get_img(img_dir)
        i += 1

    plt.scatter(wide_dict.keys(), wide_dict.values())
    plt.show()
