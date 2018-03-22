# coding=utf-8
# user=hu
import os
import matplotlib.pyplot as plt


def get_txt_data():
    data_list = []  # data_list存储形式为[[一行坐标数据],[一行坐标数据],[一行坐标数据],[一行坐标数据],.....]
    dir = r'D:\machine_learning\tianchi2018\train\txt_1000'

    # print(dir)
    for root, dirs, files in os.walk(dir):
        for file_name in files:
            s = dir + '\\' + file_name  # 构造文件绝对路径
            # print(s)
            with open(s, 'r', encoding='UTF-8') as f:
                line_list = f.readlines()  # line_list是每个文本文件的所有行数据列表
                # print(line_list)
                for lines_text in line_list:
                    for line in lines_text.splitlines():
                        text_list = line.split(',')
                        text_list.pop()
                        # print(data_list)
                        # data_list = line_text.split(',')
                        # data_list.pop()
                        data_list.append(text_list)
                        # print(data_list)
        print(len(data_list))
        return data_list


if __name__ == '__main__':
    data_l = get_txt_data()  # dtat_l是所有文字框的一个列表形式是[[一行坐标数据],[一行坐标数据],[一行坐标数据],[一行坐标数据],.....]
    wide_dict = {}  # 创建文字框的宽的字典
    for d in data_l:  # d是每一个文字框的列表样式是[左上x,左上y,左下x,左下y,右上x,右上y,右下x,右下y]
        box_wide = int(((float(d[0])-float(d[2]))**2 + (float(d[1])-float(d[3]))**2) ** 0.5)
        if wide_dict.get(box_wide):
            wide_dict[box_wide] = wide_dict[box_wide] + 1
        else:
            wide_dict[box_wide] = 1

    # print(wide_dict)

    for i in range(8):
        plt.scatter(wide_dict.keys(), wide_dict.values())
    plt.show()
