# coding=utf-8
# user=hu
"""
保存鼠标数据到mouse_date文件夹
"""

from pymouse import PyMouse
import time


if __name__ == '__main__':
    # init()
    m = PyMouse()
    for num in range(10):
        with open('./mouse_data/' + str(num) + '.mdata', 'w') as f:
            for i in range(100):
                coordinate = m.position()
                time.sleep(0.03)
                x, y = coordinate
                f.write(str(x) + ',' + str(y) + ';')
