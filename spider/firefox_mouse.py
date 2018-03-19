# coding=utf-8
# user=hu

from pymouse import PyMouse
from selenium import webdriver
from selenium.webdriver import ActionChains
from pymouse import PyMouse

import time
import random


def sliding_mouse(msdata):

    driver = webdriver.Firefox()
    driver.get('http://localhost/mouse.html')
    dr1 = ActionChains(driver)
    time.sleep(3)
    dr1.click_and_hold(driver.find_element_by_id('spliding')).release()
    # for i in range(7):
    #     dr1.move_by_offset(30, 50)
    #     dr1.move_by_offset(10, -30)
    dr1.perform()
    m = PyMouse()
    for i in range(100):
        m.move(msdata[i][0], msdata[i][1])
        time.sleep(0.03)
    time.sleep(9)
    driver.quit()


def get_msdata():
    num = random.randint(1, 9)
    print num
    with open('./mouse_data/' + str(num) + '.mdata', 'r') as f:
        data = f.readline()
        # print type(data)
    l = data.split(';')
    coordinate_list = []
    for i in l:
        if len(i): # l 最后一个元素是空因此判断最后是不是为空
            coordinate_str_list = i.split(',')

            # print coordinate_str_list
            coordinate_list.append((int(coordinate_str_list[0]), int(coordinate_str_list[1])))
            # print coordinate_str_list[0], coordinate_str_list[1]
    return coordinate_list


if __name__ == "__main__":

    msdata = get_msdata()
    # print msdata
    sliding_mouse(msdata)
