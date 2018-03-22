# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np


def get_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root, '\n', dirs, '\n', files)
    return files
    

def show_img(img_name):
    img = cv2.imread('./jpg/' + img_name, 0)
    img2 = img[1:1, 30:30]
    cv2.imshow(img_name, img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def show_img2(img_name):
    img = cv2.imread(img_name)
    img2 = img[1:2, 33:33]
    cv2.imshow(img_name, img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def img_contours():
    im = cv2.imread('./jpg/n.jpg')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num = 1
    for contour in contours:
        num += 1
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if area > 200 and area < 50000:
            cv2.drawContours(im, [box], 0, (0, 255, 0), 2)
            print(num, area)
            cv2.imshow('contour', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    #file_name_list = get_file_name('./jpg')
    #for img_name in file_name_list:
    #    show_img(img_name)
    
    # show_img2('./jpg/i.jpg')

    img_contours()