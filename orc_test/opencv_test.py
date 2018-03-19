# -*- coding: utf-8 -*-

import cv2
import os


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
    
    
if __name__ == '__main__':
    #file_name_list = get_file_name('./jpg')
    #for img_name in file_name_list:
    #    show_img(img_name)
    
    show_img2('./jpg/i.jpg')