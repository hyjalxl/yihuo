# -*- coding: utf-8 -*-

from PIL import Image
import pytesseract
import cv2
import os

def img2text(file_name):
    # im = cv2.imread(file_name)
    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('im2', im)
    # cv2.waitKey(0)
    
    text = pytesseract.image_to_string(Image.open('./jpg/' + file_name), lang='chi_sim')
    
    print(text)

def get_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        
        return files

if __name__ == '__main__':
    print('hello')
    file_name_list = get_file_name('./jpg')
    
    for file_name in file_name_list:
        s = '.jpg'
        if s in file_name:
            print(file_name)
            img2text(file_name)
            print('>' * 50)
