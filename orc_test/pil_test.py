# -*- coding: utf-8 -*-

from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np

def img_show(file_name):
    img = Image.open(file_name)
    
    
    plt.figure(num = 'astronaut', figsize=(20,20))
    
    for i in range(1, 10, 10):
        plt.subplot(2,5,i)
        print(i)
        box = (i*10, i*10, i*10+200, i*10+200)
        
        plt.imshow(img.crop(box))
        
        
def cv2_show(file_name):
    img = Image.open(file_name)
    img_np = np.array(img.convert('L'))
    img3 = img[100:100, 200:200]
    cv2.imshow('img', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('main is run')
    file_name = './jpg/i.jpg'
    img_show(file_name)
    
