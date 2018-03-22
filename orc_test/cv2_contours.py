# coding=utf-8
# user=hu
import cv2
import numpy as np
from PIL import Image
import pytesseract
import time


def contours_t():
    im = cv2.imread('./jpg/j.jpg', 1)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 1)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[4]
    cv2.drawContours(im, contours, -1, (0, 255, 0), 3)

    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def threshold_t():
    im = cv2.imread('./jpg/i.jpg', 1)

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    for i in range(1, 255, 40):
        print(i)
        ret, thresh = cv2.threshold(imgray, i, 255, 0)
        # cv2.imshow(str(i), thresh)
        cv2.imwrite('1.png', thresh)
        # time.sleep(0.5)
        im_pil = Image.open('1.png')
        text = pytesseract.image_to_string(im_pil, lang='chi_sim')
        print(text)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def cv2pil():
    img = cv2.imread('./jpg/i.jpg', 1)
    img[1:1, 20:20]
    cv2.imshow('i', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # img_pil.show()

if __name__ == '__main__':
    # contours_t()
    # threshold_t()
    cv2pil()
