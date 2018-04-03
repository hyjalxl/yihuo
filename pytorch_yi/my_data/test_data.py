import numpy as np
import cv2 as cv
import torch
# img = cv.imread('./my_data_B/TB1.3pkLXXXXXXjaFXXunYpLFXX.png', 0)
# # print(img)
# # tensor = torch.from_numpy(img)
# # print(tensor[: 1] / 255)
# cv.imshow('i', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

def i(path):
    img = cv.imread(path, 0)
    cv.imshow('p', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img

# i('/my_data_B/TB1..FLLXXXXXbCXpXXunYpLFXX.png')
# cv.imshow('p', p)
# cv.waitKey(0)
# cv.destroyAllWindows()
img = cv.imread('./my_data_B/TB1..FLLXXXXXbCXpXXunYpLFXX.png', 0)
cv.imshow('p', img)
cv.waitKey(0)
cv.destroyAllWindows()
