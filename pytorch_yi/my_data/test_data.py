import numpy as np
import cv2 as cv
import torch
img = cv.imread('./my_data_B/TB1.3pkLXXXXXXjaFXXunYpLFXX.png', 0)
# print(img)
tensor = torch.from_numpy(img)
print(tensor[: 1] / 255)
