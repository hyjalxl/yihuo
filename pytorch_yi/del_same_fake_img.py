# coding=utf-8
# user=hu
import cv2 as cv
from PIL import Image
import numpy as np
import os

ROOT = './fake_img/'


def get_fake_img_name(root):
    name_list = os.listdir(root)
    # print(name_list)
    img_array_dictionary = {}

    # img_array_dictionary.values
    # name_in_dictionary = []
    for name in name_list:
        # img_name = name[:-1]
        # print(ROOT + img_name)
        # img2 = Image.open(root + name)

        img = cv.imread(root + name, 0)
        # print(img)
        if img is not None:
            img = cv.resize(img, (512, 512))
            # img2 = cv.resize(img, (512, 512))
            if img_array_dictionary.get(img.sum()):
                # print('add')
                img_array_dictionary[img.sum()].append(name)
            else:
                # print('new')
                img_array_dictionary[img.sum()] = [name]
    # print(img_array_dictionary)
    for num_ in img_array_dictionary.keys():
        if len(img_array_dictionary[num_]) > 1:
            for show_img_name in img_array_dictionary[num_]:
                # show_img = cv.imread(ROOT + show_img_name)
                # cv.imshow(show_img_name, show_img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                with open('same_img.txt', 'a+') as f:
                    f.write(show_img_name)
                # print(img_array_dictionary[num_])
    # for i in range(len(img_array_dictionary)):
    #     for h in range(i+1, len(img_array_dictionary)):
    #         result = img_array_dictionary[i] - img_array_dictionary[h]
    #         if result.sum() < 100:
    #             print(result.sum())
    h = 1
    # for name in img_array_dictionary:
    #     print(type(name))
    #     h += 1
    #     break
    # print(h)


if __name__ == '__main__':
    get_fake_img_name(ROOT)