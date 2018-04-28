# coding=utf-8
# user=hu

import requests
import time
import os

test_img_path = '/home/yihuo/Desktop/yihuo_git/pytorch_yi/scan_img/'


def p(name):

    files = {'image01': open(name, 'rb')}
    user_info = {'name': 'huyangjie'}

    r = requests.post('http://127.0.0.1:5000/upload', data=user_info, files=files)

    print(name, '\n', r.text)


if __name__ == '__main__':
    # root = './test_img/'
    for name in os.listdir(test_img_path):
        p(os.path.join(test_img_path, name))
        # time.sleep(8)