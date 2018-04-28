# coding=utf-8
# user=hu

import requests
from urllib import request
import time
from PIL import Image

with open('fake_img_url.txt', 'r') as f:
    url_list = f.readlines()
#
# for url in url_list:
#     if 'http:' in url:
#         if '.jpg' not in url and '.png' not in url:
#             print(url.split('/')[-1] + '.png')
#         else:
#             print(url.split('/')[-1])
#     else:
#         if '.jpg' not in url and '.png' not in url:
#             print('http:' + url.split('/')[-1] + '.png')
#         else:
#             print('http:' + url.split('/')[-1])


# print(url_list)
i = 1
for url in url_list:
    print(i)
    try:
        if 'http:' in url:
            request.urlretrieve(url, './fake_img/' + url.split('/')[-1])
            print(url.split('/')[-1])
        else:
            url = 'http:' + url
            request.urlretrieve(url, './fake_img/' + url.split('/')[-1])
            print(url.split('/')[-1])
        # img = requests.get('http:' + url)
        # img = Image.open(img.content)
        # print(img.content)
        # time.sleep(1)
        i += 1
    except:
        print('wrong')

if __name__ == '__main__':
    pass