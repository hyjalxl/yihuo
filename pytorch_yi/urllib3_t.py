# coding=utf-8
# user=hu
######################
#urlib3 的post方法上传的是字符串类型，web.py获取的数据流转换成图片有点困难
#
######################

import urllib3
from PIL import Image

http = urllib3.PoolManager()
with open('./my_data/my_data_B/TB1..FLLXXXXXbCXpXXunYpLFXX.png', 'rb') as f:
    img = f.read()
with open('./my_data/my_data_B/TB1.5puLXXXXXXaaXXXunYpLFXX.png', 'rb') as f:
    img2 = f.read()
print(img)
print(img2)
# img2 = Image.open('./my_data/my_data_B/TB1.3pkLXXXXXXjaFXXunYpLFXX.png')
# img2.show()

r = http.request(
    'POST',
    'http://0.0.0.0:8080/',
    fields={
        'images': img
    }
    # body=img,
    # headers={'Content-Type': 'image/png'},
)
print(r.data)

if __name__ == '__main__':
    pass