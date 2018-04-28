# coding=utf-8
# user=hu
import requests


def images():
    url = 'http://0.0.0.0:8080/'
    file= [
        ('images',('11.png', open('./my_data/my_data_B/TB1.5puLXXXXXXaaXXXunYpLFXX.png', 'rb')))
    ]
    files = {'images': open('./my_data/my_data_B/TB1.5puLXXXXXXaaXXXunYpLFXX.png', 'rb')}
    # r = requests.post(url, files)
    r = requests.post(url, files=file)
    print(r.text)


if __name__ == '__main__':
    images()
