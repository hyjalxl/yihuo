# coding=utf-8
# user=hu

import web
from PIL import Image
import io


import numpy as np

urls = (
    '/', 'index'
)


class index:

    def GET(self):
        return "Hello, world!"

    def POST(self):
        # data = web.data()
        # print(data)
        i = web.input()
        print(i)
        # h = i.items()
        # print(h)
        png = i.get('images')
        # with open('11.png', 'wb') as f:
        #     f.write(png)
        print(png)
        image = Image.open(io.BytesIO(png))
        image.show()
        # img = Image.frombytes(mode='L', size=(32, 32), data=png.encode('hex'))
        # img = Image.fromstring(png)
        # img_array = np.fromfile(png, dtype=np.ubyte)
        # print(img_array)
        # name = i.get('name')
        return 'You name is:'


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
