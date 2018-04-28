# coding=utf-8
# user=hu
from flask import Flask, request
from torchvision import transforms
from werkzeug.utils import secure_filename
import werkzeug.datastructures
from PIL import Image
from torch.autograd import Variable
import web_app
import util.util as util
import cv2 as cv
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

model, data = web_app.init()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './datasets'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

def get_imgtensor(img):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img)
    return img_tensor

# For a giver file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def hello_world():
    return 'hello'


@app.route('/upload', methods=['POST'])
def upload():
    # str_list = []
    upload_file = request.files['image01']
    if upload_file and allowed_file(upload_file.filename):
        filename = secure_filename(upload_file.filename)
        # img 是Image格式的图片格式
        img = Image.open(upload_file.stream).resize((768, 384))
        # img_cv 是opencv的图片格式
        img_cv = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

        img_tensor = get_imgtensor(img)

        _, generated = model(
            Variable(img_tensor.view(1, 3, 384, 768)),
            Variable(data['inst']),
            Variable(data['image']),
            Variable(data['feat']),
            infer=True
        )
        generated_img = util.tensor2im(generated.data[0])
        generated_img_gray = cv.cvtColor(generated_img, cv.COLOR_BGR2GRAY)
        # cv.imshow(generated_img_gray)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # plt.imshow(generated_img)
        # plt.show()
        _, contours, _ = cv.findContours(generated_img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            area = cv.contourArea(cont)
            if area > 800:
                rect = cv.minAreaRect(cont)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(img_cv, [box], 0, (0, 0, 255), 1)
                # print(box)
                # for i in box:
                # crop_img = img.crop((box))
                # crop_img = Image.fromarray(img_cv[box[1][1]: box[0][1], box[1][0]:box[2][0]])
                # str_list.append(pytesseract.image_to_string(crop_img), lang='chi_sim')
        # print(img_cv, generated_img)
        img_and = cv.bitwise_and(img_cv, generated_img)
        img_and_plt = Image.fromarray(img_and).convert('L')
        str_list = pytesseract.image_to_string(img_and_plt, lang='chi_sim')
        plt.imshow(img_cv)
        plt.show()
        print(str_list)
        # img.show()
        # print(upload_file.stream)
        return 'ok' + str(str_list)
    else:
        return 'no'


if __name__ == '__main__':
    app.run()
