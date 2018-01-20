from flask import Flask
from flask import render_template, request
import numpy as np
import pandas as pd
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import layers_builder as layers
import model_utils as model_utils
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from datetime import datetime
from pspnet import *
from scipy.io import loadmat
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from collections import Counter
import time

app = Flask(__name__)
DATA_MEAN = np.array([[[[123.68, 116.779, 103.939]]]])  # RGB order
TIME_START = datetime.now()
colors = loadmat("data/color150.mat")['colors'] ## Load colormap
obj_df = pd.read_csv("data/object150_info.csv")


#helper functions
def colorlabel(color_list):
    obj_list = obj_df[obj_df["Idx"].isin(color_list+1)]["Name"].values
    n_obj = len(obj_list)
    img_list = ["color150/" + obj.split(";")[0] + ".jpg" for obj in obj_list]
    images = list(map(Image.open, img_list))
    widths, heights = zip(*(i.size for i in images))
    height_max = max(heights)
    width_total = sum(widths)
    new_im = Image.new('RGB', (width_total, height_max))
    x_offset = 0
    print(len(images))
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def colorEncode(labelmap, colors):
    '''
    Encode label map with predefined color
    :param labelmap: label array
    :param colors:  Colors
    :return:  Colored RGB Image
    '''
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))
    return labelmap_rgb



@app.route('/')
def index():
    return render_template('index.html',  data=None)


@app.route('/run', methods=['POST'])
def run():
    flip = True
    picture = request.form["answer"]
    url = request.form["url"]
    print(url)
    print(picture)
    input_list = ["static/" + picture]
    sess = tf.Session()
    K.set_session(sess)
    pic_pred = []
    pic_pred.append("/" + input_list[0])
    with sess.as_default():
        print("     AF Init Model", str(datetime.now()), datetime.now() - TIME_START)
        Capnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                          weights="pspnet50_ade20k")
        print("     AF Init Model", str(datetime.now()), datetime.now() - TIME_START)
        Capnet.predict(input_list, flip, output_path="static/", batch_size=5)
        pred_path = (input_list[0])[:-4] + ".npy"
        pred_array = np.load(pred_path).astype("float16") - 1
        pred_rgb = colorEncode(pred_array, colors)
        img = Image.fromarray(pred_rgb)
        img.save("static/" + picture[:-4] + "_pred.jpg")
        pic_pred.append("/static/" + picture[:-4] + "_pred.jpg")
        print("finished pred")

        prob_path = (input_list[0])[:-4] + "_prob.npy"
        pred_prob = np.load(prob_path)
        plt.imshow(pred_prob, cmap="gray")
        plt.axis('off')
        plt.savefig("static/" + picture[:-4] + "_pred_prob.jpg", bbox_inches='tight')
        plt.colorbar()
        plt.clf()
        pic_pred.append("/static/" + picture[:-4] + "_pred_prob.jpg")
        print("finished prob")

        pred_array = pred_array.flatten()
        pred_array = pred_array[pred_array > 0]
        color_list = np.array([tem[0] for tem in Counter(pred_array).most_common(10)])
        new_im = colorlabel(color_list)
        new_im.save("static/color.jpg")
        print("finished color")
    pic_pred.append("/static/color.jpg")
    return render_template('index.html',  data=pic_pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
