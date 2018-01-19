from flask import Flask
from flask import render_template, request
import numpy as np
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



app = Flask(__name__)
DATA_MEAN = np.array([[[[123.68, 116.779, 103.939]]]])  # RGB order
TIME_START = datetime.now()


@app.route('/')
def index():
    return render_template('index.html',  data=None)


@app.route('/run', methods=['POST'])
def run():
    option = request.form["Prediction Options"]
    flip = False
    input_list = ["static/ADE_val_00001772.jpg"]

    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        print("     AF Init Model", str(datetime.now()), datetime.now() - TIME_START)
        Capnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                          weights="pspnet50_ade20k")
        print("     AF Init Model", str(datetime.now()), datetime.now() - TIME_START)
        Capnet.predict(input_list, flip, output_path="/static/", batch_size=5)

    pic_pred = ["/static/validation_ADE_val_00000661.png", "/static/validation_ADE_val_00000661.png"]
    return render_template('index.html',  data=pic_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
