#!/usr/bin/env python
"""
This module is a Keras/Tensorflow based implementation of Pyramid Scene Parsing Networks.

The work is adapted from the keras/TF implementation https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow.
Original paper & code published by Hengshuang Zhao et al. (2017)


I added a list of changes and add more functions needed for my Galvanize Capstone project
1. The model weights from .npy to keras .json and .h5 is not consistent accross platforms, we provide h5 and json model
2. Attempted to change the code for variable input image size, but tensorflow doesn't support dynamic ksize avg_pool
   layer
2. Train function: Continue training on top of a model or train from scratch
2. Predict function
3. Load more models
Contact: jingbo.liu2013@gmail.com
"""

from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile
from os import environ
from math import ceil
import argparse
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import layers_builder as layers
import model_utils as model_utils
import matplotlib.pyplot as plt
from datetime import datetime


# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order
EVALUATION_SCALES = [1.0]  # must be all floats!


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017."""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        """Instanciate a PSPNet."""
        self.input_shape = input_shape

        json_path = join("weights", "keras", weights + ".json")
        h5_path = join("weights", "keras", weights + ".h5")

        if isfile(json_path) and isfile(h5_path):
            print("Keras model & weights found, loading...")
            with open(json_path, 'r') as file_handle:
                self.model = model_from_json(file_handle.read())
            self.model.load_weights(h5_path)
        else:
            self.model = layers.build_pspnet(nb_classes=nb_classes,
                                             resnet_layers=resnet_layers,
                                             input_shape=self.input_shape)
            self.model.load_weights(h5_path)

    def predict(self, img, flip_evaluation, multi_scale=[1]):
        """
        Predict segementation for a batch of images

        Arguments:
            img: must be batch_size, rowsxcolsx3
        """
        h_ori, w_ori = img.shape[1:3]

        print("     BF Raw Image Resize", str(datetime.now()), datetime.now() - TIME_START)
        if img.shape[1:3] != self.input_shape:
            print("Input %s not fitting for network size %s, resizing"
                  % (img.shape[1:3], self.input_shape))
            img = misc.imresize(img, self.input_shape)
        print("     AF Raw Image Resize", str(datetime.now()), datetime.now() - TIME_START)


        print("     BF Preprocess Image", str(datetime.now()), datetime.now() - TIME_START)
        input_data = self._preprocess_image(img)
        print("     AF Preprocess Image", str(datetime.now()), datetime.now() - TIME_START)
        # utils.debug(self.model, input_data)

        print("     BF Regular Model Prediction", str(datetime.now()), datetime.now() - TIME_START)
        regular_prediction = self.model.predict(input_data)
        print("     AF Regular Model Prediction", str(datetime.now()), datetime.now() - TIME_START)

        if False: #flip_evaluation:
            print("Predict flipped")
            flipped_prediction = np.fliplr(self.model.predict(np.flip(input_data, axis=2)))
            prediction = (regular_prediction + flipped_prediction) / 2.0
        else:
            prediction = regular_prediction

        print("     BF Zoom", str(datetime.now()), datetime.now() - TIME_START)

        prediction = np.argmax(prediction, axis=3)

        if img.shape[0:1] != self.input_shape:  # upscale prediction if necessary
            h, w = prediction.shape[:2]
            prediction = ndimage.zoom(prediction, (1, 1.*h_ori/h, 1.*w_ori/w),
                                      order=1, prefilter=False)
        print("     AF Zoom", str(datetime.now()), datetime.now() - TIME_START)

        return prediction

    def _preprocess_image(self, img):
        """Preprocess an image as input."""
        float_img = img.astype('float16')
        centered_image = float_img - DATA_MEAN
        bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
        input_data = bgr_image[:, :, :, :]  # Append sample dimension for keras
        return input_data



class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        """Instanciate a PSPNet50."""
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        """Instanciate a PSPNet101."""
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[0]
    cols_missing = target_size[1] - img.shape[1]
    padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    return padded_img


def visualize_prediction(prediction):
    """Visualize prediction."""
    cm = np.argmax(prediction, axis=2) + 1
    color_cm = model_utils.add_color(cm)
    plt.imshow(color_cm)
    plt.show()

TIME_START = datetime.now()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-il', '--input_list', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')

    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-o', '--output_path', type=str, default='results/',
                        help='Path to output')
    parser.add_argument('--id', default="0")
    parser.add_argument('-f', '--flip', action='store_true',
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('-ms', '--multi_scale', action='store_true',
                        help="Whether the network should predict on multiple scales.")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    print("     BF Init Model", str(datetime.now()), datetime.now() - TIME_START)

    with sess.as_default():
        print(args)

        if "pspnet50" in args.model:
            pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              weights=args.model)
        elif "pspnet101" in args.model:
            if "cityscapes" in args.model:
                pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                                   weights=args.model)
            if "voc2012" in args.model:
                pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                   weights=args.model)
        else:
            print("Network architecture not implemented.")

        print("     AF Init Model", str(datetime.now()), datetime.now() - TIME_START)

        if args.multi_scale:
            EVALUATION_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # must be all floats!

        list_sample = [x.rstrip() for x in open(args.input_list, 'r')]
        
        for input_name in list_sample:
            img = misc.imread(input_name, mode="RGB")
            class_image = pspnet.predict(img, args.flip)
            print("     AF Predict Model", str(datetime.now()), datetime.now() - TIME_START)
            print("Predicting for ", input_name)
            output_name = input_name.split("/")[-1].replace(".jpg", "").replace(".png", "")
            np.save(join(args.output_path,output_name), class_image.astype("int16") + 1)

        # with open('pspnet50_report.txt', 'w') as fh:
        #     # Pass the file handle in as a lambda function to make it callable
        #     pspnet.model.summary(print_fn=lambda x: fh.write(x + '\n'))

