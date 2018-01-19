#!/usr/bin/env python
"""
This module is a Keras/Tensorflow based implementation of Pyramid Scene Parsing Networks.

The work is adapted from the keras/TF implementation https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow.
Original paper & code published by Hengshuang Zhao et al. (2017)


I added a list of changes and add more functions needed for my Galvanize Capstone project
1. Robust and faster model weight loading
2. Faster model prediction
3. Support batch prediction
4. Add train function


Developer's note.
1. The model weights from .npy to keras .json and .h5 is not consistent accross platforms, we provide h5 and json model
2. Attempted to change the code for variable input image size, but tensorflow doesn't support dynamic ksize avg_pool
   layer. Current implementation is for fixed size input (473 x 473). Multi-scale prediction doesn't bring benefit in
   this case.
3. Simplified model loading process, and make it more robust to different platform
4. Train function: Continue training on top of a model or train from scratch
Contact: jingbo.liu2013@gmail.com
"""

from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile, exists
from os import environ
import os
from math import ceil
import argparse
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


# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[[123.68, 116.779, 103.939]]]])  # RGB order
TIME_START = datetime.now()

def main(args):
    environ["CUDA_VISIBLE_DEVICES"] = "0"

    sess = tf.Session()
    K.set_session(sess)

    print("     BF Init Model", str(datetime.now()), datetime.now() - TIME_START)

    with sess.as_default():
        print(args)
        # Build Model
        if "psp" in args.model and "50" in args.model:
            pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              weights=args.weights)
        else:
            print("Network architecture not implemented.")

        print("     AF Init Model", str(datetime.now()), datetime.now() - TIME_START)

        ## Batch Prediction
        pspnet.predict(args.input_list,  args.flip, output_path="results/", batch_size=5)

        print("     After Model Prediction", str(datetime.now()), datetime.now() - TIME_START)

        ## Output model report
        # if args.print_report:
        #     print_model(pspnet, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012',
                                 'test'])
    parser.add_argument('-il', '--input_list', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')

    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-o', '--output_path', type=str, default='results/',
                        help='Path to output')
    parser.add_argument('-w', '--weights', type=str, default='results/',
                                                help='name of the weights to load')
    parser.add_argument('--num_gpus', default="1")
    parser.add_argument('-f', '--flip', action='store_true',
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('--print_report', default=False)

    args = parser.parse_args()

    main(args)
