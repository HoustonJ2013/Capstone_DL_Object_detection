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
    img_list = np.array([x.rstrip() for x in open(args.train_list, 'r')])
    label_list = np.array([x.rstrip() for x in open(args.label_list, 'r')])
    n_total = len(img_list)
    batch_size = args.batch_size
    index_array = np.arange(n_total)
    n_batch = int((n_total - 1) / batch_size) + 1
    print("No. %i of batches for each epoch"%(n_batch))
    with sess.as_default():
        # Build Model and train from scratch or train on top of an existing model
        # Current implementation needs to provide pre-trained weights
        if args.weights is not None:
            if "pspnet50" in args.model:
                pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                                  weights=args.weights)
            else:
                print("Network architecture not implemented.")

            if args.optimizer =="SGD":
                optimizer_ = SGD(lr=args.learning_rate, momentum=0.95, nesterov=True)
            elif args.optimizer=="Adam":
                optimizer_ = Adam(lr=args.learning_rate, beta_1=0.95, beta_2=0.999, decay=0.0)
            else:
                print("We support SGD and Adam optimizers")
            pspnet.model.compile(optimizer=optimizer_,
                  loss='categorical_crossentropy', # Categorical_crossentropy is the same and NLL
                  metrics=[pspnet._img_pixel_accuracy]) #

            print("The GPU memory has to be more than %f Gb"%(
                        get_model_memory_usage(batch_size, pspnet.model)))
            for i in range(args.num_epoch):
                np.random.shuffle(index_array) ## Shuffle index
                for i_batch in range(n_batch):
                    if (i_batch + 1) * batch_size < n_total:
                        index_batch = index_array[i_batch * batch_size:(i_batch + 1) * batch_size]
                    else:
                        index_batch = index_array[i_batch * batch_size:]
                    input_batch = img_list[index_batch]
                    lab_batch = label_list[index_batch]
                    lss_, acc_ = pspnet.train_one_epoch(input_batch, lab_batch)
                    print("%i/%i in %i epoch finished at time %s from start, Loss is %f Acc is %f" %
                          (i_batch, n_batch, i, str(datetime.now() - TIME_START), lss_, acc_))
                if (i > 0 and i%args.save_epoch==0) or i == args.num_epoch - 1:
                    savefolder_ = args.ckpt + args.model + "_batch" + str(args.batch_size) \
                                  + "_lr" + str(round(args.learning_rate, 5)) + \
                                  "_" + args.optimizer + "/"
                    if exists(savefolder_):
                        pspnet.save_model(savefolder_, args.modelname + "_epoch"+str(i))
                    else:
                        os.mkdir(savefolder_)
                        pspnet.save_model(savefolder_, args.modelname + "_epoch" + str(i))

        ## Output model report
        if args.print_report:
            print_model(pspnet, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('--modelname', type=str, default='pspade20k',
                        help="A name to label the model weights when saved")
    parser.add_argument('-train', '--train_list', type=str,
                        default='data/ADE20K_object150_train.txt',
                        help='Path the input image')
    parser.add_argument('-lab', '--label_list', type=str, default='',
                        help='Path validation image')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epoch', type=int, default=1,
                        help='no. of epoches to train the model')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--optimizer', default="Adam")

    parser.add_argument('--ckpt', default="weights/")
    parser.add_argument('--weights', default=None,
                        help="If weights provided, training start from this weights")
    parser.add_argument('--save_epoch', type=int, default=1,
                        help="save the weights every save_epoch iteration")

    parser.add_argument('-f', '--flip', action='store_true',
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('--print_report', default=False)
    args = parser.parse_args()
    print(args)
    main(args)
