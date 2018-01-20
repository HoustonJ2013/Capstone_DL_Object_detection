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


# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[[123.68, 116.779, 103.939]]]])  # RGB order
TIME_START = datetime.now()

class PSPNet(object):
    """Pyramid Scene Parsing Network by proposed by Hengshuang Zhao et al 2017."""

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

    def predict(self, input_list, flip_evaluation, output_path = "./", batch_size = 5):
        """
        Predict segementation for a batch of images

        Arguments:
            img: A list of input images
        """

        list_sample = input_list
        n_total = len(list_sample)
        if n_total == 1:
            batch_size =1
        n_batch = int((n_total - 1) /batch_size) + 1

        for i_batch in range(n_batch):
            if (i_batch + 1) * batch_size < n_total:
                list_batch = list_sample[i_batch * batch_size:(i_batch + 1) * batch_size ]
            else:
                list_batch = list_sample[i_batch * batch_size:]
            c_batch_size = len(list_batch)
            img_batch = np.zeros((c_batch_size, 473, 473, 3))
            input_shapes = []

            # Read image into batches
            for i_c in range(c_batch_size):
                input_name = list_batch[i_c]
                img = misc.imread(input_name, mode="RGB")
                input_shapes.append((img.shape[0], img.shape[1]))
                img = misc.imresize(img, (473, 473))
                img_batch[i_c, :, :, :] = img
            # Batch prediction using keras model

            input_data = self._preprocess_image(img_batch)
            regular_prediction = self.model.predict(input_data, batch_size=batch_size)

            if flip_evaluation:
                print("Predict flipped")
                flipped_prediction = np.flip(
                                            self.model.predict(
                                                np.flip(input_data, axis=2),
                                            batch_size=batch_size),
                                            axis=2)
                prediction = (regular_prediction + flipped_prediction) / 2.0
            else:
                prediction = regular_prediction


            prediction = np.argmax(prediction, axis=3) + 1

            ## Post-scale up and save image
            for i_c in range(c_batch_size):
                h_ori, w_ori = input_shapes[i_c][0], input_shapes[i_c][1]
                h, w = prediction.shape[1:3]
                pred_i = ndimage.zoom(prediction[i_c, :, :], (1.*h_ori/h, 1.*w_ori/w),
                                          order=1, prefilter=False)
                input_name = list_batch[i_c]
                output_name = input_name.split("/")[-1][0:-4]
                print(output_name)
                np.save(join(output_path, output_name), pred_i)

    def train_one_epoch(self, input_list, label_list, n_class=150):
        '''
        train one epoch on provide input and validation images
        :param input_list: list of input images
        :param label_list: list of validation images
        :param flip_evaluation: flip preprocessing or not
        :param batch_size: batch_size
        :return: loss
        '''

        # X_batch = self._get_X_batch(input_list, ndim=3)
        # y_batch = self._get_y_batch(label_list, n_class=n_class)
        X_batch, y_batch = self._get_X_y_batch(input_list, label_list)
        ## Random flip left right
        if np.random.choice([False, True]):
            X_batch = np.flip(X_batch, axis=2)
            y_batch = np.flip(y_batch, axis=2)
        return self.model.train_on_batch(X_batch, y_batch)

    def _get_X_y_batch(self, img_list, label_list, shapes = (473, 473), ndim=3, n_class=150):
        '''
        :param img_list: a list of rgb or gray image
        :param shapes: the size of batch image
        :return: img_batch n_img x height x width x nchanel
        '''
        batch_size = len(img_list)
        X_batch = np.zeros((batch_size, shapes[0], shapes[1], ndim))
        y_batch = np.zeros((batch_size, shapes[0], shapes[1], n_class))
        ## Read image into batches
        for i_c in range(batch_size):
            input_name = img_list[i_c]
            if ndim == 3:
                img = misc.imread(input_name, mode="RGB")
            elif ndim == 2:
                img = misc.imread(input_name)
            val_name = label_list[i_c]
            val = misc.imread(val_name)
            assert img.shape[0] == val.shape[0]
            assert img.shape[1] == val.shape[1]
            # Resize X
            img = misc.imresize(img, (shapes[0], shapes[1]))
            X_batch[i_c, :, :, :] = img
            X_batch = self._preprocess_image(X_batch)
            # Resize and expand y
            cl = np.unique(val)
            cl = cl[cl > 0]
            val = misc.imresize(val, (shapes[0], shapes[1]))
            for clc in cl: # clc - 1 is the right index for clc
                cmask = (val == clc).astype(int)
                y_batch[i_c, :, :, clc - 1] = cmask
        return X_batch.astype("float16"), y_batch.astype("float16")

    def _preprocess_image(self, imgbatch):
        """Preprocess an image as input."""
        float_img = imgbatch.astype('float16')
        centered_image = float_img - DATA_MEAN
        input_data = centered_image[:, :, :, ::-1]  # RGB => BGR
        return input_data

    def save_model(self, ckpt="./", modelname="test"):
        model_json = self.model.to_json()
        with open(join(ckpt, modelname + ".json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(join(ckpt, modelname + ".h5"))
        print("Saved model to disk")

    def load_model(self, ckpt="./", modelname="test"):
        json_file = open(join(ckpt, modelname + '.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(join(ckpt, modelname + ".h5"))
        print("Loaded model from disk")

    def _img_pixel_accuracy(self, y_pred, y_true, num_class=150):
        '''
        :param y_pred: predicted label image
        :param y_true: true label image
        :return: Accuracy for this prediction
        '''
        y_pred_img_ = tf.argmax(y_pred, axis=3)
        y_true_img_ = tf.argmax(y_true, axis=3)
        classMask_ = tf.logical_and(tf.greater_equal(y_true_img_, 0),
                                    tf.less(y_true_img_, num_class))
        y_true_masked_ = tf.boolean_mask(y_true_img_, classMask_)
        y_pred_masked_ = tf.boolean_mask(y_pred_img_, classMask_)

        return tf.reduce_sum(tf.cast(tf.equal(y_true_masked_, y_pred_masked_), tf.int32))/\
               tf.reduce_sum(tf.cast(classMask_, tf.int32))


# helper functions
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


def print_model(net, args):
    with open(args.model + ".txt", 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        net.model.summary(print_fn=lambda x: fh.write(x + '\n'))


# For memory control
def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

