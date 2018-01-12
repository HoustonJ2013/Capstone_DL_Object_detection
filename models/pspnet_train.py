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

        list_sample = [x.rstrip() for x in open(args.input_list, 'r')]
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

            ## Read image into batches
            for i_c in range(c_batch_size):
                input_name = list_batch[i_c]
                img = misc.imread(input_name, mode="RGB")
                input_shapes.append((img.shape[0], img.shape[1]))
                img = misc.imresize(img, (473, 473))
                img_batch[i_c, :, :, :] = img
            ## Batch prediction using keras model

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
                np.save(join(output_path, output_name), pred_i)

    def _preprocess_image(self, img):
        """Preprocess an image as input."""
        float_img = img.astype('float16')
        centered_image = float_img - DATA_MEAN
        input_data = centered_image[:, :, :, ::-1]  # RGB => BGR
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


def main(args):
    environ["CUDA_VISIBLE_DEVICES"] = "0"

    sess = tf.Session()
    K.set_session(sess)

    print("     BF Init Model", str(datetime.now()), datetime.now() - TIME_START)

    with sess.as_default():
        print(args)
        # Build Model
        if "pspnet50" in args.model:
            pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              weights=args.model)
        else:
            print("Network architecture not implemented.")

        print("     AF Init Model", str(datetime.now()), datetime.now() - TIME_START)

        ## Batch Prediction
        pspnet.predict(args.input_list,  args.flip, output_path="results/", batch_size=5)

        print("     After Model Prediction", str(datetime.now()), datetime.now() - TIME_START)

        ## Output model report
        if args.print_report:
            with open(args.model + ".txt", 'w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                pspnet.model.summary(print_fn=lambda x: fh.write(x + '\n'))


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
    parser.add_argument('--num_gpus', default="1")
    parser.add_argument('-f', '--flip', action='store_true',
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('--print_report', default=False)

    args = parser.parse_args()

    main(args)
