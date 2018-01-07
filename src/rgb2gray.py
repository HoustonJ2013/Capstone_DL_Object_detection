import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import toimage,imsave, imread
from os import listdir
import argparse


### Gray color conversion
def standard_gamma_corrected(imgarray):
    if np.max(imgarray) > 200: ## If it is a RGB with channel maximum 256
        imgarray = imgarray/255.0
    imgarray[:, :, 0] = imgarray[:, :, 0] ** (1 / 2.2)
    imgarray[:, :, 1] = imgarray[:, :, 1] ** (1 / 2.2)
    imgarray[:, :, 2] = imgarray[:, :, 2] ** (1 / 2.2)
    return imgarray


def gleam_rgb2gray(imgpath, imgname, output_folder = "./"):
    imgarray = imread(imgpath)
    if np.ndim(imgarray) == 3:
        gc = standard_gamma_corrected(imgarray)
        gray = 1/3. * (gc[:, :, 0] + gc[:, :, 1] + gc[:, :, 2])
        imsave(output_folder + imgname[0:-4] + ".png", gray)
    else:
        imsave(output_folder + imgname[0:-4] + ".png", imgarray)
    pass


def luminance_rgb2gray(imgpath, imgname, output_folder = "./"):
    imgarray = imread(imgpath)
    if np.ndim(imgarray) == 3:
        gray = imgarray[:, :, 0] * 0.3 + imgarray[:, :, 1] * 0.59 + imgarray[:, :, 2] * 0.11
        imsave(output_folder + imgname[0:-4] + ".png", gray)
    else:
        imsave(output_folder + imgname[0:-4] + ".png", imgarray)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--input_folder',
                        default='data/ADEChallengeData2016/images/training/',
                        help="The folder has RGB pictures")
    parser.add_argument('--output_folder',
                        default='data/ADEChallengeData2016/images/training/',
                        help="The folder to put PNG pictures")
    parser.add_argument('--method', default='luminance',
                        help="The method to convert RGB to gray scale")
    args = parser.parse_args()
    rgb_pics = [tem for tem in listdir(args.input_folder) if ".jpg" in tem]
    n_pics = len(rgb_pics)
    for i in range(n_pics):
        imgpath = args.input_folder + rgb_pics[i]
        imgname = rgb_pics[i]
        if i%500 == 0:
            print(str(i) + " images converted")
        if args.method == "luminance":
            luminance_rgb2gray(imgpath, imgname, args.output_folder)
        else:
            gleam_rgb2gray(imgpath, imgname, args.output_folder)
