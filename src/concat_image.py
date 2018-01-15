import numpy as np
from scipy.misc import imread, imsave
import argparse
from scipy.io import loadmat
from os.path import join


def ConcatenateImg(raw, true, pred, colors):
    '''
    :param raw: Raw Image array RGB color  H x W X 3
    :param true: True label array  H X W
    :param pred: Pred label array H X W
    :param colors: Color map
    :return: Concatenate array  H X 3*W X 3
    '''
    true_color = colorEncode(true, colors)
    pred_color = colorEncode(pred, colors)
    return np.concatenate((raw, true_color, pred_color),
                   axis=1).astype(np.uint8)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--list_predict', default=False)
    parser.add_argument('--image_predict', default=False)
    parser.add_argument('--output_path', default="./")
    args = parser.parse_args()

    if not args.list_predict and not args.image_predict:
        raise ValueError('Your have to provide either a list_predict or image_predict')

    colors = loadmat("data/color150.mat")['colors'] ## Load colormap

    if args.list_predict:
        list_pred = [x.rstrip() for x in open(args.list_predict, 'r')]
        if "val" in list_pred[0]:
            image_folder = "data/ADEChallengeData2016/images/validation/"
            annot_folder = "data/ADEChallengeData2016/annotations/validation/"
            name_length_ = 16
        elif "train" in list_pred[0]:
            image_folder = "data/ADEChallengeData2016/images/validation/"
            annot_folder = "data/ADEChallengeData2016/annotations/validation/"
            name_length_ = 18
        else:
            raise ValueError("Not recognized: image neither training nor validation")

        for predfile in list_pred:
            predc_ = np.load(predfile) - 1
            imgname_ = predfile.split("/")[0][0:name_length_]
            raw_img = imread(join(image_folder, imgname_ + ".jpg"))
            val_ = imread(join(annot_folder, imgname_ +  ".png")).astype("int16") - 1
            img_comb_ = ConcatenateImg(raw_img, val_, predc_, colors)

            outputname_ = predfile.split("/")[0][:-4]
            imsave(join(args.output_path, outputname_ + ".jpg"), img_comb_)
    else:
        list_pred = [args.image_predict]
        if "val" in list_pred[0]:
            image_folder = "data/ADEChallengeData2016/images/validation/"
            annot_folder = "data/ADEChallengeData2016/annotations/validation/"
            name_length_ = 16
        elif "train" in list_pred[0]:
            image_folder = "data/ADEChallengeData2016/images/validation/"
            annot_folder = "data/ADEChallengeData2016/annotations/validation/"
            name_length_ = 18
        else:
            raise ValueError("Not recognized: image neither training nor validation")

        for predfile in list_pred:
            predc_ = np.load(predfile) - 1
            imgname_ = predfile.split("/")[-1][0:name_length_]
            raw_img = imread(join(image_folder, imgname_ + ".jpg"))
            val_ = imread(join(annot_folder, imgname_ +  ".png")).astype("int16") - 1
            img_comb_ = ConcatenateImg(raw_img, val_, predc_, colors)

            outputname_ = predfile.split("/")[-1][:-4]
            imsave(join(args.output_path, outputname_ + ".jpg"), img_comb_)

