import numpy as np
from scipy.misc import imread
import argparse
from scipy.io import loadmat


def Concatenate_imgs(raw, true, pred, colors):
    '''
    :param raw: Raw Image array RGB color  H x W X 3
    :param true: True label array  H X W
    :param pred: Pred label array H X W
    :param colors: Color map
    :return: Concatenate array  H X 3*W X 3
    '''
    true_color = colorEncode(true, colors)
    pred_color = colorEncode(pred, colors)
    return np.concatenate((raw, lab_color, pred_color),
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
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))
    return labelmap_rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--list_predict', required=True,
                        help="Predicted image has to be a .npy")
    parser.add_argument('--predict_image', required=True,
                        help="Predicted image has to be a .npy")
    args = parser.parse_args()


    colors = loadmat("data/color150.mat")


