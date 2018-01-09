import numpy as np
from scipy.misc import imread, imsave
from py_img_seg_eval.eval_segm import *
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--List_predict', required=True,
                        help="a list for prediction results")
    parser.add_argument('--List_true',
                        default='./data/ADE20K_object150_val.txt')
    parser.add_argument('--val_folder',
                        default='./data/ADEChallengeData2016/annotations/')

    parser.add_argument('--num_class', default=150, type=int)
    args = parser.parse_args()

    list_pred = [x.rstrip() for x in open(args.List_predict, 'r')]
    list_val = [x.rstrip() for x in open(args.List_true, 'r')]
    if (len(list_pred) != len(list_val)):
        raise EvalSegErr("Prediction and validation lists have different number")
    n_assess = len(list_val)

    mean_Accu = AverageMeter()
    InterSect_Area = AverageMeter()
    Union_Area = AverageMeter()

    for i in range(n_assess):
        pred_ = np.load(list_pred[i]) - 1
        val_ = imread(args.val_folder + list_val[i][:-4] + ".png", "I") - 1
        pix_acc, weights_ = Pixel_accuracy(pred_, val_, args.num_class)
        mean_iou_, InterAreaC_, UnionAreaC_ = mean_IU(pred_, val_, args.num_class)
        mean_Accu.update(pix_acc, weights_)
        InterSect_Area.update(InterAreaC_)
        Union_Area.update(UnionAreaC_)
        print("For pic %s Pixel_accuray is %f  Mean_IOU is %f" % (list_val[i], pix_acc, mean_iou_))

    iou_final = InterSect_Area.sum / (Union_Area.sum + 1e-10)
    print("For all the %i pictures"%(n_assess))
    print("Mean Accuracy is %f"%(mean_Accu.avg))
    print("Mean IOU is %f" % (iou_final.mean()))
