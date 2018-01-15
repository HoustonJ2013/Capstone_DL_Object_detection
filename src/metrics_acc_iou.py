import numpy as np
from scipy.misc import imread
import argparse


def Pixel_accuracy(y_pred, y_true, num_class=150):
    '''
    :param y_pred: predicted label image
    :param y_true: true label image
    :return: Accuracy for this prediction
    '''
    classMask_ = np.logical_and(y_true >= 0, y_true < num_class)
    return np.sum(y_true[classMask_] == y_pred[classMask_]) / \
           (np.sum(classMask_) + 1e-8), np.sum(classMask_)


def mean_IU(eval_segm, gt_segm, num_class=150):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    # cl, n_cl = union_classes(eval_segm, gt_segm)
    # _, n_cl_gt = extract_classes(gt_segm)

## to get consistent IOU with MIT benchmark model
    eval_segm[gt_segm < 0] = -1


    cl, n_cl = union_classes(eval_segm, gt_segm)
    cl_gt, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    IU = list([0]) * n_cl
    InterSect = np.zeros((num_class, 1))
    Union = np.zeros((num_class, 1))

    cl_mask = cl[cl < num_class]

    for i, c in enumerate(cl_mask):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        # if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
        #     continue
        ## To make it consistent with MIT benchmark
        if (np.sum(curr_eval_mask) == 0) and (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        InterSect[int(c)] = n_ii
        Union[int(c)] = t_i + n_ij - n_ii
        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_, InterSect, Union


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    cl = cl[cl >=0]
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


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
                        default='./data/ADEChallengeData2016/annotations/validation/')

    parser.add_argument('--num_class', default=150, type=int)
    args = parser.parse_args()

    list_pred = [x.rstrip() for x in open(args.List_predict, 'r')]
#    list_val = [x.rstrip() for x in open(args.List_true, 'r')]
#    if (len(list_pred) != len(list_val)):
#        raise EvalSegErr("Prediction and validation lists have different number")
    n_assess = len(list_pred)

    mean_Accu = AverageMeter()
    InterSect_Area = AverageMeter()
    Union_Area = AverageMeter()

    for i in range(n_assess):
        pred_ = np.load(list_pred[i]) - 1
        list_val = list_pred[i].split("/")[-1]
        val_ = imread(args.val_folder + list_val[:-4] + ".png", "I") - 1
        ## debug control
        # if i == 0:
        #     np.save("debug_pred_0", pred_)
        #     np.save("debug_val_0", val_)
        pix_acc, weights_ = Pixel_accuracy(pred_, val_, args.num_class)
        mean_iou_, InterAreaC_, UnionAreaC_ = mean_IU(pred_, val_, args.num_class)
        mean_Accu.update(pix_acc, weights_)
        ## debug control
        # np.save("local_intersec" + str(i), InterAreaC_)
        # np.save("local_union" + str(i), UnionAreaC_)

        InterSect_Area.update(InterAreaC_)
        Union_Area.update(UnionAreaC_)
        print("For pic %s Pixel_accuray is %f  Mean_IOU is %f" % (list_val[i], pix_acc, mean_iou_))

    iou_final = InterSect_Area.sum / (Union_Area.sum + 1e-10)
    # np.save("iou_final.npy", iou_final)
    print("For all the %i pictures"%(n_assess))
    print("Mean Accuracy is %f"%(mean_Accu.avg))
    print("Mean IOU is %f" % (iou_final.mean()))
