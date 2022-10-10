import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.measure import label


def save_plots_logs(path, loss, train, test):
    n = len(loss)
    logs = np.zeros((n + 1, 4))
    logs[:n, 0] = list(range(1, n + 1))
    logs[:n, 1] = loss
    logs[-1, 1] = max(loss)
    logs[:n, 2] = train
    logs[-1, 2] = max(train)
    logs[:n, 3] = test
    logs[-1, 3] = max(test)
    formatlist = "%i %.4f %.4f %.4f"
    np.savetxt(os.path.join(path, "logs.txt"), logs, fmt=formatlist)


def dice_est(O, L):
    L = L > 0
    O = O > 0.5  # threshold
    e = 0.001
    return (np.sum(O[L == 1]) * 2 + e) / (np.sum(O) + np.sum(L) + e)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, O, L):
        EPS = 0.00001
        return torch.log(
            1 / ((torch.sum(O[L == 1]) * 2 + EPS) / (torch.sum(O) + torch.sum(L) + EPS))
        )


class WBCELoss(nn.Module):
    # Weighted Binary cross entropy loss
    def __init__(self, weights=0.5, Pweights=1):
        super(WBCELoss, self).__init__()
        self.weights = weights  # weights to decide whether to penalize false positives or false negatives more
        self.Pweights = Pweights  # weights to penalize which class more. In theory penalizing minority class more
        # will help to emphasize minority class as much as majority class thus minority
        #  class will not be treated as noise.

    def forward(self, x, y):
        s = 0.0000001
        assert x.shape == y.shape, "inconsistent size between target and output"
        loss = self.weights * (y * torch.log(x + s)) + (1 - self.weights) * (
            (1 - y) * torch.log(1 - x + s)
        )
        return torch.neg(torch.mean(self.Pweights * loss))


def dice_loss2(input, target):
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def fix_dimensions(A):
    if A.dim() == 2:
        x1, x2 = A.size()
        A = A.view(1, 1, x1, x2)
    elif A.dim() == 3:
        x0, x1, x2 = A.size()
        A = A.view(1, 1, x0, x1, x2)
    return A


def getLargestCC(segmentation):
    labels = label(segmentation)
    idx = np.argsort(np.bincount(labels.flat))[-2:]
    if sum(segmentation[labels == idx[1]]) != 0:
        largestCC = labels == idx[1]
    else:
        largestCC = labels == idx[0]  # or idx[0])
    return largestCC


def randomDropSmallCC(segmentation, rate=0.2):
    label_img, cc_num = ndimage.label(segmentation)
    # CC = ndimage.find_objects(label_img)
    cc_areas = ndimage.sum(segmentation, label_img, range(cc_num + 1))
    armax = np.argmax(cc_areas)
    area_mask = cc_areas > 10
    area_mask = [True if x and random.random() < rate else False for x in area_mask]
    # make sure to not to drop large ones
    area_mask_large = cc_areas > max(np.max(cc_areas) * 0.40, np.median(cc_areas))
    area_mask = (area_mask + area_mask_large) > 0
    area_mask = np.array(area_mask)
    label_img[~area_mask[label_img]] = 0
    label_img[area_mask[label_img]] = 1
    return label_img


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
