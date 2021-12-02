# @Author: Zhiyong Yang

import numpy as np
from itertools import permutations
from sklearn.metrics import roc_auc_score
import os
import math


def aucMP(y_true, y_pred, num_class=None):
    '''
        A fast implementation (`O(N_C N logN )` instead of `O(N_CN^2)`) of an unbiased estimator of the AUC metric for multipartite ranking:
        
        `$AUCMP = \sum_{i=2}^N_C \sum_{j < i} \sum_{y_m = i, y_n = j} (1/ (numSample_i numSample_j)) \cdot 1[\hat{y}_m > \hat{y}_n]$
        `

        Args:

        - `y_true`: the groundtruth
        - `y_pred`: the predicted score
        - `num_class`(optional): the number of classes, which will be calculated automatically from the    given data if not provided
        
        Shape:

        - `y_true`: A integer numpy vector of shape (numSample, )
        - `y_pred`: A real-valued numpy vector of shape (numSample, )

        Returns:

          `1 - auc`: the AUC score 
    '''
    # n = y_true.shape[0]
    if num_class is not None:
        if len(np.unique(y_true) != num_class):
            raise RuntimeError('Number of Classes Mismatch: the unique classes in y_true does not match Arg num_class ')
    
    classes = np.unique(y_true)

    def bin_auc(label, pred, i, j):
        isSelected = np.logical_or(label == i, label == j)

        score = pred[isSelected]
        label = label[isSelected] == i

        nP = label.sum()
        nN = label.shape[0] - nP
        sindex = np.argsort(score)
        lSorted = label[sindex]
        auc = (np.where(lSorted == 0) - np.arange(nN)).sum()
        auc /= (nN * nP)

        return 1 - auc

    return np.mean([
        bin_auc(y_true, y_pred, i, j) for (i, j) in permutations(classes, 2)
        if i > j
    ])

def argTopk(score, k):
    return argBottomk(-score, k)

def argBottomk(score, k):
    return devec(np.argpartition(devec(score), k)[:k])

def vec(x):
    return x.reshape(-1,1)

def devec(x):
    return x.squeeze() if x.shape[0] > 1 else x 

def subIndexNegK(score, label, rneg):
    negIndex = np.where(label != 1)[0]
    posIndex = np.where(label == 1)[0]
    nN = len(negIndex)
    kneg = math.floor(nN * rneg)
    negTopIndex = negIndex[argTopk(score[negIndex], kneg)]
    return devec(np.vstack((vec(posIndex), vec(negTopIndex))))

def subIndexNegPosK(score, label, rpos, rneg):
    negIndex = np.where(label != 1)[0]
    posIndex = np.where(label == 1)[0]
    nP, nN = len(posIndex), len(negIndex)
    kpos = math.floor(nP * rpos)
    kneg = math.floor(nN * rneg)
    negTopIndex = negIndex[argTopk(score[negIndex], kneg)]
    posBotIndex = posIndex[argBottomk(score[posIndex], kpos)]
    return devec(np.vstack((vec(posBotIndex), vec(negTopIndex))))

def auc_binary(y_true, y_pred):
    if len(y_true.shape) > 1:
        y_true = y_true.squeeze()

    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()

    label = y_true == 1
    nP = label.sum()
    nN = label.shape[0] - nP
    sindex = np.argsort(y_pred)
    lSorted = label[sindex]
    auc = (np.where(lSorted != True) - np.arange(nN)).sum()
    auc /= (nN * nP)

    return 1 - auc

def pAUC(y_true, y_pred, beta):
    subIndex = subIndexNegK(y_pred, y_true, beta)
    y_true = y_true[subIndex]
    y_pred = y_pred[subIndex]
    return auc_binary(y_true, y_pred)

def p2AUC(y_true, y_pred, alpha, beta):
    subIndex = subIndexNegPosK(y_pred, y_true, alpha, beta)
    y_true = y_true[subIndex]
    y_pred = y_pred[subIndex]
    # return roc_auc_score(y_true, y_pred)
    return auc_binary(y_true, y_pred)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
