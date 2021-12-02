import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from .base_loss import BaseLoss


__all__ = ['NNRank']


class NNRank(BaseLoss):
    def __init__(self, num_classes):
        super(NNRank, self).__init__(num_classes)
        pass

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        score = torch.sigmoid(logit)
        loss = 0
        for i in range(self.num_classes - 1):
            score_i = score[:, i]
            target_i = (target > i).float()
            loss += ((score_i - target_i)**2).mean()
            # loss += torch.nn.BCEWithLogitsLoss()(logit_i, target_i)

        return loss / (self.num_classes - 1)

    def logit2label(self, logit):
        score = torch.sigmoid(logit).view(-1, self.num_classes - 1).sum(-1)
        pred = score * 0
        return pred, score
