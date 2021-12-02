import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from .base_loss import BaseLoss


__all__ = ['FandH']


class FandH(BaseLoss):
    def __init__(self, num_classes):
        super(FandH, self).__init__(num_classes)
        pass

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        loss = 0
        for i in range(self.num_classes - 1):
            logit_i = logit[:, i]
            target_i = (target > i).float()
            loss += torch.nn.BCEWithLogitsLoss()(logit_i, target_i)
            # logit_pos = torch.index_select(logit, 0, (target > i).nonzero().view(-1))
            # logit_neg = torch.index_select(logit, 0, (target <= i).nonzero().view(-1))
            # if len(logit_pos) == 0 or len(logit_neg) == 0:
            #     continue
            # loss += self._surrogate_loss_fn((logit_pos - logit_neg).view(-1)).mean()

        return loss / (self.num_classes - 1)

    def logit2label(self, logit):
        score = torch.sigmoid(logit).view(-1, self.num_classes - 1).sum(-1)
        pred = score * 0
        return pred, score
