import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from .base_loss import BaseLoss


__all__ = ['LogitRank', 'HingeRank']


class BaseRank(BaseLoss):
    def __init__(self, num_classes):
        super(BaseRank, self).__init__(num_classes)
        pass

    @abstractmethod
    def _surrogate_loss_fn(self, score, target):
        pass

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        loss = 0
        for i in range(self.num_classes - 1):
            logit_pos = torch.index_select(logit, 0, (target > i).nonzero().view(-1))
            logit_neg = torch.index_select(logit, 0, (target <= i).nonzero().view(-1))
            logit_pos = logit_pos.view(-1, 1)
            logit_neg = logit_neg.view(1, -1)
            loss += self._surrogate_loss_fn((logit_pos - logit_neg).view(-1)).mean()

        return loss / (self.num_classes - 1)

    def logit2label(self, logit):
        score = torch.sigmoid(logit).view(-1)
        pred = score * 0
        return pred, score

class LogitRank(BaseRank):
    def __init__(self, num_classes=2):
        super(LogitRank, self).__init__(num_classes)

    def _surrogate_loss_fn(self, x):
        loss = torch.log(1 + torch.exp(-x))
        return loss

class HingeRank(BaseRank):
    def __init__(self, num_classes=2):
        super(HingeRank, self).__init__(num_classes)

    def _surrogate_loss_fn(self, x):
        loss = torch.relu(1 - x)
        return loss
