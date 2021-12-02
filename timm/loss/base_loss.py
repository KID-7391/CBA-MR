import torch
import torch.nn as nn
import torch.nn.functional as F
import abc


class BaseLoss(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(BaseLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logit, target):
        n = len(logit)
        logit = logit.view(n, -1).float()
        target = target.view(n).long()
        return logit, target

    @abc.abstractmethod
    def logit2label(self, logit):
        pass
