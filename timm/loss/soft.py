import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from .base_loss import BaseLoss


__all__ = ['SoftLabel']


class SoftLabel(BaseLoss):
    def __init__(self, num_classes):
        super(SoftLabel, self).__init__(num_classes)
        pass

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        r = torch.arange(self.num_classes).cuda()
        dist = - torch.abs(target.view(-1, 1) - r.view(1, -1)).float()
        target_soft = torch.softmax(dist, dim=1)

        logit = F.log_softmax(logit, dim=-1)
        loss = nn.KLDivLoss(reduction='batchmean')(logit, target_soft)

        # logit = F.log_softmax(logit, dim=-1)
        # loss = - (target_soft * logit).sum(-1).mean()



        return loss

    def logit2label(self, logit):
        num_classes = self.num_classes
        pred = logit.argmax(1)
        sf = F.softmax(logit, 1)
        score = (sf * torch.arange(num_classes).cuda().view(-1, num_classes)).sum(1) / (num_classes - 1)        
        return pred, score
