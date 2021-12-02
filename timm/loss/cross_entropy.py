import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_loss import BaseLoss


__all__ = ['CrossEntropyLoss', 'BinaryCrossEntropyLoss', 'LabelSmoothingCrossEntropy']


class CrossEntropyLoss(BaseLoss):
    def __init__(self, num_classes):
        super(CrossEntropyLoss, self).__init__(num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        loss = self.loss_fn(logit, target)
        return loss.mean()

    def logit2label(self, logit):
        num_classes = self.num_classes
        pred = logit.argmax(1)
        sf = F.softmax(logit, 1)
        score = (sf * torch.arange(num_classes).cuda().view(-1, num_classes)).sum(1) / (num_classes - 1)        
        return pred, score

class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, num_classes):
        super(BinaryCrossEntropyLoss, self).__init__(num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        loss = self.loss_fn(logit.view(-1), target.float())
        return loss.mean()

    def logit2label(self, logit):
        pred = (logit > 0).long().view(-1)
        score = torch.sigmoid(logit).view(-1)
        return pred, score

class LabelSmoothingCrossEntropy(BaseLoss):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, num_classes, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__(num_classes)
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        logprobs = F.log_softmax(logit, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
    def logit2label(self, logit):
        num_classes = self.num_classes
        pred = logit.argmax(1)
        sf = F.softmax(logit, 1)
        score = (sf * torch.arange(num_classes).cuda().view(-1, num_classes)).sum(1) / (num_classes - 1)        
        return pred, score

# class SoftTargetCrossEntropy(nn.Module):

#     def __init__(self):
#         super(SoftTargetCrossEntropy, self).__init__()

#     def forward(self, x, target):
#         loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
#         return loss.mean()
