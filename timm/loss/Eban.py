import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from .base_loss import BaseLoss


__all__ = ['Eban']


class Eban(BaseLoss):
    def __init__(self, num_classes, alpha, lr=1e-3):
        super(Eban, self).__init__(num_classes)
        self._lambda = nn.Parameter(torch.ones(num_classes - 1), requires_grad=False)
        self._alpha = alpha
        self._lr = lr

    def _surrogate_loss_fn(self, logit, target):
        y = 2 * target - 1
        loss = torch.log(1 + torch.exp( - y * logit.view(-1)))
        return loss

    def _update_lambda(self, loss_neg, i):
        self._lambda[i].data += self._lr * (loss_neg.mean() - self._alpha)
        self._lambda[i].data = torch.clamp(self._lambda[i].data, 0, 1e5)

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        losses = 0
        for i in range(self.num_classes - 1):
            # logit_i = logit[:, i]
            target_i = (target > i).float()
            # loss = torch.nn.BCEWithLogitsLoss(reduction='none')(logit_i, target_i)
            loss = self._surrogate_loss_fn(logit, target_i)

            loss_pos = torch.index_select(loss, 0, target_i.nonzero().view(-1))
            loss_neg = torch.index_select(loss, 0, (1 - target_i).nonzero().view(-1))

            self._update_lambda(loss_neg, i)
            losses += loss_pos.mean() + self._lambda[i] * loss_neg.mean()

        return losses / (self.num_classes - 1)

    def logit2label(self, logit):
        score = torch.sigmoid(logit).view(-1, 1).sum(-1)
        pred = score * 0
        return pred, score
