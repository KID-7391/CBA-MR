import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from tqdm import tqdm

from .base_loss import BaseLoss
from timm.utils import reduce_tensor


__all__ = ['LogitCBA', 'HingeCBA']


class BaseCBA(BaseLoss):
    def __init__(self, num_classes, num_sync=512, FPR=0.1, weight=40, sigma=0.5, world_size=1, rank=0):
        # sigma = 0.05 0.1 0.3 0.5 0.7 1 
        super(BaseCBA, self).__init__(num_classes)
        self._weight = weight
        self._FPR = FPR
        self._sigma = sigma
        self._num_sync = num_sync
        self._index = torch.ones(num_classes, num_sync).cuda()
        self._world_size = world_size
        self._rank = rank

    @abstractmethod
    def _surrogate_loss_fn(self, score, target):
        pass

    def init_sync_feats(self, dataloader, model):
        feats = []
        scores = []
        targets = []
        indics = []
        model.eval()
        with torch.no_grad():
            print('Initializing sync features...')
            for batch_idx, (image, target, index) in enumerate(tqdm(dataloader)):
                image = image.cuda()
                target = target.cuda().view(-1)
                index = index.cuda()
                feat = model.forward_features(image)
                score = model.head.fc(feat)
                feats.append(feat.data)
                scores.append(score.data)
                targets.append(target.data)
                indics.append(index.data)

        scores = torch.cat(scores, dim=0).view(-1)
        targets = torch.cat(targets, dim=0).view(-1)
        feats = torch.cat(feats, dim=0).view(targets.shape[0], -1)
        indics = torch.cat(indics, dim=0).view(-1)

        # torch.save({
        #     'indics': indics,
        #     'scores': scores,
        #     'targets': targets,
        #     'feats': feats
        # }, 'debug.pth')

        _sync_feats = torch.zeros(
            self.num_classes,
            self._num_sync,
            model.num_features
        ).cuda()

        _index = torch.zeros(
            self.num_classes,
            self._num_sync
        ).cuda()

        for i in range(self.num_classes):
            idx = (targets == i).nonzero().view(-1)
            feats_i = torch.index_select(feats, 0, idx)
            idx_neg = torch.index_select(indics, 0, idx)

            if len(feats_i) < self._num_sync:
                feats_i = torch.cat(
                    [feats_i] * int((self._num_sync - 1) / len(feats_i) + 1),
                    dim=0
                )
                idx_neg = torch.cat(
                    [idx_neg] * int((self._num_sync - 1) / len(idx_neg) + 1),
                    dim=0
                )

            _sync_feats[i] = feats_i[:self._num_sync]
            _index[i] = idx_neg[:self._num_sync]

        self._sync_feats = torch.zeros(
            self.num_classes,
            self._world_size * _sync_feats.shape[1],
            _sync_feats.shape[2]
        ).cuda()
        
        self._sync_feats[:, self._rank*_sync_feats.shape[1]:(self._rank+1)*_sync_feats.shape[1]] = _sync_feats
        if self._world_size > 1:
            self._sync_feats = reduce_tensor(self._sync_feats, 1)

    def forward(self, logit, target, feature, fc, index, **kwargs):
        logit, target = super().forward(logit, target)
        index = index.view(-1)
        loss = torch.zeros(1).cuda()

        n = len(feature)
        dim_feat = feature.shape[1]

        # update memory
        _sync_feats = torch.zeros_like(self._sync_feats).cuda()
        for i in range(self.num_classes):
            feat_cls = torch.index_select(
                feature, 0,
                (target == i).nonzero().view(-1)
            )
            n = len(feat_cls)
            feats_cat = torch.cat([
                self._sync_feats[i, self._rank * self._num_sync + n : (self._rank+1) * self._num_sync],
                feat_cls
            ], dim=0)
            _sync_feats[i, self._rank * self._num_sync : (self._rank+1) * self._num_sync] = feats_cat.detach()

        if self._world_size > 1:
            self._sync_feats = reduce_tensor(_sync_feats, 1)

        for i in range(self.num_classes - 1):
            # idx = (target > i).nonzero().view(-1)
            logit_pos = torch.index_select(logit, 0, (target > i).nonzero().view(-1))
            logit_neg = torch.index_select(logit, 0, (target <= i).nonzero().view(-1))

            if len(logit_pos) == 0 or len(logit_neg) == 0:
                continue

            ########################################
            mem = self._sync_feats[:i+1].view(-1, dim_feat)

            with torch.no_grad():
                logit_mem = fc(mem).detach().view(-1)
            idx_mem = torch.argsort(logit_mem)
            rank_mem = torch.argsort(idx_mem).float() / len(mem)

            feat_neg = torch.index_select(feature, 0, (target <= i).nonzero().view(-1))
            feat_neg = F.normalize(feat_neg, dim=1).view(-1, 1, dim_feat)
            mem = F.normalize(mem, dim=1).view(1, -1, dim_feat)
            similarity = torch.softmax(self._weight * (feat_neg * mem).sum(-1), dim=-1)

            rank_hat = (similarity * rank_mem.view(1, -1)).sum(-1)
            weight = torch.exp(- ((1 - rank_hat) - self._FPR)**2 / self._sigma**2).view(1, -1).detach()
            weight *= len(logit) / weight.sum()
            ########################################

            #######################################
            loss_i = self._surrogate_loss_fn(
                logit_pos.view(-1, 1) - logit_neg.view(1, -1)
            )

            loss += (loss_i * weight).mean()
            #######################################

        return loss / (self.num_classes - 1)

    def logit2label(self, logit):
        score = torch.sigmoid(logit).view(-1)
        pred = score * 0
        return pred, score

class LogitCBA(BaseCBA):
    def __init__(self, num_classes, **kwargs):
        super(LogitCBA, self).__init__(num_classes, **kwargs)

    def _surrogate_loss_fn(self, x):
        loss = torch.log(1 + torch.exp(-x))
        return loss

class HingeCBA(BaseCBA):
    def __init__(self, num_classes, **kwargs):
        super(HingeCBA, self).__init__(num_classes, **kwargs)

    def _surrogate_loss_fn(self, x):
        loss = torch.relu(1 - x)
        return loss
