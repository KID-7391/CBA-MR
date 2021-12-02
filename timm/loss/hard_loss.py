import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from tqdm import tqdm

from .base_loss import BaseLoss


__all__ = ['LogitHard', 'HingeHard']


class BaseHard(BaseLoss):
    def __init__(self, num_classes, num_sync=4096, FPR=0.1):
        super(BaseHard, self).__init__(num_classes)
        self._thres = nn.Parameter(torch.zeros(num_classes - 1), requires_grad=False)
        # self._lr = lr
        self._FPR = FPR
        self._num_sync = num_sync
        # self._time_decay = time_decay
        self._sync_feats = torch.zeros(num_classes - 1, num_sync, 2560).cuda()
        self._weights = torch.ones(num_classes - 1, num_sync).cuda()
        self._index = torch.ones(num_classes - 1, num_sync).cuda(),

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
                # image = torch.index_select(image, 0, (target))
                feat = model.forward_features(image)
                score = model.head.fc(feat)
                feats.append(feat.data)
                scores.append(score.data)
                targets.append(target.data)
                indics.append(index.data)
                # print(feat.shape, len(feats))

        scores = torch.cat(scores, dim=0).view(-1)
        targets = torch.cat(targets, dim=0).view(-1)
        feats = torch.cat(feats, dim=0).view(targets.shape[0], -1)
        indics = torch.cat(indics, dim=0).view(-1)

        torch.save({
            'scores': scores,
            'targets': targets,
            'feats': feats
        }, 'debug.pth')

        # res = torch.load('debug.pth')
        # scores = res['scores']
        # targets = res['targets']
        # feats = res['feats']

        _sync_feats = torch.zeros(
            self.num_classes - 1,
            self._num_sync,
            model.num_features
        ).cuda()

        _index = torch.zeros(
            self.num_classes - 1,
            self._num_sync
        ).cuda()

        for i in range(self.num_classes - 1):
            idx = (targets == i).nonzero().view(-1)
            feats_neg = torch.index_select(feats, 0, idx)
            # scores_neg = torch.index_select(scores, 0, idx)
            idx_neg = torch.index_select(indics, 0, idx)

            # order = torch.argsort(-scores_neg)
            # scores_neg = scores_neg[order]

            if len(feats_neg) < self._num_sync:
                feats_neg = torch.cat(
                    [feats_neg] * int((self._num_sync - 1) / len(feats_neg) + 1),
                    dim=0
                )
                idx_neg = torch.cat(
                    [idx_neg] * int((self._num_sync - 1) / len(idx_neg) + 1),
                    dim=0
                )

            _sync_feats[i] = feats_neg[:self._num_sync]
            _index[i] = idx_neg[:self._num_sync]

        self._sync_feats = _sync_feats
        self._index = _index

    def update_thres(self, new_thres):
        with torch.no_grad():
            self._thres.data = torch.from_numpy(new_thres).cuda()

    def update_lr(self, new_lr):
        self._lr = lr
    
    def _find_nearest(self, a, b):
        assert len(a.shape) == 2 and len(b.shape) == 2
        num_dim = a.shape[1]
        assert b.shape[1] == num_dim
        dist = ((a.view(-1, 1, num_dim) - b.view(1, -1, num_dim))**2).mean(-1)
        idx = torch.argmin(dist, 1)
        return idx

    def _update_sync_feat(self, feat_neg, feat_sync):
        # num_dim = feat_neg.shape[1]
        # assert feat_sync.shape[1] == num_dim
        # dist = ((feat_neg.view(-1, 1, num_dim) 
        #    - feat_sync.view(1, -1, num_dim))**2).mean(-1)
        # idx = torch.argmax(dist, 1)# + torch.range(len(feat_neg)) * len(feat_sync)

        # feat_sync = torch.index_select(feat_sync, 0, idx)

        idx = self._find_nearest(feat_neg, feat_sync)
        feat_sync = torch.index_select(feat_sync, 0, idx)
        loss_sync = ((feat_sync - feat_neg.detach())**2).sum(-1)
        loss_neg = ((feat_sync.detach() - feat_sync)**2).sum(-1)
        loss = loss_sync + 0.1 * loss_neg
        return loss.mean()

    def forward(self, logit, target, feature, fc, index):
        logit, target = super().forward(logit, target)
        index = index.view(-1)
        loss = 0

        def select_by_clas(item, target, clas, pos=False):
            if pos:
                return torch.index_select(
                    item,
                    0,
                    (target > clas).nonzero().view(-1)
                )
            else:
                return torch.index_select(
                    item,
                    0,
                    (target <= clas).nonzero().view(-1)
                )

        # score = torch.sigmoid(logit).view(-1)

        try:
            self.cnt += 1
        except:
            self.cnt = 0

        # update memory
        for i in range(self.num_classes - 1):
            feat_cls = torch.index_select(
                feature, 0,
                (target == i).nonzero().view(-1)
            )
            idx_cls = torch.index_select(
                index, 0,
                (target == i).nonzero().view(-1)
            )

            n = len(feat_cls)
            feats_cat = torch.cat([
                self._sync_feats[i, n:],
                feat_cls
            ], dim=0)
            index_cat = torch.cat([
                self._index[i, n:],
                idx_cls
            ], dim=0)

            self._sync_feats[i] = feats_cat.detach()
            self._index[i] = index_cat.detach()

            # weights_cat = torch.cat([
            #     self._time_decay * self._weights[i],
            #     torch.ones(n).cuda()
            # ], dim=0)

            # l = fc(feats_cat).view(-1)
            # l = torch.sigmoid(l) * weights_cat
            # idx_keep = torch.argsort(-l)[:self._num_sync]
            # self._sync_feats[i].data = torch.index_select(feats_cat, 0, idx_keep)
            # self._weights[i].data = torch.index_select(weights_cat, 0, idx_keep)

        # cal rank loss
        sync_shape = self._sync_feats.shape
        logit_sync = fc(self._sync_feats.view(-1, sync_shape[2])).view(self.num_classes - 1, -1)
        for i in range(self.num_classes - 1):
            # sort the features
            # feat_i = self._sync_feats[:i+1].view((i + 1) * self._num_sync, -1)
            idx = torch.argsort(-logit_sync[:i+1].view(-1))
            tau = logit_sync[:i+1].view(-1)[idx][int(self._FPR * len(logit_sync))].detach()

            logit_pos = select_by_clas(logit, target, i, True)
            # print((logit <= tau).shape, (target <= i).shape, logit.shape, tau)
            logit_neg = torch.index_select(logit, 0, ((target <= i)).nonzero().reshape(-1))
            # logit_neg = select_by_clas(logit, target, i)
            if len(logit_pos) > 0:
                loss += self._surrogate_loss_fn(
                    logit_pos.view(-1, 1) - tau
                ).mean()

            if len(logit_neg) > 0:
                loss += self._surrogate_loss_fn(
                    - logit_neg
                ).mean()
            
            if self.cnt % 100 == 0:
                print(i, tau)

        return loss / (self.num_classes - 1)

    def logit2label(self, logit):
        score = torch.sigmoid(logit).view(-1)
        pred = score * 0
        return pred, score

class LogitHard(BaseHard):
    def __init__(self, num_classes, **kwargs):
        super(LogitHard, self).__init__(num_classes, **kwargs)

    def _surrogate_loss_fn(self, x):
        loss = torch.log(1 + torch.exp(-x))
        return loss

class HingeHard(BaseHard):
    def __init__(self, num_classes, **kwargs):
        super(HingeHard, self).__init__(num_classes, **kwargs)

    def _surrogate_loss_fn(self, x):
        loss = torch.relu(1 - x)
        return loss
