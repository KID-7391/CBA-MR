""" Classifier head and layer factory

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .linear import Linear


def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False, fc_type='fc'):
    flatten = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten)
    num_pooled_features = num_features * global_pool.feat_mult()
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_pooled_features, num_classes, 1, bias=True)
    else:
        # NOTE: using my Linear wrapper that fixes AMP + torchscript casting issue
        if fc_type == 'fc':
            fc = Linear(num_pooled_features, num_classes, bias=True)
        elif fc_type == 'mlp':
            fc = nn.Sequential(
                nn.BatchNorm1d(num_pooled_features),
                nn.ReLU(),
                Linear(num_pooled_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                Linear(512, num_classes, bias=True),
                nn.BatchNorm1d(1),
            )
            # nn.init.kaiming_normal_(fc[-1].weight, mode='fan_out', a=0.1)

    return global_pool, fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, self.fc = create_classifier(in_chs, num_classes, pool_type=pool_type)

    def forward(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


class ClassifierMLPHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.):
        super(ClassifierMLPHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, self.fc = create_classifier(in_chs, num_classes, pool_type=pool_type, fc_type='mlp')

    def forward(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


class MultiRankTreeHead(nn.Module):
    """
    An implementation of tree multipartite ranking :

    Args:
    - `in_chs`: the number of input channels

    - `mid_chs`: the number of features in tree

    - `num_classes`: the number of classes in the given dataset

    """
    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0., mid_chs=2048):
        super(MultiRankTreeHead, self).__init__()

        self.drop_rate = drop_rate
        self.mid_chs = mid_chs
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
        self.fc_in = Linear(in_chs, mid_chs, bias=True)
        self.lstm = nn.LSTM(mid_chs, mid_chs, dropout=drop_rate)
        self.build_tree(num_classes)
        self.eps = 1e-6

    def build_tree(self, num_classes):
        self.fa = [0] * (2 * num_classes - 1)
        self.is_left = [None] * (2 * num_classes - 1)
        self.interval = [[0, num_classes - 1]] * (2 * num_classes - 1)
        self.score_fn = nn.ModuleList(
            [nn.Sequential(
                # nn.BatchNorm1d(self.mid_chs),
                Linear(self.mid_chs, 1, bias=True)
            ) for _ in  range(num_classes - 1)]
        )

        idx_node = 1
        queue = [0]
        while len(queue) > 0:
            x = queue[0]
            queue = queue[1:]

            intv_x_l = self.interval[x][0]
            intv_x_r = self.interval[x][1]
            if intv_x_r - intv_x_l == 0:
                continue
            elif intv_x_r - intv_x_l == 1:
                idx_left = intv_x_l + num_classes - 1
                idx_right = intv_x_r + num_classes - 1
            elif intv_x_r - intv_x_l == 2:
                idx_left = idx_node
                idx_right = intv_x_r + num_classes - 1
                idx_node += 1
            else:
                idx_left = idx_node
                idx_right = idx_node + 1
                idx_node += 2

            queue += [idx_left, idx_right]

            self.fa[idx_left] = x
            self.fa[idx_right] = x

            self.is_left[idx_left] = True
            self.is_left[idx_right] = False

            self.interval[idx_left] = [intv_x_l, (intv_x_l + intv_x_r) // 2]
            self.interval[idx_right] = [(intv_x_l + intv_x_r) // 2 + 1, intv_x_r]

    def forward_node(self, x, node_idx, h=None):
        batchsize = x.shape[0]
        if node_idx == 0:
            h = (torch.zeros_like(x).float().to(x.device).view(1, batchsize, -1),
                torch.zeros_like(x).float().to(x.device).view(1, batchsize, -1))
        assert h is not None

        x = x.view(1, batchsize, -1)
        o_cur, h_cur = self.lstm(x, h)
        o_cur = self.score_fn[node_idx](o_cur.squeeze(0))
        return o_cur, h_cur

    def forward_tree(self, x):
        o_all = [None] * (self.num_classes - 1)
        h_all = [None] * (self.num_classes - 1)
        logit_all = [None] * (2 * self.num_classes - 1)
        logit = [None] * self.num_classes
        batchsize = x.shape[0]

        for i in range(self.num_classes - 1):
            h_fa = h_all[self.fa[i]]
            o, h = self.forward_node(x, i, h_fa)
            o_all[i] = o
            h_all[i] = h

        logit_local = torch.stack(o_all, 0).view(-1, batchsize)
        logit_all[0] = torch.ones(batchsize).float().to(x.device)
        for i in range(1, 2 * self.num_classes - 1):
            logit_fa = logit_all[self.fa[i]]
            if self.is_left[i]:
                logit_all[i] = logit_fa + torch.log(1 + torch.exp(logit_local[self.fa[i]]))
            else:
                logit_all[i] = logit_fa + torch.log(1 + torch.exp(-logit_local[self.fa[i]]))
            
            if i >= self.num_classes - 1:
                logit[i - self.num_classes + 1] = logit_all[i]

        # score = torch.stack(score, dim=1)
        # logit = torch.log(score + self.eps)
        logit = torch.stack(logit, dim=1)
        return logit

    def forward(self, x):
        x = self.global_pool(x)
        x = self.fc_in(x)
        return self.forward_tree(x)
