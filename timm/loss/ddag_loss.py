import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from .base_loss import BaseLoss


__all__ = ['CR_DDAG', 'PR_DDAG']


class DDAG(BaseLoss):
    def __init__(self, num_classes):
        super(DDAG, self).__init__(num_classes)
        node = [None for _ in  range((1 + num_classes) * num_classes // 2)]
        index = 0
        visited = dict()

        queue = [(0, num_classes - 1, -1, 0)]
        while len(queue) > 0:
            i,j,fa,right = queue[0]
            # print(queue[0], index, (i,j) in visited.keys())
            queue = queue[1:]
            if (i,j) in visited.keys():
                if fa >= 0:
                    node[fa][-1][right] = visited[(i,j)]    
                continue
            visited[(i,j)] = index
            node[index] = [i, j, index, [-1, -1]]
            if fa >= 0:
                node[fa][-1][right] = index
            index += 1
            if i == j:
                continue

            queue.append((i, j - 1, index - 1, 0))
            queue.append((i + 1, j, index - 1, 1))

        self._node = node

    def forward(self, logit, target):
        logit, target = super().forward(logit, target)
        loss = 0
        for i,j,index,_ in self._node:
            if i == j:
                break
            logit_i = torch.index_select(logit, 0, (target == i).nonzero().view(-1))            
            logit_j = torch.index_select(logit, 0, (target == j).nonzero().view(-1))
            loss += torch.nn.BCEWithLogitsLoss()(
                torch.cat([logit_i[:, index], logit_j[:, index]], 0),
                torch.cat([torch.zeros(len(logit_i)).cuda().float(), torch.ones(len(logit_j)).cuda().float()], 0),
            )

        return loss

    @abstractmethod
    def logit2label(self, logit):
        pass
        score = torch.sigmoid(logit).view(-1)
        pred = score * 0
        return pred, score

class CR_DDAG(DDAG):
    def __init__(self, num_classes):
        super(CR_DDAG, self).__init__(num_classes)

    def logit2label(self, logit):
        # num_classes = self.num_classes

        # score_sub = torch.sigmoid(logit)
        # # p_k = [None for _ in range(len(self._node))]
        # pred = []
        # score = []
        # for i in range(len(logit)):
        #     score_i = score_sub[i]
        #     k = 0
        #     while True:
        #         next_idx = 0 if score_i[k] < 0.5 else 1
        #         next_k = self._node[k][-1][next_idx]
        #         if next_k >= len(score_i):
        #             pred.append(next_k - (num_classes - 1) * num_classes // 2)
        #             score.append(pred[-1] + score_i[k])
        #             break
        #         k = next_k

        num_classes = self.num_classes

        score_sub = torch.sigmoid(logit)
        select_k = torch.zeros(len(self._node), len(logit)).cuda().float()
        p_k = torch.zeros(len(self._node), len(logit)).cuda().float()
        pred = []
        score = []

        select_k[0] += 1
        for nd in self._node:
            i,j,index,(son_l, son_r) = nd
            if son_l < 0:
                break

            select_k[son_l] += select_k[index] * (score_sub[:,index] <= 0.5).float()
            select_k[son_r] += select_k[index] * (score_sub[:,index] > 0.5).float()
            p_k[son_l] += (1 - score_sub[:,index]) * select_k[index] * (score_sub[:,index] <= 0.5).float()
            p_k[son_r] += score_sub[:,index] * select_k[index] * (score_sub[:,index] > 0.5).float()

        pred = torch.argmax(select_k[-num_classes:], 0)
        score = pred + p_k[-num_classes:].sum(0)
        # print(select_k[-num_classes:].sum(0))
        # print(p_k[-num_classes:])
        # score = (torch.arange(1, num_classes + 1).view(-1, 1).cuda() * select_k[-num_classes:]).sum(0) + 

        return torch.tensor(pred), torch.tensor(score)

class PR_DDAG(DDAG):
    def __init__(self, num_classes):
        super(PR_DDAG, self).__init__(num_classes)

    def logit2label(self, logit):
        num_classes = self.num_classes

        score_sub = torch.sigmoid(logit)
        p_k = torch.zeros(len(self._node), len(logit)).cuda().float()
        pred = []
        score = []

        p_k[0] += 1
        for nd in self._node:
            i,j,index,(son_l, son_r) = nd
            if son_l < 0:
                break

            p_k[son_l] += p_k[index] * (1 - score_sub[:,index])
            p_k[son_r] += p_k[index] * score_sub[:,index]

        score = (torch.arange(1, num_classes + 1).view(-1, 1).cuda() * p_k[-num_classes:]).sum(0)
        pred = torch.argmax(p_k[-num_classes:], 0)

        return pred, score

if __name__ == '__main__':
    num_classes = 5
    logit = torch.randn(20, num_classes * (num_classes - 1) // 2).cuda()
    target = torch.tensor([0,1,2,3,4,0,1,2,3,4,4,3,2,1,0,0,3,2,1,2]).cuda()
    cr = CR_DDAG(num_classes).cuda()
    pr = PR_DDAG(num_classes).cuda()
    # for i in cr._node:
    #     print(i)

    loss = cr(logit, target)
    print(loss)

    pred, score = cr.logit2label(logit)
    # pred, score = pr.logit2label(logit)
    print(score, pred)
    print(score - pred)
