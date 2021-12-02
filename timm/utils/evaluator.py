import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm
import os
import os.path as osp
import cv2

from timm.utils import reduce_array
from .metrics import aucMP, pAUC


class Evaluator(object):
    def __init__(self, save_root=None, data_config=None, FPR=0.1):
        self.reset()

        self.FPR = FPR

        if save_root is not None:
            self.saver = Saver(save_root, data_config)
        else:
            self.saver = None

    # def acc(self):
    #     return (self.y_pred == self.y_gt).sum() / len(self.y_pred)

    def recall_mean(self):
        return metrics.recall_score(self.y_gt_array, self.y_pred_array, average='macro')

    def precision_mean(self):
        return metrics.precision_score(self.y_gt_array, self.y_pred_array, average='macro')

    def precision_all(self):
        return metrics.precision_score(self.y_gt_array, self.y_pred_array, average='micro')

    def F1_mean(self):
        return metrics.f1_score(self.y_gt_array, self.y_pred_array, average='macro')

    def auc(self):
        return aucMP(self.y_gt_array, self.score_array)

    def pauc(self, cls_thres):
        y_gt = self.y_gt_array.copy()
        y_gt[y_gt <= cls_thres] = 0
        y_gt[y_gt > cls_thres] = 1
        return pAUC(y_gt, self.score_array, beta=0.1)

    def tpr_with_thres(self, cls_thres):
        target_fpr = self.FPR
        y_gt = self.y_gt_array.copy()
        score = self.score_array.copy()

        y_gt[y_gt <= cls_thres] = 0
        y_gt[y_gt > cls_thres] = 1

        min_dist = 1e6
        final_tpr = 0
        final_thres = 0

        sorted_score = np.sort(score)[::-1]
        for thres in sorted_score:
            pred = (score > thres).astype(np.int32)
            tp = ((pred == y_gt) * (y_gt == 1)).sum()
            tn = ((pred == y_gt) * (y_gt == 0)).sum()
            fp = ((pred != y_gt) * (y_gt == 0)).sum()
            fn = ((pred != y_gt) * (y_gt == 1)).sum()
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)

            if (fp + tn) == 0 or (tp + fn) == 0:
                continue

            if abs(fpr - target_fpr) < min_dist:
                min_dist = abs(fpr - target_fpr)
                final_tpr = tpr
                final_thres = thres

        return final_tpr, final_thres

    def pauc_m(self):
        num_classes = int(self.y_gt_array.max() + 1)
        res = 0
        for c in range(num_classes - 1):
            res += self.pauc(c)
        return res / (num_classes - 1)

    def tpr_with_thres_m(self):
        num_classes = int(self.y_gt_array.max() + 1)
        tpr = []
        thres = []
        res = 0
        for c in range(num_classes - 1):
            r = self.tpr_with_thres(c)
            res += r[0]
            tpr.append(r[0])
            thres.append(r[1])
        return res / (num_classes - 1), tpr, thres

    def add_batch(self, logit, gt, loss_fn, image=None, feature=None):
        pred, score = self.logit2pred_and_score(logit.detach(), loss_fn)
        gt = gt.detach()

        if self.y_pred is None:
            self.y_pred = pred
        else:
            self.y_pred = torch.cat([self.y_pred, pred], 0)

        if self.y_gt is None:
            self.y_gt = gt
        else:
            self.y_gt = torch.cat([self.y_gt, gt], 0)

        if self.score is None:
            self.score = score
        else:
            self.score = torch.cat([self.score, score], 0)

        if self.logit is None:
            self.logit = logit.view(-1)
        else:
            self.logit = torch.cat([self.logit, logit.view(-1)], 0)

        if feature is not None:
            try:
                self.feat = torch.cat([self.feat, feature.data], 0)
            except:
                self.feat = feature.data

        if self.saver:
            self.saver.save(image, pred, score, gt)
        
    def hist(self):
        # score = self.score_array
        # y_gt = self.y_gt_array
        # hist = np.zeros((4, 10))
        # for i in range(len(score)):
        #     for j in range(10):
        #         if score[i] < (j + 1) * 0.1:
        #             hist[y_gt[i], j] += 1
        #             break

        # np.save('hist.npy', hist)
        np.save(
            'result.npy',
            {
                'feature': self.feat.cpu().numpy(),
                'score': self.score_array,
                'y_pred': self.y_pred_array,
                'y_gt': self.y_gt_array    
            }
        )

    def get_metrics(self, n):
        self.y_pred_array = (self.y_pred + 0.05).cpu().numpy().astype(np.int32)
        self.logit_array = self.logit.cpu().numpy()
        # self.score = torch.sigmoid(self.logit - self.logit.mean())
        self.score_array = self.score.cpu().numpy()
        self.y_gt_array = (self.y_gt + 0.05).cpu().numpy().astype(np.int32)

        # rec = self.recall_mean()
        # prec = self.precision_mean()
        # f1 = self.F1_mean()
        # auc = self.auc()
        pauc = self.pauc_m()
        tpr_mean, tpr, thres = self.tpr_with_thres_m()

        # self.hist()

        res = {
            # 'recall': np.array(rec),
            # 'precision': np.array(prec),
            # 'f1': np.array(f1),
            # 'auc': np.array(auc),
            # 'pauc': np.array(pauc),
            'tpr_m': np.array(tpr_mean),
            'tpr': np.array(tpr),
            'thres': np.array(thres)
        }

        if n > 1:
            for key in res.keys():
                res[key] = reduce_array(res[key], n)

        return res

    def logit2pred_and_score(self, logit, loss_fn, **kwargs):
        assert len(logit.shape) == 2
        assert isinstance(logit, torch.Tensor)

        num_classes = logit.shape[1]
        return loss_fn.logit2label(logit)

    def reset(self):
        self.score = None
        self.y_pred = None
        self.y_gt = None
        self.logit = None


class Saver(object):
    def __init__(self, save_root, data_config):
        if not osp.exists(save_root):
            os.makedirs(save_root)
        self.save_root = save_root
        self.mean = np.array(data_config['mean']).reshape((1,1,1,3))
        self.std = np.array(data_config['std']).reshape((1,1,1,3))
        self.cnt = 0

        self.mean_conf = np.zeros((4, 4, 2))

    def save(self, images, preds, scores, targets):

        if isinstance(images, torch.Tensor):
            images = images.permute(0, 2, 3, 1)
            images = images.cpu().numpy()
            images *= self.std
            images += self.mean
            images = images[...,::-1]

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy().flatten()

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy().flatten()

        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy().flatten()

        for i in range(len(images)):
            path = osp.join(self.save_root, 'lbl_%d_pred_%d_%.2f_%d.jpg'%(targets[i], preds[i], scores[i], self.cnt))
            self.cnt += 1
            img = (images[i] * 255).astype(np.uint8)
            if self.cnt % 10 == 0:
                cv2.imwrite(path, img)

            self.mean_conf[targets[i], preds[i], 0] += scores[i]
            self.mean_conf[targets[i], preds[i], 1] += 1


def predict_bd_auc(logit, threshold=None):
    """
    Transfer predicted scores into classes

    Args:

    - `score`: the predicted score vector

    Returns:

    - `y_pred`: predicted classification results
    - `score_all`: overall scores (for AUC calculation)

    """

    score = F.sigmoid(logit)

    score_all = score.mean(1)

    if threshold is not None:
        y_pred = torch.zeros_like(score_all)
        for i, th in enumerate(threshold):
            y_pred = torch.where(score_all > th, torch.ones_like(y_pred) * (i+1), y_pred)
    else:
        n = score.shape[0]
        score = torch.cat([
            torch.ones(n, 1).to(score.device),
            score,
            torch.zeros(n, 1).to(score.device)
        ], dim=1)

        diff = score[:,:-1] - score[:,1:]
        y_pred = torch.argmax(diff, dim=1)

    return y_pred, score_all