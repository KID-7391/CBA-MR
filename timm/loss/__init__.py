from .cross_entropy import *
from .rank_loss import *
from .hard_loss import *
from .F_and_H import *
from .cba_loss import *
from .nnrank import *
from .soft import *
from .ddag_loss import *
from .Eban import *


def gen_loss(loss_fn_type, args):
    if loss_fn_type == 'CE':
        loss_fn = CrossEntropyLoss(args.num_classes)
    elif loss_fn_type == 'BCE':
        loss_fn = BinaryCrossEntropyLoss(args.num_classes)
    elif loss_fn_type == 'CE_smooth':
        loss_fn = LabelSmoothingCrossEntropy(args.num_classes)
    elif loss_fn_type == 'logit_rank':
        loss_fn = LogitRank(args.num_classes)
    elif loss_fn_type == 'hinge_rank':
        loss_fn = HingeRank(args.num_classes)
    elif loss_fn_type == 'logit_hard':
        loss_fn = LogitHard(args.num_classes, FPR=args.FPR)
    elif loss_fn_type == 'hinge_hard':
        loss_fn = HingeHard(args.num_classes, FPR=args.FPR)
    elif loss_fn_type == 'FandH':
        loss_fn = FandH(args.num_classes)
    elif loss_fn_type == 'logit_cba':
        loss_fn = LogitCBA(
            args.num_classes,
            FPR=args.FPR,
            num_sync=args.num_sync,
            sigma=args.sigma,
            world_size=args.world_size,
            rank=args.rank
        )
    elif loss_fn_type == 'hinge_cba':
        loss_fn = HingeCBA(
            args.num_classes,
            FPR=args.FPR,
            num_sync=args.num_sync,
            sigma=args.sigma,
            world_size=args.world_size,
            rank=args.rank
        )
    elif loss_fn_type == 'nnrank':
        loss_fn = NNRank(args.num_classes)
    elif loss_fn_type == 'soft':
        loss_fn = SoftLabel(args.num_classes)
    elif loss_fn_type == 'cr_ddag':
        loss_fn = CR_DDAG(args.num_classes)
    elif loss_fn_type == 'pr_ddag':
        loss_fn = PR_DDAG(args.num_classes)
    elif loss_fn_type == 'eban':
        loss_fn = Eban(args.num_classes, args.FPR)
    else:
        raise NotImplementedError("Unknown loss function: %s"%args.loss_fn)

    loss_fn.cuda()
    return loss_fn
