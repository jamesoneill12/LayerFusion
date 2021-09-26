from models.loss.ce import *
from models.loss.cosine import *
from models.loss.divergences import *
from models.loss.hinge import *
from models.loss.huber import *
from models.loss.mse import *
from models.loss.old_vonmises import VonMisesLoss
from models.loss.adaptive import AdaptiveLoss
from models.loss.sinkhorn import SinkhornDistance
from models.loss.nce import IndexLinear


def get_criterion(loss, cutoff=None, nhid=None, ntoken = None, rs=False,
                  noise=None, noise_ratio=None, norm_term=None, sinkhorn_eps = 0.1,
                  sinkhorn_iter = 100, code_size=None, temp=1.):

    if loss == 'ada' and cutoff is not None:
        return AdaptiveLoss(cutoff)
    elif loss == 'ce':
        return nn.CrossEntropyLoss()
    elif loss == "nce" and nhid is not None and ntoken is not None:
        criterion = IndexLinear(
            nhid,
            ntoken,
            noise=noise,
            noise_ratio=noise_ratio,
            norm_term=norm_term,
            # loss_type=args.loss,
            reduction='none',
        )
        return criterion
    elif loss == 'rce':
        return RewardSmoother
    elif loss == 'rce_neighbor':
        """FINISH"""
        return print("rce neighbor still needs to be implemented")
    elif loss == 'mse':
        return nn.MSELoss()
    elif loss == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    elif loss == 'power_bce':
        """ 
        similar to bce only that loss is weighted average
        where weights correspond to bit position
        """
        if code_size is None:
            ValueError("Need to enter code_size when using power_bce")
        return PowerBCEWithLogitsLoss(code_size, temp = temp)
    elif loss == 'ce_all' or rs:
        return CrossEntropyLoss
    elif loss == 'mae':
        return MAE()
    elif loss == 'huber':
        return HuberLoss()
    elif loss == 'logcosh':
        return LogCoshLoss()
    elif loss == 'hinge':
        return HingeLoss()
    elif loss == 'bce':
        return nn.BCELoss()
    elif loss == 'jsd':
        return js_loss
    elif loss == 'mmd':
        return mmd
    elif loss == 'cosine':
        return CosineLoss()
    elif loss == 'sinkhorn':
        return SinkhornDistance(eps=sinkhorn_eps, max_iter=sinkhorn_iter, reduction=None)
    elif loss == 'vonmises':
        return VonMisesLoss


