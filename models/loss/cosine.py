import torch
from torch import nn
import torch.functional as F

class CosineLoss(torch.nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, pred, label):
        cosine_sim = F.cosine_similarity(pred, label)
        loss = torch.pow(1 - cosine_sim, 2)
        cosine_loss = torch.mean(loss)
        return cosine_loss


class LogCoshLoss(nn.Module):
    """
    Log-cosh is another function used in regression tasks that’s smoother than L2.
     Log-cosh is the logarithm of the hyperbolic cosine of the prediction error.
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_target):
        e = torch.log(torch.cosh(y_pred - y_target))/y_target.shape[0]
        return e
