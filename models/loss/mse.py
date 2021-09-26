import torch
from torch import nn
import numpy as np


class MAE(nn.Module):
    """
    Calculates the mean absolute error.
    """
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, y_pred, y_target):
        absolute_errors = torch.abs(y_pred - y_target.view_as(y_pred))
        return torch.sum(absolute_errors).item()/y_target.shape[0]


class RMSLE:
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
    Used for better separation between predictions which can be minimal
    Args:
        p - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """

    def __init__(self):
        super(RMSLE, self).__init__()

    def forward(self, p, y):
        return np.sqrt(torch.square(torch.log(p + 1) - torch.log(y + 1)).mean())