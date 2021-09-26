import torch
from torch import nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, y_pred, y_target):
        e = torch.max(torch.abs(0.5 - y_pred * y_target), dim=0)/y_target.shape[0]
        return e