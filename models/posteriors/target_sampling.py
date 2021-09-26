import torch
from torch import nn

class TargetSampling(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None):
        super(TargetSampling, self).__init__()

    def forward(self, x):
        pass


if __name__ == "__main__":
    pass