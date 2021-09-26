""" Free Energy Bound Loss for VAEs with Normalizing Flows """
from models.networks.generative.flows.nf import safe_log
from torch import nn


class FreeEnergyBound(nn.Module):

    def __init__(self, density):
        super().__init__()

        self.density = density

    def forward(self, zk, log_jacobians):

        sum_of_log_jacobians = sum(log_jacobians)
        return (-sum_of_log_jacobians - safe_log(self.density(zk))).mean()