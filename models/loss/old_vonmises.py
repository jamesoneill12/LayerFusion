import torch
import math
from torch.autograd import Variable
from torch import nn
import scipy.special as sc
import numpy as np


def ratio(v, z):
    # return z/(v-0.5+torch.sqrt((v-0.5)*(v-0.5) + z*z))
    return z / (v - 1 + torch.sqrt((v + 1) * (v + 1) + z * z))


class Logcmk(torch.autograd.Function):
    """
   The exponentially scaled modified Bessel function of the first kind
   """

    @staticmethod
    def forward(ctx, k):
        """
       In the forward pass we receive a Tensor containing the input and return
       a Tensor containing the output. ctx is a context object that can be used
       to stash information for backward computation. You can cache arbitrary
       objects for use in the backward pass using the ctx.save_for_backward method.
       """
        m = 300
        ctx.save_for_backward(k)
        k = k.double()
        answer = (m / 2 - 1) * torch.log(k) - torch.log(sc.ive(m / 2 - 1, k)).cuda() - k - (m / 2) * np.log(
            2 * np.pi)
        answer = answer.float()
        return answer

    @staticmethod
    def backward(ctx, grad_output):
        """
       In the backward pass we receive a Tensor containing the gradient of the loss
       with respect to the output, and we need to compute the gradient of the loss
       with respect to the input.
       """
        k, = ctx.saved_tensors
        m = 300
        # x = -ratio(m/2, k)
        k = k.double()
        x = -((scipy.special.ive(m / 2, k)) / (scipy.special.ive(m / 2 - 1, k))).cuda()
        x = x.float()
        return grad_output * Variable(x)


def bessel(x, k=2):
    """Finish: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0.html"""

    nom = torch.pow(torch.pow(x, 2)/4.0, k)
    denom = math.pow(math.factorial(k), 2)
    out = torch.sum(nom/denom)
    return out


class VonMisesLoss(nn.Module):
    """
    VonMisesLoss based on scipy implementation
    """

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = Variable(torch.FloatTensor(class_weights).cuda())

    def _rvs(self, kappa):
        return self._random_state.vonmises(0.0, kappa, size=self._size)

    def _pdf(self, x, kappa):
        # vonmises.pdf(x, \kappa) = exp(\kappa * cos(x)) / (2*pi*I[0](\kappa))
        return torch.exp(kappa * torch.cos(x)) / (2*math.pi*sc.i0(kappa))

    def _cdf(self, x, kappa):
        """commented out because of _stats not available"""
        # return _stats.von_mises_cdf(kappa, x)

    def _stats_skip(self, kappa):
        return 0, None, 0, None

    def _entropy(self, kappa):
        return (-kappa * sc.i1(kappa) / sc.i0(kappa) +
                torch.log(2 * math.pi * sc.i0(kappa)))
