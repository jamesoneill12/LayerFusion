# Author: Mathieu Blondel
# License: Simplified BSD

"""
PyTorch implementation of

Learning Classifiers with Fenchel-Young Losses:
    Generalized Entropies, Margins, and Algorithms.
Mathieu Blondel, André F. T. Martins, Vlad Niculae.
https://arxiv.org/abs/1805.09717
"""


import torch


class ConjugateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, grad, Omega):
        ctx.save_for_backward(grad)
        return torch.sum(theta * grad, dim=1) - Omega(grad)

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output.view(-1, 1), None, None


class FYLoss(torch.nn.Module):

    def __init__(self, weights="average"):
        self.weights = weights
        super(FYLoss, self).__init__()

    def forward(self, theta, y_true):
        self.y_pred = self.predict(theta)
        ret = ConjugateFunction.apply(theta, self.y_pred, self.Omega)

        if len(y_true.shape) == 2:
            # y_true contains label proportions
            ret += self.Omega(y_true)
            ret -= torch.sum(y_true * theta, dim=1)

        elif len(y_true.shape) == 1:
            # y_true contains label integers (0, ..., n_classes-1)

            if y_true.dtype != torch.long:
                raise ValueError("y_true should contains long integers.")

            all_rows = torch.arange(y_true.shape[0])
            ret -= theta[all_rows, y_true]

        else:
            raise ValueError("Invalid shape for y_true.")

        if self.weights == "average":
            return torch.mean(ret)
        else:
            return torch.sum(ret)


class SquaredLoss(FYLoss):

    def Omega(self, mu):
        return 0.5 * torch.sum((mu ** 2), dim=1)

    def predict(self, theta):
        return theta


class PerceptronLoss(FYLoss):

    def predict(self, theta):
        ret = torch.zeros_like(theta)
        all_rows = torch.arange(theta.shape[0])
        ret[all_rows, torch.argmax(theta, dim=1)] = 1
        return ret

    def Omega(self, theta):
        return 0


def Shannon_negentropy(p, dim):
    tmp = torch.zeros_like(p)
    mask = p > 0
    tmp[mask] = p[mask] * torch.log(p[mask])
    return torch.sum(tmp, dim)


class LogisticLoss(FYLoss):

    def predict(self, theta):
        return torch.nn.Softmax(dim=1)(theta)

    def Omega(self, p):
        return Shannon_negentropy(p, dim=1)


class Logistic_OVA_Loss(FYLoss):

    def predict(self, theta):
        return torch.nn.Sigmoid()(theta)

    def Omega(self, p):
        return Shannon_negentropy(p, dim=1) + Shannon_negentropy(1 - p, dim=1)

