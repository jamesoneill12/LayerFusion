from torch import nn
import torch
import numpy as np
import scipy as sp

class InvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs @ self.W, torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)



class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)