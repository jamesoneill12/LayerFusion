import torch
from torch import nn


class SparseAutoencoder(nn.Module):

    def __init__(self, x_dim, emb_dim, exp_decay=False):
        super(SparseAutoencoder, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(x_dim, emb_dim))
        self.b1 = nn.Parameter(torch.FloatTensor(emb_dim))
        self.w2 = nn.Parameter(torch.FloatTensor(emb_dim, x_dim))
        self.b2 = nn.Parameter(torch.FloatTensor(x_dim))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        z1 = torch.spmm(x, self.w1) + self.b1
        a1 = self.tanh(z1)
        z2 = torch.mm(a1, self.w2) + self.b2
        return z2

    def get_embedding(self, x):
        x = self.tanh(torch.spmm(x, self.w1) + self.b1)
        return x