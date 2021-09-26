import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, x_dim, emb_dim, exp_decay=False):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, x_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x

    def get_embedding(self, x):
        x = self.tanh(self.fc1(x))
        return x

    def forward_context(self, x, context):
        k = len(context)
        x = self.forward(x)
        context = [self.forward(cword) for cword in context]
        context = torch.exp(-k) * context
        return x, context



"""
 Performs a bilinear transformation of input pairs which have different size y = x1*A*x2 + b
 Shape:
    - x1 - N x d1
    - x2 - N x d2
    - A - (d1, d2, feature_size) 
"""


class SubMatrixAE(nn.Module):
    def __init__(self, xdim1, xdim2, emb_dim, exp_decay=False, bias=True, train_scheme='individual'):
        super(SubMatrixAE, self).__init__()

        self.f1 = nn.Biliinear(xdim1, xdim2, emb_dim)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2):
        x = self.tanh(self.f1(x1, x2))
        return (x)



