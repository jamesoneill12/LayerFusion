""" Distances metrics based on the covariance matrix (mostly in the context of merging and compress)"""
import torch
import numpy as np
import torch.nn.functional as F
np.random.seed(0)


def cov(m, y=None):
    """computes covariance of m"""
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def cov_norm(m, y):
    """computes similarity of x, y covariance matrices"""
    m = (m - m.mean(dim=0)) / m.std(dim=0)
    y = (y - y.mean(dim=0)) / y.std(dim=0)
    # print(m.size())
    # print(y.size())
    m = cov(m)
    y = cov(y)
    return torch.norm(m) - torch.norm(y)


def get_svd(m, y):
    m = (m - m.mean(dim=0)) / m.std(dim=0)
    y = (y - y.mean(dim=0)) / y.std(dim=0)
    u1, s1, v1 = torch.svd(m)
    u2, s2, v2 = torch.svd(y)
    return s1, s2


def cov_eig(m, y, k=None):
    """computes similarity of x, y covariance matrices"""
    s1, s2 = get_svd(m, y)
    d = (s1 - s2) if k is None else (s1[:k] - s2[:k])
    d = d.sum().abs()
    return d


def cov_eig_kl(m, y, k=None):
    """computes similarity of x, y covariance matrices"""
    s1, s2 = get_svd(m, y)
    if k is not None: s1, s2 = s1[:k] - s2[:k]
    d = F.kl_div(F.softmax(s1) - F.softmax(s2))
    return d


def cov_kl(m, y, k=None):
    """computes similarity of x, y covariance matrices"""
    m_p = F.softmax(m.flatten())
    y_p = F.softmax(y.flatten())
    d = F.kl_div(m_p, y_p)
    return d


if __name__ == "__main__":

    x = torch.randn((100, 20))
    y = torch.randn((100, 50))
    print(cov_norm(x, y))


