"""Computes centered kernel alignment to measure similarity between layers
Refer to this also - https://arxiv.org/pdf/1905.00414.pdf
"""

import numpy as np
import torch


def np_center_kernel(K, copy=True):
    '''
    Centered version of a kernel matrix (corresponding to centering the)
    implicit feature map.
    '''
    means = K.mean(axis=0)
    if copy:
        K = K - means[None, :]
    else:
        K -= means[None, :]
    K -= means[:, None]
    K += means.mean()
    return K


def np_alignment(K1, K2):
    '''
    Returns the kernel alignment
        <K1, K2>_F / (||K1||_F ||K2||_F)
    defined by
        Cristianini, Shawe-Taylor, Elisseeff, and Kandola (2001).
        On Kernel-Target Alignment. NIPS.
    Note that the centered kernel alignment of
        Cortes, Mohri, and Rostamizadeh (2012).
        Algorithms for Learning Kernels Based on Centered Alignment. JMLR 13.
    is just this applied to center_kernel()s.
    '''
    return np.sum(K1 * K2) / np.linalg.norm(K1) / np.linalg.norm(K2)


def center_kernel(X, copy=True):
    '''
    Centered version of a kernel matrix (corresponding to centering the)
    implicit feature map.
    '''
    means = X.mean(axis=0)
    if copy:
        X = X - means[None, :]
    else:
        X -= means[None, :]
    X -= means[:, None]
    X += means.mean()
    return X


def alignment(x, y): return torch.sum(x * y) / torch.norm(x) / torch.norm(y)


def test_alignment(x=None, y=None, nump=False):
    if x is None and y is None:
        x = torch.randn((100, 10))
        y = torch.randn((100, 10))
    y = y.t()
    print(alignment(x, y))
    if nump:
        print(np_alignment(x.numpy(), y.numpy()))


if __name__ == "__main__":

    test_alignment(nump=True)