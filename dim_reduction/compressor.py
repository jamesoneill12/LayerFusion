from scipy import linalg
import torch
from torch.autograd import Variable
import numpy as np


def sp_svd(p, perc):
    U, s, Vh = linalg.svd(p)
    S = np.diag(s)
    k = int(len(U) * perc) # keep top k eigenvalues
    print(U.shape, S.shape, Vh.shape)
    svd_p = np.dot(U[k], np.dot(S, Vh))
    svd_p = torch.from_numpy(svd_p)
    return svd_p


def svd_compress(model, perc = 0.5):
    """percent: 0.5 means reduce dimensionality to 50%"""
    sizes = {}
    for p_name, p in model.named_parameters():
        if 'weight' in p_name.lower() \
                and 'layernorm' not in p_name.lower() \
                and len(p.data.size()) != 1 :
            p = p.detach().cpu().numpy()
            # print(p_name)
            print(p_name, p.shape)
            sizes[p_name] = [p.shape]
            U, s, V = randomized_svd(p, perc=perc, cuda=False)
            #svd_p = U @ s[]
            sizes[p_name].append(U.size())
            # model.state_dict()[p_name].data.copy_(U)
            model.state_dict()[p_name].data = U
    for name, sizes in sizes.items():
        if 'intermediate' in name:
            print("{} param: {} -> {}".format(name, sizes[0], sizes[1]))
    return model.cuda()


def pca_compress(model, perc=0.5):

    """percent: 0.5 means reduce dimensionality to 50%"""
    sizes = {}
    for p_name, p in model.named_parameters():
        p_name = p_name.lower()
        if 'weight' in p_name and p_name not in ['layernorm', 'intermediate']:
            p = p.detach().cpu()
            m, n = p.size()
            k = int(n * perc)
            sizes[p_name] = [p.shape]
            p_mean = torch.mean(p, 0)
            p = p - p_mean.expand_as(p)
            U, S, V = torch.svd(torch.t(p))
            z = torch.mm(p, U[:, :k])
            # f = z * V[:, :k])
            pca_X = Variable(z.cuda(), requires_grad=True)
            sizes[p_name].append(pca_X.size())
            # model.state_dict()[p_name].data.copy_(U)
            model.state_dict()[p_name].data = pca_X
    for name, sizes in sizes.items():
        # if 'intermediate' in name:
        print("{} param: {} -> {}".format(name, sizes[0], sizes[1]))
    return model


def randomized_svd(M, k=10, perc=0, cuda=True):

    B = torch.tensor(M)
    if cuda: B = B.cuda(0)

    if len(B.shape) == 1:
        n = B.size(0); m = 1
    else:
        m, n = B.size()

    if perc !=0: k = int(n * perc) # keep top k eigenvalues
    transpose = False

    if m < n:
        transpose = True
        if len(B.shape) == 1:
            B = B.unsqueeze(0)
        B = B.transpose(0, 1)
        if cuda: B = B.cuda(0)
        m, n = B.size()

    try:
        rand_matrix = torch.rand((n, k), dtype=torch.float).cuda(0)  # short side by k
        Q, _ = torch.qr(B @ rand_matrix)  # long side by k
        Q.cuda(0)
        smaller_matrix = (Q.transpose(0, 1) @ B).cuda(0)  # k by short side
        U_hat, s, V = torch.svd(smaller_matrix, False)
        U_hat.cuda(0)
        U = (Q @ U_hat)

    except:
        # if it doesn't fit on the GPU, switch to numpy svd
        rand_matrix = torch.rand((n, k), dtype=torch.float).numpy()  # short side by k
        # print(Q.size(), type(Q))
        B = B.numpy()
        Q, _ = linalg.qr(B @ rand_matrix)  # long side by k
        smaller_matrix = (Q.transpose(0, 1) @ B)  # k by short side
        print(type(smaller_matrix))
        U_hat, s, V = linalg.svd(smaller_matrix, full_matrices=False)
        U = (Q @ U_hat)
        U, s, V = torch.from_numpy(U).cuda(), \
                  torch.from_numpy(s).cuda(), torch.from_numpy(V).cuda()

    if transpose:
        return V.transpose(0, 1), s, U.transpose(0, 1)
    else:
        return U, s, V