"""
Distance measures that can be used for various torch.tensor operations
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from scipy.spatial.distance import cosine


def get_predict_token_vector(pred, target, k=10, s=1,  tau=1):
    """
    pred: l_2 normed prediction vector (batch_size*sent_length x dim)
    target: embedding matrix (vocab_size x dim)
    k: choose top k values
    s: number of samples to choose for each output
        (we can average embeddings of multiple samples drawn from top k)
    tau: temperature for controlling softmax kurtosis
    """
    cos_sims = pairwise_distances(pred, target)
    vals, inds = torch.topk(cos_sims, k)
    sample_probs = F.softmax(vals/tau, 1)
    # print(sample_probs)
    # k_cands = target[inds]
    sample_size = torch.Size((sample_probs.size(0), s))
    # should maybe consider averaging over sampled embeddings cand embeddings
    samp_inds = Categorical(sample_probs).sample()
    cand_ind = inds[list(range(sample_probs.size(0))), samp_inds]
    cand = Variable(target[cand_ind], requires_grad=True)
    return cand, cand_ind


# get nearest word in vocab to prediction
def get_nearest_token(pred, vocab, k=1):
    cos_sims = pairwise_distances(pred, vocab)
    vals, inds = torch.topk(cos_sims, k)
    return inds


# cosine dist
def pairwise_distances(x, y, clamp = False):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = (x_norm + y_norm - 2.0 * torch.mm(x, y_t)).flatten()
    if clamp: return torch.clamp(dist, 0.0,  np.inf)
    return dist


def pearson_correlation(x, y):
    num = (x * y).sum(1).view(-1, 1)
    dx = torch.sqrt((x ** 2).sum(1).view(1, -1))
    dy = torch.sqrt((y ** 2).sum(1).view(1, -1))
    pc = num / (dx * dy)
    return torch.clamp(pc, 0.0, 1)


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def csim_np(x, y):
    """Numpy version for calculating cosine_sim for ith row of both x and y"""
    print(x.shape)
    print(y.shape)
    assert x.shape[0] == y.shape[0]
    total = 0.
    for i in range(x.shape[0]):
        # cos_sim(u, v) = 1 - cos_dist(u, v)
        total += 1 - cosine(x[i, :],y[i, :])
    return total/x.shape[0]


def exp_dist(x, y): return torch.clamp(torch.exp(-abs(x - y)), 0.0, 1)

def batthacaryya_dist(output, target): return torch.sum(torch.sqrt(torch.abs(torch.mul(output, target))))


def hamming_distance(pred, target, weight=1):
    """So far, just used for comparing binary codes"""

    if isinstance(pred, torch.Tensor):
        if weight != 1 or weight != None:
            weight = torch.ones(target.size(1)).cuda()
            # print("pred : \t {} \t\t target : {}".format(pred.size(), target.size()))
        return round(float(torch.sum((pred * weight != target * weight))) / pred.numel(), 4)
    elif isinstance(pred, np.ndarray):
        return np.count_nonzero(pred != target) / pred.numel()


if __name__ == "__main__":

    x = torch.randn((10000, 100))
    y = torch.randn((1, 100))

    result = pairwise_distances(x, y).flatten()
    idx = torch.topk(result, result.size(0))[1]
