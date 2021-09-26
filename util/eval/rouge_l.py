"""There is already an impleentation of rouge_l in rouge.py but this is specifically for AC model """
import numpy as np
import torch
from loaders.caption import PAD

def _lcs(x, y):
    n = len(x)
    m = len(y)
    table = dict()

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])

    def recon(i, j):
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return recon(i - 1, j - 1) + [x[i - 1]]
        elif table[i - 1, j] > table[i, j - 1]:
            return recon(i - 1, j)
        else:
            return recon(i, j - 1)

    return len(recon(n, m)), n, m


def rouge_l(evals, refs):
    # if evals.size() != refs.size():
    # print(evals.size())
    # print(refs.size())
    use_cuda = evals.is_cuda

    evals, refs = map(lambda x: x.data.cpu().numpy(), [evals, refs])

    scores = []
    for eva, ref in zip(evals, refs):
        same_len, eva_len, ref_len = map(float,
                                         _lcs(eva, ref[np.where(ref > PAD)]))

        if ref_len != 0:
            # print("{} ref len \t {} eva len".format(ref_len, eva_len))
            r_lcs, p_lcs = same_len / ref_len, same_len / eva_len

        beta = p_lcs / (r_lcs + 1e-12)
        f_lcs = ((1 + (beta**2)) * r_lcs * p_lcs) / \
            (r_lcs + ((beta**2) * p_lcs) + 1e-12)
        scores.append(f_lcs)

    scores = np.asarray(scores, dtype=np.float32)
    scores = torch.autograd.Variable(torch.from_numpy(scores), requires_grad=False)

    if use_cuda:
        scores = scores.cuda()

    return scores


def mask_score(props, words, scores):
    assert words.size() == scores.size()
    mask = (words > 0).float()

    return props * scores * mask