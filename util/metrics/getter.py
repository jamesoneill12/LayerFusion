from pyemd import emd, emd_with_flow
from scipy.stats import wasserstein_distance
from util.metrics.covariance import cov_norm, cov_eig, cov_eig_kl, cov_kl
from util.metrics.kernel import alignment
from joblib import Parallel, delayed
import multiprocessing
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import numpy as np
import ot
import torch


def get_sim_measure(metric):
    if metric == 'cov': return cov_norm
    elif metric == 'cov_eig': return cov_eig
    elif metric == 'cov_kl': return cov_kl
    elif metric == 'cov_eig_kl': return cov_eig_kl
    elif metric == 'euclidean' or metric == 'manhattan': return torch.dist
    # centered kernel alignment
    elif metric == 'cka': return alignment
    elif metric == 'cos': return F.cosine_similarity
    # needs dist mat as 3rd arg
    elif metric == 'emd': return emd
    # returns the minimum flow path as well
    # we could use this as a way to visualize how
    # information is propogated through the network
    elif metric == 'emd_flow': return emd_with_flow
    elif metric == 'sinkhorn': return ot.sinkhorn
    elif metric == 'wasserstein': return wasserstein_distance
    elif metric == 'kl': return F.kl_div
    else: print(f"{metric} is not an option, 'cov' has been chosen "); return F.kl_div


def compute_reg(f_param, s_param, met): pass


def to_probs(f_param, s_param):
    f_param, s_param = F.softmax(f_param.flatten(), dim=0), F.softmax(s_param.flatten(), dim=0)
    fp, sp = f_param.detach().cpu().numpy(), s_param.detach().cpu().numpy()
    fp, sp = np.expand_dims(fp.astype('float64'), axis=1), \
             np.expand_dims(sp.astype('float64'), axis=1)
    return fp, sp


def compute_cost_matrix(f_param, s_param, dist_met):
    fp, sp = to_probs(f_param, s_param)
    M = cdist(fp, sp, metric=dist_met)
    return fp, sp, M


def compute_perm_sim(sp, fp, f_param, s_param,
                     met, metric, samp_size):

    s_num = sp.shape[0] if fp.shape[0] > sp.shape[0] else fp.shape[0]
    samp_inds = np.random.choice(s_num, samp_size, replace=False)

    if metric in ['emd', 'emd_flow', 'sinkhorn', 'wasserstein', 'kl', 'cos']:
        # fp, sp, M = compute_cost_matrix(f_param, s_param, dist_met)
        fp, sp = to_probs(f_param, s_param)

    if 'emd' in metric:
        M = ot.dist(fp[samp_inds], sp[samp_inds])
        M /= M.max()
    if metric == 'emd':
        sim = met(fp.squeeze()[samp_inds], sp.squeeze()[samp_inds], M)
    elif metric == 'emd_flow':
        sim = met(f_param, s_param, M)
    elif metric == 'sinkhorn':
        reg = compute_reg(f_param, s_param)
        sim = metric(f_param, s_param, reg)
    elif metric == 'wassserstein':
        sim = met(f_param, s_param)
    elif metric == 'euclidean':
        sim = met(f_param, s_param, 2)
    elif metric == 'manhattan':
        sim = met(f_param, s_param, 1)
    elif metric == 'kl':
        sim = met(f_param, s_param).abs()
    else:
        sim = met(f_param, s_param)

    return sim


def compute_similarity(f_param, s_param, met, metric, dist_met='euclidean',
                       nsamps=10, samp_size=100, par=True):
    """
    desc: expects a 2d tensor for f_param & s_param corresponding to weights
    :param
        nsamps: when using emd (sinkhorn included), it is too expensive to compute
         emd for a whole weight matrix if large. Hence, if f_param or s_param is >
         than some threshold, we sample nsamps number of weight matrix subset of
         size samp_size and average over the results to approximate emd
    """

    #if metric in ['emd', 'emd_flow', 'sinkhorn', 'wasserstein']:
    #    # fp, sp, M = compute_cost_matrix(f_param, s_param, dist_met)
    fp, sp = to_probs(f_param, s_param)
    if par:
        num_cores = multiprocessing.cpu_count()
        try:
            sims = Parallel(n_jobs=num_cores)(delayed(compute_perm_sim)(
            sp, fp, f_param, s_param, met, metric, samp_size) for nsamp in range(nsamps))
        except:
            sims = []
            for nsamp in range(nsamps):
                sim = compute_perm_sim(sp, fp, f_param, s_param, met, metric, samp_size)
                sims.append(sim)
    else:
        sims = []
        for nsamp in range(nsamps):
            sim = compute_perm_sim(sp, fp, f_param, s_param, met, metric, samp_size)
            sims.append(sim)

    sim = sum(sims)/len(sims)
    return sim


if __name__ == "__main__":

    x, y = torch.randn((100, 10)), torch.randn((512, 128))
    compute_similarity(x, y, 'emd', )