import torch

def sif(pred_vecs, targ_vectors, targ_inds, freqs, k=1):
    """
    smooth inverse frequency (sif)
    we get the smooth inverse frequency average of target embeddings
    and compare its distance to the sif average predicted embeddings
    :param pred_inds: predicted decoder vector (2d)
    :param targ_inds: corresponding target indices (1d)
    :param freqs: freqs corresponding to each word ind
    :return: soft inverse frequency
    """
    soft = torch.nn.Softmax()
    pred_inds, pred_val = torch.topk(pred_vecs, k)
    norm_pred_sif = (soft(freqs[targ_inds], 1) * pred_vecs) .sum(1)
    norm_pred_sif = soft(freqs[targ_inds], 1)