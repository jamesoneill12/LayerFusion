import torch
import numpy as np


def get_partition_idx(ntokens, partitions, tau = 0.6):
    """paritions based on pure heuristic tau=0.6"""
    partition_idx = ([int(ntokens * np.exp(-i * tau)) for i in range(partitions)])[::-1]
    return partition_idx


def counts2partition(uni_dist, paritions, tau = 0.6):
    """
    unigram_dist should be either a dictionary with idx2freq
    or a index ordered numpy matrix/torch.Tensor
    """
    if isinstance(uni_dist, dict):
        sorted_uni = sorted(uni_dist.items(), key=lambda kv: kv[1])
        parition_idx = get_partition_idx(len(uni_dist), paritions, tau)
        sorted_uni_token_idx = np.array(list(sorted_uni.keys()))
        uni_paritions = np.array_split(sorted_uni_token_idx, parition_idx)

    elif isinstance(uni_dist, np.ndarray) or isinstance(uni_dist, torch.Tensor):
        ntokens = uni_dist.shape[0] if len(uni_dist.shape) > 1 else len(uni_dist)
        partition_idx = get_partition_idx(ntokens, paritions, tau)[:-1]
        if isinstance(uni_dist, torch.Tensor):
            uni_dist = uni_dist.numpy()
        uni_partitions = np.array_split(uni_dist, partition_idx)
        if isinstance(uni_dist, torch.Tensor):
            uni_partitions = torch.from_numpy(uni_partitions)
    return uni_partitions


if __name__ == "__main__":

    ntokens = 100000
    partitions = 10
    tau = 0.6

    # get_partition_idx(ntokens, partitions, tau)
    uni_dist = [10000] * 1000 + [400] * 1000 + [100] * 500 \
               + [40] * 300 + [20] * 500 + [10] * 1000
    uni_dist = np.array(uni_dist)

    parts = counts2partition(uni_dist, partitions)
    for part in parts:
        print(len(part))