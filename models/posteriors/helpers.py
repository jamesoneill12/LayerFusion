import torch
import numpy as np
from torch import nn


def get_partition_idx(ntokens, partitions, tau=0.6):
    """paritions based on pure heuristic tau=0.6"""
    partition_idx = ([int(ntokens * np.exp(-i * tau)) for i in range(partitions)])[::-1]
    return partition_idx


flatten = lambda l: [item for sublist in l for item in sublist]


def partition_idx(partitions):
    intervals = flatten([[i] * len(partition) for i, partition in enumerate(partitions)])
    return torch.LongTensor(intervals)


def counts2partition(uni_dist, paritions, tau=0.6):
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
        if isinstance(uni_dist, torch.Tensor): uni_dist = uni_dist.numpy()
        uni_partitions = np.array_split(uni_dist, partition_idx)
        if isinstance(uni_dist, torch.Tensor):
            uni_partitions = torch.from_numpy(uni_partitions)
    return uni_partitions


def construct_weights(uni_dist, partitions, nhid):
    partitioned_arrays = counts2partition(uni_dist, partitions)
    weight_groups = []
    for i in range(len(partitioned_arrays)):
        weight_groups.append(nn.Linear(nhid, len(partitioned_arrays[i])))
    weight_groups = nn.ModuleList(weight_groups)
    return weight_groups, partitioned_arrays


def get_sparse_weights(ntokens, part_lens, nhid):
    """not really sparse linear layer, but close enough"""
    theta = nn.Linear(nhid, ntokens)
    for plen in part_lens:
        theta[:, plen].grad = 0
    return theta



