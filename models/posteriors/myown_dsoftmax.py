"""

Differentiable Softmax (not sure why differentiable) assigns weights proportional to term frequency

Don't bother working on this until sparse linear layers are in pytorch, otherwise it is slow and defeats
the purpose.

"""

import torch
from torch import nn
from models.posteriors.helpers import construct_weights, partition_idx
import numpy as np
from models.posteriors.helpers import counts2partition


class DifferentiableSoftmax(nn.Module):

    def __init__(self, ntokens, nhid, depth=2, unigram_dist=False,
                 partitions=10, ntokens_per_class=None):
        super(DifferentiableSoftmax, self).__init__()

        if unigram_dist:
            self.weight_groups, part_arrays = construct_weights(uni_dist, partitions, nhid)
            out_sizes = [len(p_array) for p_array in part_arrays]
            """assigns word indices to index of weight group list 
                so that words that correspond to """
            self.partidx = partition_idx(part_arrays)
            self.idx2wg_idx = 1

    def forward(self, input_idx, decoder):
        """Need to pass input_idx as well so we can index the
        sparse linear layer (i.e differentiated softmax)"""

        # decoder = 35 x 20 x 200
        print(self.partidx.size())
        print(input_idx.size())
        partidx = self.partidx[input_idx]
        # partidx = 35 x 20 containing partition idx

        self.weights_groups[partidx]

        # original softmax linear layer: 200 x 1000 (vocab size)
        # dif version: embedding matrix-> 10 (parition num) x 200 x 1000


if __name__ == "__main__":
    # get_partition_idx(ntokens, partitions, tau)

    partitions = 10
    nhid = 300
    depth = 2
    tau = 0.6
    seq_len, bsize = 35, 20
    in_shape = (seq_len, bsize, nhid)
    uni_dist = [10000] * 1000 + [400] * 1000 + [100] * 500 + [40] * 300 + [20] * 500 + [10] * 1000
    uni_dist = np.array(uni_dist)
    ntokens = len(uni_dist)

    parts = counts2partition(uni_dist, partitions)

    decoder = torch.randn(in_shape)
    input_idx = torch.LongTensor(torch.randn(seq_len, bsize).size()).random_(0, ntokens)

    ds = DifferentiableSoftmax(ntokens, nhid, depth=depth, unigram_dist=True, partitions=10)
    out = ds(input_idx, decoder)

    # for part in parts:
    #    print(len(part))