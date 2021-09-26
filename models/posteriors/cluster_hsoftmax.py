"""Hierarhical Softmax"""
import torch
from torch import nn
import numpy as np


class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None, cluster=None, tree_depth=2):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid
        self.tree_depth = tree_depth

        self.W = []
        self.b = []
        w_sizes = []
        b_sizes = []

        if tree_depth > 2:
            ub, lb = 0.65, 0.5
            if tree_depth > 3:
                # interval = (ub-lb) /(tree_depth - 2)
                # ntokens_per_class = [int(ntokens) for i in ]
                powers = np.linspace(lb, ub, tree_depth-1)
                ntokens_per_class = [int(ntokens**power) for power in powers]
            else:
                ntokens_per_class = [int(ntokens**0.65), int(ntokens**0.5)]
            self.ntokens_per_class = ntokens_per_class
            self.nclasses = [int(np.ceil(self.ntokens * 1. / tpc)) for tpc in self.ntokens_per_class]
            self.ntokens_actual = [self.nclasses * tpc for tpc in self.ntokens_per_class]
            w_sizes.append((self.nhid, self.nclasses))
            b_sizes.append(self.nclasses)
        else:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))
            self.ntokens_per_class = ntokens_per_class
            self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
            self.ntokens_actual = self.nclasses * self.ntokens_per_class
            w_sizes.append((self.nhid, self.nclasses))
            b_sizes.append(self.nclasses)

        if cluster is not None:
            # should be able to reorganize tree based on order
            # order can be given by unigram dist, or clustering alg.
            pass

        if tree_depth > 2:
            for i in range(1, tree_depth-1):
                w_sizes.append((self.nclasses[i-1], self.nclasses[i]))
                b_sizes.append(self.nclasses[i])

        w_sizes.append((self.nclasses, self.nhid, self.ntokens_per_class))
        b_sizes.append((self.nclasses, self.ntokens_per_class))

        print(w_sizes)
        for i in range(tree_depth):
            self.W.append(nn.Parameter(torch.FloatTensor(w_sizes[i]),
                                       requires_grad=True))
            self.b.append(nn.Parameter(torch.FloatTensor(b_sizes[i]),
                                       requires_grad=True))
        self.W = nn.ModuleList(self.W)
        self.b = nn.ModuleList(self.b)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

        """
        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)
        
        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)
        """

    def init_weights(self):
        initrange = 0.1
        if self.tree_depth >  1:
            for i in range(self.tree_depth):
                self.W[i].data.uniform_(-initrange, initrange)
                self.b[i].data.fill_(0)
        else:
            self.W.data.uniform_(-initrange, initrange)
            self.b.data.fill_(0)

    def forward(self, inputs, labels = None):

        batch_size, d = inputs.size()

        if labels is not None:

            label_position = []
            layer_logits = []
            layer_probs = []
            target_probs = 0.0

            label_position.append(labels / self.ntokens_per_class[0])
            label_position.append(labels % self.ntokens_per_class[0])
            layer_logits.append(torch.matmul(inputs, self.W[0]) + self.b[0])
            layer_probs.append(self.softmax(layer_logits[0]))

            for i in range(self.tree_depth):
                inputs = torch.bmm(torch.unsqueeze(inputs, dim=1), self.W[i])
                layer_logits = torch.squeeze(inputs, dim=1) + self.b[i]
                layer_probs.append(self.softmax(layer_logits[i]))

            for i in range(self.tree_depth):
                target_probs *= layer_probs[i][torch.arange(batch_size).long(), label_position[i]]

            return target_probs

        else:

            layer_logits = []
            word_probs = []
            layer_probs = []

            layer_logits.append(torch.matmul(inputs, self.W[0]) + self.b[0])
            for j in range(self.ntokens_per_class):
                layer_probs.append(self.softmax(layer_logits[0][:, j].unsqueeze(1)))
                for i in range(self.tree_depth):
                    layer_logits.append(torch.matmul(inputs, self.W[i]) + self.b[i])
                    word_probs.append(self.softmax(layer_logits[i]) * layer_probs[i]) # [:, 0]
                    for k in range(1, self.nclasses):
                        word_probs[i] = torch.cat((word_probs[i], layer_probs[i] * self.softmax(
                            torch.matmul(inputs, self.W[i]) + self.b[i])), dim=1)
            return word_probs

    def forward_sample(self, inputs, search_num=None):
        """
        Sample only few of the outputs and choose the argmax of this instead.
        Returns sampled word_inds aswell so that the are retrieve from targets[word_inds]
        """
        if search_num is None:
            search_num = int(np.log2(self.nclasses))

        word_ind = torch.randint(self.nclasses, (search_num,))
        layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
        for j in range(self.ntokens_per_class):
            layer_top_probs = self.softmax(layer_top_logits[:, j].unsqueeze(1))
            bottom_logits = torch.matmul(inputs, self.layer_bottom_W[int(word_ind[0])]) + self.layer_bottom_b[int(word_ind[0])]
            word_probs = self.softmax(bottom_logits) * layer_top_probs  # [:, 0]
            for i in word_ind:
                word_probs = torch.cat((word_probs, layer_top_probs * self.softmax(
                    torch.matmul(inputs, self.layer_bottom_W[int(i)]) + self.layer_bottom_b[int(i)])), dim=1)
        return word_probs, word_ind


if __name__ == "__main__":

    ntoken, nhid, ntokens_per_class, bsize, seq_len, tree_depth = 100, 50, 10, 20, 35, 4
    hs = HierarchicalSoftmax(ntoken, nhid, ntokens_per_class, tree_depth=tree_depth)

    output = torch.randn((seq_len, bsize, nhid))
    targets = torch.LongTensor(torch.randn(seq_len * bsize).size()).random_(0, ntoken)

    out = hs(output.view(-1, output.size(2)))
