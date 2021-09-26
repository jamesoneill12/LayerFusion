"""Hierarhical Softmax"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from models.helpers import get_n_params
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import matplotlib.pyplot as plt


"""
# layer_bottom_probs = self.latent_mixture_labels(inputs, self.layer_bottom_W,
# self.layer_bottom_b, labels = label_position_bottom, top=False)

# bottom_logit_in = torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top])
# layer_bottom_logits = torch.squeeze(bottom_logit_in, dim=1) + self.layer_bottom_b[label_position_top]
# layer_bottom_probs = self.softmax(layer_bottom_logits)

# print(inputs.size()) 700 x 50
# print(self.layer_bottom_W[label_position_top].size()) 70 x 50 x 10

# print(x.size())
# print(labels.size()) # should be [700]
# print(w[0].size(), b[0].size()) # (10, 50, 10) - (10, 10) -> should be (700, 50, 10)
# print(w[1].size(), b[1].size())

bottom_logit_in = torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top])
layer_bottom_logits = torch.squeeze(bottom_logit_in, dim=1) + self.layer_bottom_b[label_position_top]
layer_bottom_probs = self.softmax(layer_bottom_logits)


# print("0 {}".format(torch.matmul(torch.unsqueeze(x, dim=1), w[0][labels]).size()))
# print("1 {}".format(b[0][labels].size()))
# print("2 {} ".format(torch.squeeze(torch.matmul(torch.unsqueeze(x, dim=1), w[0][labels]) + b[0][labels], dim=0).size()))
# print("3 {} ".format((torch.matmul(torch.unsqueeze(x, dim=1), w[0][labels]) + torch.unsqueeze(b[0][labels], dim=1)).size()))

"""


class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None, rs=False,
                 cluster=None, mix_num=None, tune=False, temp=2.5):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        if cluster is not None:
            # should be able to reorganize tree based on order
            # order can be given by unigram dist, or clustering alg.
            pass

        self.ntokens_per_class = ntokens_per_class
        self.mix_num = mix_num
        self.tune = tune
        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class
        self.rs = rs
        self.temp = temp

        if mix_num is not None:
            if rs:
                self.softmax = RelaxedOneHotCategorical
            else:
                self.softmax = nn.Softmax(dim=2)
            self.p = nn.Parameter(torch.ones(mix_num)) if tune else None
            # self.softmax = SoftmaxMixture(nhid, ntokens, mix_num = mix_num, tune=tune)
            ltop_w = [nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True) for _ in range(mix_num)]
            self.layer_top_W = nn.ParameterList(ltop_w)
            ltop_b = [nn.Parameter(
                torch.FloatTensor(self.nclasses), requires_grad=True) for _ in range(mix_num)]
            self.layer_top_b = nn.ParameterList(ltop_b)
            lbottom_w = [nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid,
                self.ntokens_per_class), requires_grad=True) for _ in range(mix_num)]
            self.layer_bottom_W = nn.ParameterList(lbottom_w)
            lbottom_b = [nn.Parameter(torch.FloatTensor(self.nclasses,
                self.ntokens_per_class), requires_grad=True) for _ in range(mix_num)]
            self.layer_bottom_b = nn.ParameterList(lbottom_b)
        else:
            if rs:
                self.softmax = RelaxedOneHotCategorical
            else:
                self.softmax = nn.Softmax(dim=1)
            self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
            self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)
            self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
            self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.mix_num is not None:
            for i in range(self.mix_num):
                self.layer_top_W[i].data.uniform_(-initrange, initrange)
                self.layer_top_b[i].data.fill_(0)
                self.layer_bottom_W[i].data.uniform_(-initrange, initrange)
                self.layer_bottom_b[i].data.fill_(0)
        else:
            self.layer_top_W.data.uniform_(-initrange, initrange)
            self.layer_top_b.data.fill_(0)
            self.layer_bottom_W.data.uniform_(-initrange, initrange)
            self.layer_bottom_b.data.fill_(0)

    def forward(self, inputs, labels = None):

        batch_size, d = inputs.size()

        if labels is not None:

            label_position_top = labels / self.ntokens_per_class
            label_position_bottom = labels % self.ntokens_per_class
            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            if self.rs:
                layer_top_probs = self.softmax(self.temp, layer_top_logits).sample()
            else:
                layer_top_probs = self.softmax(layer_top_logits)
            bottom_logit_in = torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top])
            layer_bottom_logits = torch.squeeze(bottom_logit_in, dim=1) + self.layer_bottom_b[label_position_top]

            if self.rs:
                layer_bottom_probs = self.softmax(self.temp, layer_bottom_logits).sample()
            else:
                layer_bottom_probs = self.softmax(layer_bottom_logits)

            target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * \
                           layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
            return target_probs
        else:
            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            for j in range(self.ntokens_per_class):
                if self.rs:
                    layer_top_probs = self.softmax(self.temp, layer_top_logits[:, j].unsqueeze(1))
                    layer_top_probs = layer_top_probs.sample()
                else:
                    layer_top_probs = self.softmax(layer_top_logits[:, j].unsqueeze(1))

                bottom_logits = torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0]

                if self.rs:
                    word_probs = self.softmax(self.temp, bottom_logits).sample() * layer_top_probs
                else:
                    word_probs = self.softmax(bottom_logits) * layer_top_probs  # [:, 0]

                for i in range(1, self.nclasses):
                    word_probs = torch.cat((word_probs, layer_top_probs * self.softmax(
                        torch.matmul(inputs, self.layer_bottom_W[i]) + self.layer_bottom_b[i])), dim=1)
            # print(word_probs.sum(1))
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

            if self.rs:
                layer_top_probs = self.softmax(self.temp, layer_top_logits[:, j].unsqueeze(1)).sample()
            else:
                layer_top_probs = self.softmax(layer_top_logits[:, j].unsqueeze(1))

            bottom_logits = torch.matmul(inputs, self.layer_bottom_W[int(word_ind[0])]) + self.layer_bottom_b[int(word_ind[0])]
            if self.rs:
                word_probs = self.softmax(self.temp, bottom_logits).sample() * layer_top_probs  # [:, 0]
            else:
                word_probs = self.softmax(bottom_logits) * layer_top_probs  # [:, 0]
            for i in word_ind:
                word_probs = torch.cat((word_probs, layer_top_probs * self.softmax(
                    torch.matmul(inputs, self.layer_bottom_W[int(i)]) + self.layer_bottom_b[int(i)])), dim=1)
        return word_probs, word_ind

    def forward_learn2sample(self, inputs):
        """Doesn't compute the whole probability distribution at test time,
        but instead carries out an approximate search
        """


class HierarchicalSoftmaxMixture(HierarchicalSoftmax):

    def __init__(self, ntokens, nhid, ntokens_per_class=None,
                 cluster=None, mix_num=None, tune=False):
        super(HierarchicalSoftmaxMixture, self).__init__(ntokens,
             nhid, ntokens_per_class, cluster, mix_num, tune)

    def latent_mixture_labels(self, x, w, b, labels, top=True):
        """In the case where labels are provided """
        if top:
            x = torch.cat([torch.matmul(x, w[i]) + b[i] for i in range(self.mix_num)], 1)
            x = x.view(x.size(0), int(x.size(1)/self.mix_num), self.mix_num)
            if self.p is not None:
                x = x * F.softmax(self.p)
                x = x.sum(dim=2)
            else:
                x = torch.mean(x, 2)
        else:
            x = torch.cat([torch.matmul(torch.unsqueeze(x, dim=1), w[0][labels]) + torch.unsqueeze(b[0][labels], dim=1)
                           for i in range(self.mix_num)], 2)
            # output should (700, 1, 50)
            x = x.view(x.size(0), x.size(1), int(x.size(2)/self.mix_num), self.mix_num)
            if self.p is not None:
                x = x * F.softmax(self.p)
                x = x.sum(dim=3)
            else:
                x = torch.mean(x, 3)
            x = x.squeeze()
        return x

    def latent_mixture(self, inputs):
        for k in range(self.mix_num):
            layer_top_logits = torch.matmul(inputs, self.layer_top_W[k]) + self.layer_top_b[k]
            for j in range(self.ntokens_per_class):
                layer_top_probs = F.softmax(layer_top_logits[:, j].unsqueeze(1))
                bottom_logits = torch.matmul(inputs, self.layer_bottom_W[k][0]) + self.layer_bottom_b[k][0]
                word_probs = F.softmax(bottom_logits) * layer_top_probs  # [:, 0]
                for i in range(1, self.nclasses):
                    word_probs = torch.cat((word_probs, layer_top_probs * F.softmax(
                        torch.matmul(inputs, self.layer_bottom_W[k][i]) + self.layer_bottom_b[k][i])), dim=1)
            if k == 0:
                ens_word_probs = word_probs.unsqueeze(2)
            else:
                ens_word_probs = torch.cat((ens_word_probs, word_probs.unsqueeze(2)), dim=2)

        if self.p is not None:
            ens_word_probs = (ens_word_probs * F.softmax(self.p)).sum(dim=2)
        else:
            ens_word_probs = torch.mean(ens_word_probs, dim=2)
        return ens_word_probs

    def forward(self, inputs, labels=None):

        batch_size, d = inputs.size()
        if labels is not None:
            label_position_top = labels / self.ntokens_per_class
            label_position_bottom = labels % self.ntokens_per_class
            layer_top_probs = self.latent_mixture_labels(inputs, self.layer_top_W,
                  self.layer_top_b, labels = label_position_bottom, top=True)
            layer_bottom_probs = self.latent_mixture_labels(inputs, self.layer_bottom_W,
                  self.layer_bottom_b, labels = label_position_bottom, top=False)
            target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * \
                           layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
            return target_probs
        else:
            word_probs = self.latent_mixture(inputs)
            return word_probs


if __name__ == "__main__":

    ntoken, nhid, ntokens_per_class, bsize, seq_len = 50000, 400, 1, 20, 35
    # PTB 10 per class: 4411000 FULL: 8020000
    # WIKI2 10 per class: 22055000 FULL:40100000
    output = torch.randn((seq_len, bsize, nhid))
    targets = torch.LongTensor(torch.randn(seq_len * bsize).size()).random_(0, ntoken)
    mix_num = None
    test_relaxed = True

    if mix_num is None:
        hs = HierarchicalSoftmax(ntoken, nhid, ntokens_per_class)
        # out = hs.forward_sample(output.view(-1, output.size(2)))
        out = hs(output.view(-1, output.size(2)), targets)
    else:
        hs = HierarchicalSoftmax(ntoken, nhid, ntokens_per_class,
                                 mix_num=mix_num, tune=True)
        out = hs.forward_mixture(output.view(-1, output.size(2)) , targets)

    print("There are {} number of parameters".format(get_n_params(hs)))

    if test_relaxed:
        softmax = torch.nn.Softmax()
        targets = torch.randn(seq_len * bsize)
        targets = softmax(targets)
        t = RelaxedOneHotCategorical(2, targets)

        for i in range(1):
            plt.plot(t.sample().cpu().numpy())
        plt.plot(targets, 'r')
        plt.show()

