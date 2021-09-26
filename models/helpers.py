import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.networks.recurrent.goru import GORU
from models.networks.recurrent.urnn import EURNN
from models.networks.convolutional.highway import RecurrentHighwayText


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def embedding_torch_matrix(pretrained, tune=False):
    em_words = torch.from_numpy(np.array(list(pretrained.values())))
    embedding = nn.Embedding(em_words.size(0), em_words.size(1))
    embedding.weight.data.copy_(em_words)
    embedding.weight.requires_grad = tune
    return embedding


def get_rnn(rnn_type, ninp, nhid, nlayers, dropout,
            drop_method='standard', batch_first=True,
            bidirectional=False, cuda = True):
    rnn_type = rnn_type.upper()
    if rnn_type in ['LSTM', 'GRU']:
        rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout,
                                    batch_first=batch_first, bidirectional=bidirectional)
    elif rnn_type == 'URNN':
        rnn = EURNN(ninp, nhid, capacity=nlayers, cuda=cuda)
    elif rnn_type == 'GORU':
        rnn = GORU(ninp, nhid, num_layer=nlayers, dropout=dropout,
                   capacity=nlayers,  embedding=False)
    elif rnn_type == 'HIGHWAY':
        rnn = RecurrentHighwayText(ninp, nhid, nlayers, drop_method=drop_method, lm=True,
                                   dropout_rate=dropout, embedding=False, attention=False)
    else:
        try:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        except KeyError:
            raise ValueError("""An invalid option for `--model` was supplied,
                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
    return rnn


def word_count(corpus):
    counter = [0] * len(corpus.dictionary.idx2word)
    for i in corpus.train:
        counter[i] += 1
    for i in corpus.valid:
        counter[i] += 1
    for i in corpus.test:
        counter[i] += 1
    return np.array(counter).astype(int)


def word_freq_ordered(corpus):
    # Given a word_freq_rank, we could find the word_idx of that word in the corpus
    counter = word_count(corpus = corpus)
    # idx_order: freq from large to small (from left to right)
    idx_order = np.argsort(-counter)
    return idx_order.astype(int)


def word_rank_dictionary(corpus):
    # Given a word_idx, we could find the frequency rank (0-N, the
    # smaller the rank, the higher frequency the word) of that word in the corpus
    idx_order = word_freq_ordered(corpus = corpus)
    # Reverse
    rank_dictionary = np.zeros(len(idx_order))
    for rank, word_idx in enumerate(idx_order):
        rank_dictionary[word_idx] = rank
    return rank_dictionary.astype(int)


# default neat-style init
def weight_init_default(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        m.weight.data.uniform_(-1.0, 1.0)


# xavier initialization (apparently better than pytorch default)
def weight_init_xavier(m):
    # print "xavier init."
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = np.prod(size[1:])  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)


# xavier initialization (apparently better than pytorch default)
# now He initialization for conv layers
def weight_init_he(m):
    # print "he init."
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # print "xavierizing..."
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = np.prod(size[1:])  # number of columns
        # variance = np.sqrt(2.0/(fan_in + fan_out))
        variance = np.sqrt(2.0) * np.sqrt(1.0 / fan_in)
        m.weight.data.normal_(0.0, variance)
        # print fan_out,fan_in,variance


def weight_norm(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.utils.weight_norm(m)


# layer norm (control mechanism for dealing with really deep nets)
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1).expand_as(x)
        std = x.std(-1).expand_as(x)

        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)


def linear(x):
    return x


class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.alpha * (F.elu(-1 * F.relu(-1 * x)))
        return temp1 + temp2


class silu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)
