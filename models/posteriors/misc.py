from loaders import data
import torch
from torch import nn

class Rand_Idxed_Corpus(object):
    # Corpus using word rank as index
    def __init__(self, corpus, word_rank):
        self.dictionary = self.convert_dictionary(dictionary = corpus.dictionary, word_rank = word_rank)
        self.train = self.convert_tokens(tokens = corpus.train, word_rank = word_rank)
        self.valid = self.convert_tokens(tokens = corpus.valid, word_rank = word_rank)
        self.test = self.convert_tokens(tokens = corpus.test, word_rank = word_rank)

    def convert_tokens(self, tokens, word_rank):
        rank_tokens = torch.LongTensor(len(tokens))
        for i in range(len(tokens)):
            rank_tokens[i] = int(word_rank[tokens[i]])
        return rank_tokens

    def convert_dictionary(self, dictionary, word_rank):
        rank_dictionary = data.Dictionary()
        rank_dictionary.idx2word = [''] * len(dictionary.idx2word)
        for idx, word in enumerate(dictionary.idx2word):

            rank = word_rank[idx]
            rank_dictionary.idx2word[rank] = word
            if word not in rank_dictionary.word2idx:
                rank_dictionary.word2idx[word] = rank
        return rank_dictionary



class Word2VecEncoder(nn.Module):

    def __init__(self, ntoken, ninp, dropout):
        super(Word2VecEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):

        emb = self.encoder(input)
        emb = self.drop(emb)
        return emb

class LinearDecoder(nn.Module):
    def __init__(self, nhid, ntoken):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        decoded = self.decoder(inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2)))
        return decoded.view(inputs.size(0), inputs.size(1), decoded.size(1))