import torch.nn as nn
import torch


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, nlayers=2):
        super(EncoderRNN, self).__init__()

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=nlayers)

    def forward(self, input, hidden):
        input = self.embedding(input)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, bsz):
        #weight = next(self.parameters())
        #return weight.new_zeros(self.nlayers, bsz, self.hidden_size)
        #return Variable(torch.randn(self.nlayers, bsz, self.hidden_size, device='cuda'), requires_grad=True)
        return torch.zeros(self.nlayers, bsz, self.hidden_size, device='cuda')


"""
# use this one when not doing multi-task learning as a baseline
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=2):
        super(EncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, nlayers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, bsz):
        return torch.zeros(self.nlayers, bsz, self.hidden_size, device='gpu')
"""
