import torch.nn as nn
import torch
from models.networks.convolutional.attention import Attention
import torch.nn.functional as F
from util.batchers import check_nan


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, nlayers,
                 dropout_p=0.1, max_length=40, attention=False):
        super(DecoderRNN, self).__init__()

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.att = attention

        # I think output size is the size of dictionary for target vocab.
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=nlayers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        if self.att:
            self.attention = Attention(self.hidden_size, max_length)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        output = F.dropout(embedded, p=0.2)
        if self.att:
            output, attn_weights = self.attention(output, hidden, encoder_outputs)
        output = F.tanh(output)
        output, hidden = self.gru(output, hidden)
        # output contains all outputs
        # so why is output[0] used ?
        # this seems like it would just return the first output from the decoder
        output = self.out(output.view(output.size(0)*output.size(1), output.size(2)))
        # output = F.log_softmax(self.out(output[0]), dim=1)
        if self.att:
            return output, hidden, attn_weights
        else:
            return output, hidden

    # less certain about the looping here, make sure its correct.
    def forward_ss(self, input, hidden, encoder_outputs, inds=None):
        # enc_check = torch.isnan(input.cpu()).type(torch.ByteTensor).any()
        # So problem is here when the embedding matrix is looked up,
        # must make sure that the correct indicies are lookup up and that emb_matrix
        # is the correct size.
        emb = self.embedding(input)
        emb = F.dropout(emb, 0.2)
        assert check_nan(emb).item() == 0
        if self.att:
            output, attn_weights = self.attention(emb, hidden, encoder_outputs)
        else:
            output = emb
        output = F.tanh(output)

        if inds is None:
            inds = []
        outputs = []
        for i in range(emb.size(0)):
            if i in inds and i != 0:
                output, hidden = self.gru(output, hidden)
                inds = list(filter(lambda a: a != i, inds))
            else:
                x = emb[i, :, :].unsqueeze(0)
                output, hidden = self.gru(x, hidden)
            outputs.append(output)

        assert inds == [] or inds == [0]
        output = torch.cat(outputs, 0)
        # output = F.dropout(output)

        # need to understand what the deal is with output[0] here,
        # otherwise i should pass output
        output = self.out(output.view(output.size(0)*output.size(1), output.size(2)))
        #output = F.log_softmax(self.out(output[0]), dim=1)
        output = F.log_softmax(output, dim=1)
        if self.att:
            return output, hidden, attn_weights
        else:
            return output, hidden

    def initHidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.hidden_size)
