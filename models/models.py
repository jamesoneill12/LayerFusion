"""
This is the original seq2seqq model that
 for some reason is not updating the gradients of the encoder
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from util.batchers import repackage_hidden
from models.networks.recurrent.decoder import DecoderRNN
from models.networks.recurrent.encoder import EncoderRNN


class Seq2Seq(nn.Module):
    def __init__(self, svocab_size, hidden_size, tvocab_size, nlayers=2, attention=False):
        super(Seq2Seq, self).__init__()
        self.attention = attention
        self.hidden_size = hidden_size
        self.svocab_size = svocab_size
        self.nlayers = nlayers

        self.encoder = EncoderRNN(svocab_size, hidden_size, nlayers=nlayers)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.decoder = DecoderRNN(tvocab_size, hidden_size, nlayers=nlayers,  attention=attention)

    # (i think) only reason to return decoder and encoder hidden is to zero out the gradients
    def forward(self, source_input, target_input, source_hidden, target_hidden):
        encoder_output, encoder_hidden = self.encoder(source_input, source_hidden)
        decoder_output, decoder_hidden = self.decoder(target_input, target_hidden, encoder_hidden[-1])
        return decoder_output, encoder_hidden, decoder_hidden

    def forward_ss(self, source_input, target_input, source_hidden, target_hidden, inds):
        encoder_output, encoder_hidden = self.encoder(source_input, source_hidden)
        encoder_output = torch.tanh(self.linear(encoder_output))

        if self.attention:
            decoder_output, decoder_hidden, attn_weights =\
                self.decoder.forward_ss(target_input, target_hidden, encoder_output, inds)
            return decoder_output, encoder_hidden, decoder_hidden, attn_weights
        else:
            decoder_output, decoder_hidden =\
                self.decoder.forward_ss(target_input, target_hidden, encoder_output, inds)

            return decoder_output, encoder_hidden, decoder_hidden

    def init_hidden(self, bsz):
        enc_hidden = self.encoder.initHidden(bsz)
        dec_hidden = self.decoder.initHidden(bsz)
        return enc_hidden, dec_hidden


if __name__ == "__main__":

    """
    Seems like its supposed to be None at the start because it is with RNNModel so everything is ok here.
    Just need to figure out why I cannot use SGD then in mt_train because its fine here.
    """
    lr = 10
    ntoken = 100
    hidden_size = 20
    sent_len = 10
    batch_size = 10
    loss = nn.CrossEntropyLoss()
    enc = EncoderRNN(vocab_size=ntoken, hidden_size=hidden_size, nlayers=2)

    for name, p in enc.named_parameters():
        print(name)
        print(p.grad is not None)

    #enc = RNNModel(rnn_type='LSTM', ntoken=ntoken,
    #               ninp=300, nhid=300, nlayers=2, pretrained=None)
    #enc.initHidden(20)
    optimizer = optim.SGD(enc.parameters(), lr=lr)
    hidden = enc.initHidden(batch_size).cpu()
    x = Variable(torch.randint(0, ntoken - 1, (sent_len, batch_size)).type(torch.LongTensor))
    # change hidden_size here to ntoken when testing RNNModel()
    target = Variable(torch.randint(0, hidden_size - 1, (ntoken,1)).type(torch.LongTensor)).squeeze()

    hidden = repackage_hidden(hidden)
    enc.zero_grad()

    pred, hidden = enc(x, hidden)
    print(pred.view(pred.size(0)*pred.size(1), pred.size(2)).size())
    print(target.size())
    loss = loss(pred.view(pred.size(0)*pred.size(1), pred.size(2)), target)
    loss.backward()
    optimizer.step()

    print(loss)

    for name, p in enc.named_parameters():
        print(name)
        print(p.grad is not None)
        p.data.add_(-lr, p.grad.data)
