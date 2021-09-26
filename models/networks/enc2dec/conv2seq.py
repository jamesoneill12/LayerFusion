"""Conv2Seq is not to be confused img2seq, img2seq strictly deals with img on encoder"""
from models.networks.enc2dec.seq2seq import Decoder
from models.networks.convolutional.convolution import CNN
import torch.nn as nn


class Conv2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi):
        super(Conv2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword
        self.encoder = CNN(src_nword, embed_dim, hidden_dim, s2s=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim, max_len, trg_soi)

    def forward(self, source, src_length=None, target=None):
        enc_h = self.encoder(source)  # B x S x 2*H / 2 x B x H

        print(enc_h.size())

        dec_h0 = torch.tanh(self.linear(enc_h))  # B x 1 x 2*H

        out = self.decoder(enc_h, dec_h0, target)  # B x S x H
        out = torch.log_softmax(out.contiguous().view(-1, self.trg_nword), dim=1)
        return out


if __name__ == "__main__":

    from models.networks.enc2dec.test_var import *

    c2s = Conv2Seq(src_nword=src_vocab_size, trg_nword=trg_vocab_size,
                   num_layer=nlayers, embed_dim=emb_size, hidden_dim=hidden_size,
                   max_len=sent_len, trg_soi=trg_soi)

    y = c2s(source=x_src, target=x_trg)
    print(y.size())