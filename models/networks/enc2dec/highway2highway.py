from models.networks.enc2dec.seq2seq import Decoder
from models.networks.convolutional.highway import RecurrentHighwayText, RNNHighwayDecoder
from torch import nn


class RNNHway2RNNHway(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer,
                 embed_dim, hidden_dim, max_len, trg_soi, drate=0.2, cuda=True):
        super(RNNHway2RNNHway, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword

        # vocab_size, hidden_size, nlayers=2,
        #                  dropout_rate=0.2, class_num=None, hyper=False
        self.encoder = RecurrentHighwayText(vocab_size=src_nword, hidden_size=hidden_dim,
                                   nlayers=num_layer, dropout_rate=drate)

        # vocab_size, embed_dim, hidden_dim,
        #                  max_len, trg_soi, nlayers=2, gate_drop=0.2
        self.decoder = RNNHighwayDecoder(vocab_size=trg_nword, embed_dim=embed_dim,
                                hidden_dim=hidden_dim, max_len=max_len,
                                trg_soi=trg_soi, nlayers=num_layer,
                                dropout_rate=drate, cuda=cuda)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, source, src_length=None, target=None):
        if target is not None:
            target = target.t()
        enc_h_t, dec_h0 = self.encoder(source)  # B x S x 2*H / 2 x B x H
        # dec_h0 = enc_h_t  # B x H
        dec_h0 = torch.tanh(self.linear(dec_h0))  # B x 1 x 2*H

        # should be and need all enc_h because of attention,
        # otherwise just enc_h_t with 40, 400
        # print(enc_h.size()) 40, 20, 800
        # print(dec_h0.size()) 40, 400
        # print(target.size()) 20, 400

        # should be
        # enc_h: torch.Size([40, 20, 800])
        # dec_h0: torch.Size([40, 400])
        # target: torch.Size([20, 40])

        # print(enc_h_t.size())
        # print(dec_h0.size())
        # print(target.size())

        out = self.decoder(enc_h_t, dec_h0, target)  # B x S x H
        out = torch.log_softmax(out.contiguous().view(-1, self.trg_nword), dim=1)
        return out


class RNNHighway2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim,
                 hidden_dim, max_len, trg_soi, drate=0.2, cuda=True):
        super(RNNHighway2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword

        self.encoder = RecurrentHighwayText(vocab_size=src_nword, hidden_size=hidden_dim,
                                   nlayers=num_layer, dropout_rate=drate)

        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim,
                               max_len, trg_soi, cuda=cuda, highway=True)

    def forward(self, source, src_length=None, target=None):
        if target is not None:
            target = target.t()
        enc_h_t, dec_h0 = self.encoder(source)  # B x S x 2*H / 2 x B x H
        dec_h0 = torch.tanh(self.linear(dec_h0))  # B x 1 x 2*H

        # should be
        # enc_h: torch.Size([40, 20, 800])
        # dec_h0: torch.Size([40, 400])
        # target: torch.Size([20, 40])
        # print(enc_h_t.size())
        # print(dec_h0.size())
        # print(target.size())

        out = self.decoder(enc_h_t, dec_h0, target)  # B x S x H
        out = torch.log_softmax(out.contiguous().view(-1, self.trg_nword), dim=1)
        # expect (800, 5000) = (batch_size*sent_len, ntrg_word )
        return out


if __name__ == "__main__":

    from models.networks.enc2dec.test_var import *
    h2h = False
    cuda = False
    if h2h:
        # using highway for both encoding and decoding
        hway_net = RNNHway2RNNHway(src_nword=src_vocab_size, trg_nword=trg_vocab_size, num_layer=nlayers,
                 embed_dim=hidden_size, hidden_dim=hidden_size, max_len=40,
                                   trg_soi=trg_soi, drate=dropout_rate, cuda=cuda)
    else:
        # using highway for encoding and using the gru for decoding
        # actually how do you assign weights from different networks ? not sure if this is meaningful
        hway_net = RNNHighway2Seq(src_nword=src_vocab_size, trg_nword=trg_vocab_size, num_layer=nlayers,
                 embed_dim=hidden_size, hidden_dim=hidden_size, max_len=40,
                                   trg_soi=trg_soi, drate=dropout_rate, cuda=cuda)

    # always (sent_len, batch_size, vocab_size) prior to flattening

    y = hway_net(source=x_src, target=x_trg)
    print(y.size())
    # torch.Size([20, 40, 5000])
    # torch.Size([800, 5000])

    assert y.size() == torch.Size([batch_size * sent_len, trg_vocab_size])