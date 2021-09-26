from torch import nn
from models.networks.recurrent.goru import GORU


class Goru2Goru(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer,
                 embed_dim, hidden_dim, max_len, trg_soi, drate=0.2):
        super(Goru2Goru, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword
        self.encoder = GORU(input_size=src_nword, hidden_size=hidden_dim,
                            num_layer=num_layer, capacity=2, dropout=drate, embedding=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = GORU(input_size=trg_nword, hidden_size=hidden_dim,
                            num_layer=num_layer, capacity=2, dropout=drate, embedding=True)

    def forward(self, source, src_length=None, target=None):
        batch_size = source.size(0)
        enc_h, enc_h_t = self.encoder(source, src_length)  # B x S x 2*H / 2 x B x H
        # print(enc_h_t.size())
        dec_h0 = enc_h_t[-1]  # B x H
        dec_h0 = torch.tanh(self.linear(dec_h0))  # B x 1 x 2*H
        # print("dec_h0")
        # print(dec_h0.cpu().size())
        out = self.decoder(enc_h, dec_h0, target)  # B x S x H
        out = torch.log_softmax(out.contiguous().view(-1, self.trg_nword), dim=1)
        return out


if __name__ == '__main__':

    from models.networks.enc2dec.test_var import *

    # always (sent_len, batch_size, vocab_size) prior to flattening


    g2g = Goru2Goru(src_nword=src_vocab_size, trg_nword=trg_vocab_size, num_layer=nlayers,
              embed_dim=emb_size, hidden_dim=hidden_size, max_len=sent_len,
              trg_soi=trg_soi, drate=0.2)

    y = g2g(x_src)
    print(y.size())