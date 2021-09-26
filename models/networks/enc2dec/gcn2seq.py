from models.networks.enc2dec.seq2seq import Decoder
from models.networks.graph.gcn import GCN
from torch import nn
import torch

"""
Whats the idea ?
Embedding in GRC Encoder considers input embedding as 2d matrix with an associated adjacency matrix.

What is the adjacency matrix for this embedding though ? 
Number of possibilities - 
    Predefined Dependency Links 
    information extraction adj matrix
    semantic roles adj matrix
    
Why ?
When you want strictly only chooses certain words from the source.
Maybe not the past way to go but worth testing out.
"""


class GCN2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi):
        super(GCN2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword
        self.encoder = GCN(src_nword, embed_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim, max_len, trg_soi)

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