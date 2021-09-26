from models.networks.convolutional.convolution import CNN, DeCNN
from torch import nn


class Conv2Conv(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, kernel_num, embed_dim, hidden_dim, max_len, trg_soi, kernel_sizes=None):
        super(Conv2Conv, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword
        self.kernel_sizes = (2, 3, 4) if kernel_sizes is None else kernel_sizes
        self.encoder = CNN(src_nword, embed_dim, hidden_dim, kernel_sizes=kernel_sizes, s2s=True)
        self.decoder = DeCNN(trg_nword, embed_dim, hidden_dim, kernel_sizes=kernel_sizes, s2s=False)

        # 900, 400
        self.fc1 = nn.Linear(kernel_num * len(self.kernel_sizes), hidden_size)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, source, src_length=None, target=None):
        enc_h = self.encoder(source)  # B x S x 2*H / 2 x B x H
        enc_h = torch.tanh(self.fc1(enc_h))  # B x 1 x 2*H
        out = self.decoder(target, enc_h)  # B x S x H
        out = torch.log_softmax(out.contiguous().view(-1, self.trg_nword), dim=1)
        return out


if __name__ == "__main__":

    # always (sent_len, batch_size, vocab_size) prior to flattening

    from models.networks.enc2dec.test_var import *

    conv2conv = Conv2Conv(src_nword=src_vocab_size, trg_nword=trg_vocab_size, kernel_num=300,
                   num_layer=nlayers, embed_dim=emb_size, hidden_dim=hidden_size,
                   max_len=sent_len, trg_soi=trg_soi)

    y = conv2conv(source=x_src, target=x_trg)

