from torch.autograd import Variable
from models.helpers import *
from models.networks.convolutional.attention import Attention



def check_decoder_sizes(enc_h, prev_s, target):
    print(enc_h.size())
    print(prev_s.size())
    print(target.size())

    # print("prev_s")
    # print(prev_s.size())
    # print("dec_h")
    # print(dec_h.size())


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_len, trg_soi, cuda=True, highway=False, dropout=True):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.trg_soi = trg_soi
        self.cuda = cuda
        self.highway = highway
        self.dropout = dropout

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_dim)
        self.decodercell = DecoderCell(embed_dim, hidden_dim)
        self.dec2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_h, prev_s, target=None):
        '''
        enc_h  : B x S x 2*H
        prev_s : B x H
        '''

        if target is not None:
            target_len, batch_size = target.size(0), target.size(1)
            dec_h = Variable(torch.zeros(batch_size, target_len, self.hidden_dim))
            if self.cuda:
                dec_h = dec_h.cuda()

            target = self.embed(target)

            if self.dropout:
                target = F.dropout(target, 0.2)

            for i in range(target_len):
                ctx = self.attention(enc_h, prev_s)
                prev_s = self.decodercell(target[i, :], prev_s, ctx)
                print("pres s {}".format(prev_s.size()))
                dec_h[:, i, :] = prev_s # .unsqueeze(1)
            if self.highway:
                dec_h = dec_h.permute(1, 0, 2)
            outputs = self.dec2word(dec_h)
        # for prediction
        else:
            batch_size = enc_h.size(1)
            target = Variable(torch.LongTensor([self.trg_soi] * batch_size), volatile=True).view(batch_size, 1)
            outputs = Variable(torch.zeros(batch_size, self.max_len, self.vocab_size))

            if self.cuda:
                target = target.cuda()
                outputs = outputs.cuda()

            print("max len {}".format(self.max_len))

            for i in range(self.max_len):
                target = self.embed(target).squeeze(1)
                ctx = self.attention(enc_h, prev_s)
                prev_s = self.decodercell(target, prev_s, ctx)
                print("pres s {}".format(prev_s.size()))
                output = self.dec2word(prev_s)
                outputs[:, i, :] = output
                target = output.topk(1)[1]
        return outputs

    # enc_h, prev_s, target = None):
    def forward_ss(self, enc_h, prev_s, target, inds):

        target_len, batch_size = target.size(0), target.size(1)
        dec_h = Variable(torch.zeros(batch_size, target_len, self.hidden_dim))

        if self.cuda:
            dec_h = dec_h.cuda()

        target = self.embed(target)

        print("target length {}".format(target_len))

        for i in range(target_len):
            ctx = self.attention(enc_h, prev_s)

            if i in inds:
                output = self.dec2word(prev_s)
                outputs[:, i, :] = output
                target = output.topk(1)[1]
                target = self.embed(target)
            else:
                prev_s = self.decodercell(target[i, :], prev_s, ctx)
                dec_h[:, i, :] = prev_s  # .unsqueeze(1)
                outputs = self.dec2word(dec_h)

        if self.highway:
            dec_h = dec_h.permute(1, 0, 2)

        return outputs


class DecoderCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(DecoderCell, self).__init__()

        self.input_weights = nn.Linear(embed_dim, hidden_dim * 2)
        self.hidden_weights = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ctx_weights = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        self.input_in = nn.Linear(embed_dim, hidden_dim)
        self.hidden_in = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_in = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, trg_word, prev_s, ctx):
        '''
        trg_word : B x E
        prev_s   : B x H
        ctx      : B x 2*H
        '''
        gates = self.input_weights(trg_word) + self.hidden_weights(prev_s) + self.ctx_weights(ctx)
        reset_gate, update_gate = gates.chunk(2, 1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        prev_s_tilde = self.input_in(trg_word) + self.hidden_in(prev_s) + self.ctx_in(ctx)
        prev_s_tilde = torch.tanh(prev_s_tilde)
        prev_s = torch.mul((1 - reset_gate), prev_s) + torch.mul(reset_gate, prev_s_tilde)
        return prev_s


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 rnn_type='gru', bidir=True, dropout=True, dropout_rate = 0.2):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bidir = bidir
        self.dropout = dropout
        self.drate = dropout_rate

        # (input_size, hidden_size, num_layers)
        self.gru = get_rnn(rnn_type, embed_dim,
                           self.hidden_dim, self.num_layers, self.drate,
                           batch_first=True, bidirectional=bidir)
        #self.gru = nn.GRU(embed_dim, self.hidden_dim,
        #                  self.num_layers, batch_first=True, bidirectional=bidir)

    # GRU = (seq_len, batch, input_size)
    def forward(self, source, src_length=None, hidden=None):
        '''
        source: B x T
        '''
        batch_size = source.size(1)
        src_embed = self.embedding(source.t())

        if self.dropout:
            src_embed = F.dropout(src_embed, 0.2)

        if hidden is None:
            num_layer_direction = self.num_layers * 2 if self.bidir else self.num_layers
            h_size = (num_layer_direction, batch_size, self.hidden_dim)
            enc_h_0 = Variable(src_embed.data.new(*h_size).zero_(), requires_grad=False)

        if src_length is not None:
            src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_length, batch_first=True)

        # print(src_embed.size()) # torch.Size([batch_size, seq_length, emb_size])
        # print(enc_h_0.size()) # torch.Size([seq_len, batch_size, hid_size]) makes sense 2 hidden state and W, U for both
        enc_h, enc_h_t = self.gru(src_embed, enc_h_0)
        # expects seq_len, batch, num_directions * hidden_size

        if src_length is not None:
            enc_h, _ = nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

        # print(enc_h.size())
        # print(enc_h_t.size())

        return enc_h, enc_h_t


class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer,
                 embed_dim, hidden_dim, max_len, trg_soi,
                 cuda=True, dropout=True, bidir=True):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword
        self.num_layer = num_layer
        self.encoder = Encoder(src_nword, embed_dim, hidden_dim, dropout=dropout, bidir=bidir)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim,
                               max_len, trg_soi, cuda=cuda, dropout=dropout)

    def forward(self, source, src_length=None, target=None):
        batch_size = source.size(0)
        enc_h, enc_h_t = self.encoder(source, src_length)  # B x S x 2*H / 2 x B x H
        print(enc_h_t.size())
        dec_h0 = enc_h_t[-1]  # B x H
        dec_h0 = torch.tanh(self.linear(dec_h0))  # B x 1 x 2*H
        print(dec_h0.size())
        # enc_h: (40, 20, 800) for attention mechanism

        out = self.decoder(enc_h, dec_h0, target)  # B x S x H
        out = torch.log_softmax(out.contiguous().view(-1, self.trg_nword), dim=1)
        return out


if __name__ == "__main__":
    from models.networks.enc2dec.test_var import *

    # always (sent_len, batch_size, vocab_size) prior to flattening

    ss_model = Seq2Seq(src_nword=src_vocab_size, trg_nword=trg_vocab_size,
                       num_layer=nlayers, embed_dim=emb_size, hidden_dim=hidden_size,
                       max_len=sent_len, trg_soi=trg_soi, cuda=False)

    ss_model(x_src, target=x_trg)