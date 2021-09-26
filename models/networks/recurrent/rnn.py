"""
Works well for language modelling
"""

import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from models.networks.recurrent.goru import GORU
from models.networks.recurrent.urnn import EURNN
from models.networks.recurrent.qrnn import QRNN
from models.networks.convolutional.highway import RecurrentHighwayText
from models.networks.recurrent.gridlstm import GridLSTM
from models.regularizers.dropout import get_dropout
from models.regularizers.dropconnect import dropconnect
from models.posteriors.softmax import get_softmax
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from models.loss.loss import get_criterion
from torch.distributions import Categorical


def embedding_torch_matrix(pretrained, tune=False):
    em_words = torch.from_numpy(np.array(list(pretrained.values())))
    embedding = nn.Embedding(em_words.size(0), em_words.size(1))
    embedding.weight.data.copy_(em_words)
    embedding.weight.requires_grad = tune
    return embedding


# enc_nhid note: added because dim mismatch otherwise,
# on dec_nhid: it was nhid
"""better to pass model_args that contain argparse config for particular model"""


def get_rnn(rnn_type, ninp, nhid, nlayers, dropout, vocab=None, max_len=50,
            drop_method = 'standard', attention=False, in_dropr=0.0,
            hid_dropr=0.0, att_dropr=0.0, cuda = True, batch_first=True,
            num_enc_heads=3, enc_nhid = None, total_enc_key_depth = None,
            total_enc_value_depth = None, enc_filter_size = 10, vocab_size=None,
            num_dec_heads = 4, dec_nhid=None, total_dec_key_depth = None,
            total_dec_value_depth = None,  filter_size = 10, model_args=None):

    rnn_type = rnn_type.upper()
    if rnn_type in ['LSTM', 'GRU']:
        rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
    elif rnn_type == 'URNN':
        rnn = EURNN(ninp, nhid, capacity=nlayers, cuda=cuda)
    elif rnn_type == 'GORU':
        rnn = GORU(ninp, nhid, num_layer=nlayers, dropout=dropout,
                   capacity=nlayers,  embedding=False)
    elif rnn_type == "QRNN":
        # need to install cuda driver first so i can install cupy
        # which means i need a download of visual studio
        rnn = QRNN(ninp, nhid,  vocab, num_layers=nlayers, dropout=dropout)
    elif rnn_type == "GRID_LSTM":
        rnn = GridLSTM(1, nhid, nhid)
    elif rnn_type == 'HIGHWAY':
        rnn = RecurrentHighwayText(ninp, nhid, nlayers, drop_method=drop_method, lm=True,
                                   dropout_rate=dropout, embedding=False, attention=attention)
    else:
        try:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        except KeyError:
            raise ValueError("""An invalid option for `--model` was supplied,
                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
    return rnn



class RNNModel(nn.Module):
    """
    If latent is true, it treats the output as nhid so to be used for LNLM
    """
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, bsize, pretrained, approx_softmax=None, noise_ratio=0.5,
                 norm_term=9, drop_rate=0.2, drop_method='standard', drop_position=1, fixed_drop=True, # codebook=None,
                 unigram_dist=None, pos=False, nptb_tags=None, adversarial=False, batch_norm=False, dropc=False, temp=1,
                 ecoc_nbits=None, cw_mix=False, ss_emb=False, ss_soft_emb=False, nud_tags=None, tie_weights=False,
                 tune_weights=False, latent=False, cutoff=[2000, 10000], vocab_size=None, alpha=None, beta=None,
                 kappa=0.15):
        super(RNNModel, self).__init__()

        drop_dim = nhid if drop_method == 'variational' else None

        self.temp = temp
        self.ntokens = ntoken
        self.batch_norm = batch_norm
        self.cutoff = cutoff  # used for adaptive softmax
        self.pos = pos
        self.nptb_tags = nptb_tags
        self.nud_tags = nud_tags
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.adversarial = adversarial
        self.drop_method = drop_method
        self.drop_position = drop_position
        self.alpha = alpha
        self.beta = beta
        self.dropc = dropc
        self.batch_size = bsize
        self.reg_loss = 0
        self.fixed_dropout = fixed_drop
        self.approx_softmax = approx_softmax
        self.unigram_dist = unigram_dist # needed for nce criterion

        # choice between standard ss, decoder output ss or cw mixture sampling
        # self.codebook = codebook

        """
        if (self.ecoc is not None and self.cw_mix) and (self.codebook is False)\
                or (self.ecoc is not None and self.ss_emb) and (self.codebook is False):
            raise NameError("Codebook needs to be passed if you're"
                            " going to use ecoc with standard SS or Codeword Mixture Sampling")
        """

        """
        You need to pass codebook when using ECOC with sampling methods because 
        the prediction needs to be looked up. Moreover, you cannot use predicted decoder 
        output instead of predicted token in this case because decoder output is not
        same dimension as embedding input (14 codes instead of 400)
        """

        # if ss_emb=True, then we retrieve embedding of predicted token,
        # if ss_emb=False, we instead pass the decoder output as input to model
        self.ss_emb = ss_emb
        self.ss_soft_emb = ss_soft_emb
        # ecoc_nbits
        self.ecoc = ecoc_nbits
        # whether to use codeword mixture sampling
        self.cw_mix = cw_mix

        # this is not finished, decided to just use separate these in
        # original github awd_lstm implementation for the moment instead
        if drop_method == 'fraternal':
            self.double_target = True
            self.eval_auxiliary = False
            if kappa <= 0: self.kappa = 0.15
        elif drop_method == 'eld':
            self.double_target = False
            self.eval_auxiliary = True
            if kappa <= 0: self.kappa = 0.25
        elif drop_method == 'pm':
            self.double_target = False
            self.eval_auxiliary = False
            if kappa <= 0: self.kappa = 0.15

        if (self.ecoc is not None and self.cw_mix) or (self.ecoc is not None and self.ss_emb):
            bin_conv = [pow(2, i) for i in reversed(range(ecoc_nbits))]
            self.bin2int = torch.Tensor(bin_conv).unsqueeze(1).cuda()

        if self.cw_mix:
            self.sig = torch.nn.Sigmoid()

        if self.batch_norm:
            self.bnorm_in = nn.BatchNorm1d(nhid)
            self.bnorm_out = nn.BatchNorm1d(nhid)

        self.rnn = get_rnn(rnn_type, ninp, nhid,  nlayers, drop_rate, drop_method, vocab_size=vocab_size)

        if pretrained is not None:
            if type(pretrained) == list:
                self.encoder = [embedding_torch_matrix(wvs, tune_weights) for wvs in pretrained]
            else:
                self.encoder = embedding_torch_matrix(pretrained, tune_weights)
            print("Pretrained Vocab Shape {}".format(self.encoder.weight.size()))

            """take note of priority order"""
            if approx_softmax is not None:
                if "relaxed" in approx_softmax:
                    self.decoder = get_softmax(None, nhid, ntoken)
                elif approx_softmax == "adasoftmax":
                    self.decoder = get_softmax(approx_softmax, nhid, ntoken, self.cutoff)
                elif approx_softmax != "nce":
                    self.decoder = get_softmax(approx_softmax, nhid, ntoken)
            elif ecoc_nbits is not None:
                self.decoder = nn.Linear(nhid, ecoc_nbits)
            elif latent:
                self.decoder = nn.Linear(nhid, nhid)
            elif pos:
                self.decoder = nn.Linear(nhid, nud_tags)
            else:
                self.decoder = nn.Linear(nhid, ntoken)
            self.init_decoder_weights()

        else:
            self.encoder = nn.Embedding(ntoken, ninp)
            print("Vocab Shape {}".format(self.encoder.weight.size()))

            """take note of priority order"""
            if approx_softmax == "nce":
                self.criterion = get_criterion(approx_softmax, nhid=nhid, ntoken=ntoken,
                                               noise=unigram_dist, noise_ratio=noise_ratio,
                                               norm_term=norm_term)
            elif approx_softmax is not None:
                if "relaxed" in approx_softmax:
                    if approx_softmax == "relaxed_ecoc":
                        self.sig = torch.nn.Sigmoid()
                    elif approx_softmax == "relaxed_softmax":
                        self.softmax = torch.nn.Softmax()
                    self.decoder = get_softmax(None, nhid, ntoken)
                else:
                    self.decoder = get_softmax(approx_softmax, nhid, ntoken)
            elif ecoc_nbits is not None:
                self.decoder = nn.Linear(nhid, ecoc_nbits)
            elif latent:
                self.decoder = nn.Linear(nhid, nhid)
            elif pos:
                self.decoder = nn.Linear(nhid, nud_tags)
            else:
                if self.rnn_type.lower() == "highway":
                    self.decoder = nn.Linear(nhid * 2, ntoken)
                else:
                    self.decoder = nn.Linear(nhid, ntoken)

        if approx_softmax is not None:
            self.init_encoder_weights()
        else:
            self.init_weights()

        if nptb_tags is not None and pos:
            self.ptb_decoder = nn.Linear(nhid, nptb_tags)

        if tie_weights and pos is False and latent is False:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        "module, weights, drop_position, drop_rate, drop_dim, drop_method"
        # weight_ih_l, 'weight_hh_l{}{} 'bias_ih_l{}{}', 'bias_hh_l{}{}'
        rnn_param_names = [name for name, _ in self.rnn.named_parameters()]
        encoder_param_names = [name for name, _ in self.encoder.named_parameters()]
        if approx_softmax != "nce" and approx_softmax is not None:
            if "relaxed" not in approx_softmax:
                decoder_param_names = [name for name, _ in self.decoder.named_parameters()]

        """Takes care of weight dropping"""
        if self.dropc:
            self.dropc_in = dropconnect(self.encoder, encoder_param_names,
                                        drop_rate, drop_dim, drop_method)
            self.dropc_out = dropconnect(self.decoder, decoder_param_names,
                                         drop_rate, drop_dim, drop_method)
            self.dropc_rnn = dropconnect(self.rnn, rnn_param_names,
                                         drop_rate, drop_dim, drop_method)

        """Takes care of activation dropout on input and output"""
        self.drop_in, self.drop_out = get_dropout(drop_position,
                                                  drop_rate, drop_dim,
                                                  drop_method, fixed_drop)

        """Takes care of activation dropout on input and output if using a
         dropout that requires to pass a nn layer (concrete, standout) """
        if self.drop_method == 'concrete':
            if self.drop_in is not False:
                self.drop_in = self.drop_in(self.rnn, input_shape=(bsize, nhid),
                                            weight_regularizer=1e-6,
                                            dropout_regularizer=1e-5, locked=fixed_drop)
            if self.drop_out is not False:
                self.drop_out = self.drop_out(self.decoder, input_shape=(bsize, ntoken),
                                              weight_regularizer=1e-6,
                                              dropout_regularizer=1e-5, locked=fixed_drop)
        elif self.drop_method == 'standout':
            if alpha is None or beta is None:
                ValueError("alpha and beta required for fraternal dropout")
            if self.drop_in is not False:
                self.drop_in = self.drop_in(self.rnn, alpha, beta)
            if self.drop_out is not False:
                self.drop_in = self.drop_out(self.decoder, alpha, beta)

    def init_encoder_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, 0.1)

    def init_decoder_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, 0.1)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_input_mask(self, emb, p=None):
        """assigned input dropout, unless concrete in which
        case its already wrapped as apart of self.recurrent"""
        if self.drop_position in [1, 3]:
            if self.drop_method == 'curriculum' and p is not None:
                self.drop_in.p = p
                # emb = self.drop_in(emb, p)
            elif self.drop_method != 'concrete':
                emb = self.drop_in(emb)
        return emb

    def get_output_mask(self, output, p = None):
        """assigned output dropout, unless concrete in which
        case its already wrapped as apart of self.decoder"""
        if self.drop_position in [2, 3]:
            if self.drop_method == 'curriculum' and p is not None:
                self.drop_out.p = p
            if self.drop_method != 'concrete':
                output = self.drop_out(output)
        return output

    def forward_pass(self, emb, hidden):
        """Takes care of the mess for highway networks and concrete
        i.e when using concrete, recurrent already wrapped and when using highway,
        only need to pass emb and not hidden
        """
        if self.drop_position in [1, 3]:
            if self.rnn_type.lower() == "highway" and self.drop_method != 'concrete':
                output, hidden = self.rnn(emb)
            elif self.rnn_type.lower() == "highway" and self.drop_method == 'concrete':
                output, hidden = self.drop_in(emb)
                self.reg_loss += self.drop_in.regularisation()
            else:

                if 'transformer' in self.rnn_type.lower():
                    return self.rnn(emb)
                else:
                    output, hidden = self.rnn(emb, hidden)
        else:
            if self.rnn_type.lower() == "highway":
                output, hidden = self.rnn(emb)
            else:
                if 'transformer' in self.rnn_type.lower():
                    return self.rnn(emb), None
                else:
                    output, hidden = self.rnn(emb, hidden)
        return output, hidden

    def forward_gen(self, input, hidden, hidden_gen=None):
        """takes discriminator input and generated sequence a return disc prediction"""
        if hidden_gen is None:
            hidden_gen = hidden
        if self.drop_in:
            emb = self.drop_in(self.encoder(input))
        x = emb[0, :, :].unsqueeze(0)
        outputs = []
        for i in range(emb.size(0)):
            x, hidden_gen = self.rnn(x, hidden_gen)
            outputs.append(x)
        output = torch.cat(outputs, 0)
        if self.drop_out:
            gen_output = self.drop_out(output)
        gen_out = self.decoder(output.view(gen_output.size(0) * output.size(1), output.size(2)))
        out, hidden = self.forward(input, hidden)
        return out, hidden, gen_out, hidden_gen

    """Seems to be problem with calling backward after sampling (),
     but seems to be ok for hierarchical softmax which is weird ?"""
    def _relaxed_logit(self, pred):
        pred_relaxed = LogitRelaxedBernoulli(self.temp, logits=self.sig(pred))
        return pred_relaxed.sample()

    def _sampled_softmax(self, pred, targets=None):
        if targets is None:
            logits = self.decoder.full(pred.view(-1, pred.size(2)))
            return logits
        else:
            logits, new_targets = self.decoder(pred.view(-1, pred.size(2)), targets)
            return logits, new_targets

    def _relaxed_softmax(self, pred):
        logits = self.softmax(pred)
        pred_relaxed = RelaxedOneHotCategorical(self.temp, probs=logits)
        return Variable(pred_relaxed.sample(), requires_grad=True)

    def _get_pred_emb(self, pred, k=1):
        """
        Soft reparameterized sampling - Notice this has no effect on the backpass,
                                        sampling only used to choose single input
        :param pred: passes decoder output
        :return: returns out, embedding for next time-step
        """
        # act greedy if k=1,
        if k == 1:
            idx = torch.topk(pred, k=k, dim=1)[1]
        else:
            vals, idx = torch.topk(pred, k=k, dim=1)
            probs = self.norm(vals)
            m = Categorical(probs)
            idx = m.sample()
        out = self.encoder(idx)
        return out

    def _get_softplus_pred_emb(self, pred, k=30):
        """ Soft Argmax: Linear combination of embeddings
            NOT for ECOC since no normalization is used in ECOC
            pred (batch_size, ntokens)
        """
        vals, idx = torch.topk(pred, k=k, dim=1)
        probs = self.norm(vals)
        # (batch_size, k, ntokens)
        emb = self.encoder(idx)
        emb = emb.squeeze()
        norm_emb = probs.unsqueeze(2) * emb
        norm_emb = torch.sum(norm_emb, 1)
        return norm_emb

    def _get_cw_mixture(self, p_cw, t_cw, mix_prob=0.1):
        """
        Expects (batch_size * seq_len, codelength) for both
        pred_cw is are floats,  so we still need to round to ints to get mixture
        """

        pred = p_cw.detach()
        target_cw = t_cw.detach()
        num_samps = int(pred.size(1) * mix_prob)

        if num_samps == 0: return t_cw
        probs = self.sig(pred)
        pred_cw = torch.round(probs).type(torch.cuda.LongTensor)

        pred_x_ind = list(range(pred_cw.size(0)))
        x_ind = torch.cuda.LongTensor(pred_x_ind * num_samps)
        y_ind = torch.multinomial(probs, num_samps)
        target_cw[x_ind, y_ind.view(-1)] = probs[x_ind, y_ind.view(-1)]
        print("target: \t {}".format(target_cw.size()))
        return target_cw

    def cw_forward_ss(self):
        """codeword mixture sampling for ECOC"""
        # not used thus far
        pass

    def forward_ss(self, x, hidden, inds=None, p = 0.1, target=None, k=1):
        """FIXED (_get_pred_emb): At the minute it passes the decoder output as input,
        instead of the predicted token that is passed through the embedding mat

         Note: When using cw_mixture sampling, we don't use inds that are passed in this function
         since it is the same sampled inds for each sample in the batch. Samples in _get_cw_mixture
         randomize for each sample
        """

        self.reg_loss = 0

        if inds is None:  inds = []
        # if self.drop_method != 'concrete':

        # print("x: {}".format(x.size()))
        emb = self.encoder(x)
        emb = self.get_input_mask(emb, p)

        if self.cw_mix:
            target = target.view(x.size(0), x.size(1), -1)

        # 35x20x200
        outputs = []

        for i in range(emb.size(0)):
            if (i in inds and i != 0) or (self.cw_mix and i != 0):

                if self.rnn_type.lower() == "highway":
                    output, hidden = self.rnn(output)
                else:
                    output, hidden = self.rnn(output, hidden)

                # important additions (note priority)
                if self.ecoc is None:
                    if self.ss_emb:
                        output = self._get_pred_emb(output, k=1)
                    elif self.ss_soft_emb:
                        output = self._get_softplus_pred_emb(output, k=1)

                inds = list(filter(lambda a: a != i, inds))

                if self.ecoc is not None:
                    prediction = self.decoder(output.view(-1, output.size(2)))
                    if self.cw_mix:
                        # need to get output first
                        cw_pred = self._get_cw_mixture(prediction, target[i, :],
                                                        mix_prob=p)
                        # print("cwp {} self.bin2int {}".format(cw_pred.size(), self.bin2int.size()))
                        pred_ind = torch.mm(cw_pred, self.bin2int).type(torch.cuda.LongTensor).t()
                        output = self.encoder(pred_ind) # emb[pred_ind, :, :]
                        # print("{}-{}".format(i, output.size()))
                    else:
                        """need to convert predicted codewords (output) back 
                            to ints so to look embedding matrix"""
                        pred = torch.round(prediction).type(torch.cuda.LongTensor)
                        pred_ind = torch.mm(self.bin2int, pred).type(torch.cuda.LongTensor).t()
                        output = self.encoder(pred_ind) # emb[pred_ind, :, :]

                        # output = emb[pred_ind, :, :]
            else:
                x = emb[i, :, :].unsqueeze(0)
                if self.rnn_type.lower() == "highway":
                    output, hidden = self.rnn(x)
                else:
                    # print("{}-{}".format(i, x.size()))
                    output, hidden = self.rnn(x, hidden)

                """needs to be here because if at bottom it does sample step by step"""
                if self.approx_softmax == "relaxed_softmax":
                    output = self._relaxed_softmax(self.decoder(output))
                elif self.approx_softmax == "relaxed_ecoc":
                    output = self._relaxed_logit(self.decoder(output))
                # relaxed_hsoftmax takes care of itself

            outputs.append(output)

        assert inds == [] or inds == [0]
        output = torch.cat(outputs, 0)

        output = self.get_output_mask(output, p)

        if self.approx_softmax == "nce":
            return output, hidden

        elif self.approx_softmax == "hsoftmax":
            decoded = self.decoder(output.view(-1, output.size(2)), labels=target)
            return decoded, hidden
            # decoded = perform_softmax(self.approx_softmax, self.decoder, target=None)

        elif self.approx_softmax == "adasoftmax":
            decoded = self.decoder(output.view(-1, output.size(2)), target)
            return decoded, hidden

        elif self.approx_softmax == "sampled_softmax":
            decoded = self._sampled_softmax(output)
            return decoded, hidden

        elif self.approx_softmax == "soft_mix":
            decoded = self.decoder(output)
            return decoded, hidden

        if self.drop_position in [2, 3]:
            if self.rnn_type.lower() == 'highway' and self.drop_method != 'concrete':
                decoded = self.decoder(output.view(output.size(0) *
                                                   output.size(1), output.size(2)))
            elif self.drop_method == 'concrete':
                decoded = self.drop_out(output)
                self.reg_loss += self.drop_out.regularisation()
                print("second")
                return decoded, hidden
        else:
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        if self.nptb_tags is not None and self.pos:
            ptb_decoded = self.ptb_decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            return (ptb_decoded.view(output.size(0), output.size(1), decoded.size(1)),
                    decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden)

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    """ gamma and p are only needed when using curriculum dropout I think. """
    """ CORRECTED: Just need to update nn.Dropout().p"""
    def forward(self, x, hidden, p=None, target=None):

        self.reg_loss = 0

        emb = self.encoder(x)
        emb = self.get_input_mask(emb, p)

        """Not sure if correct or not """
        if self.dropc:
            emb = self.dropc_in(emb) if p is None else self.dropc_in(emb, p)

        if self.batch_norm:
            emb = self.bnorm_in(emb)

        output, hidden = self.forward_pass(emb, hidden)

        """Not sure if correct or not """
        if self.dropc:
            output = self.dropc(output) if p is None else self.dropc_in(output, p)

        if self.batch_norm:
            output = self.bnorm_out(output)

        output = self.get_output_mask(output, p)

        if self.approx_softmax == "nce":
            """Since IndexLinear is used in NCELoss, no need for decoder layer here.
            We could also change this to return loss, but for uniformity we use seperate 
            func for this.
            """
            return output, hidden

        elif self.approx_softmax == "hsoftmax" or self.approx_softmax == "hsoft_mix":
            decoded = self.decoder(output.view(-1, output.size(2)), labels=target)
            return decoded, hidden
            # decoded = perform_softmax(self.approx_softmax, self.decoder, target=None)

        elif self.approx_softmax == "adasoftmax":
            decoded = self.decoder(output.view(-1, output.size(2)), target)
            return decoded, hidden

        elif self.approx_softmax == "soft_mix" or self.approx_softmax == "ecoc_mix" or \
                self.approx_softmax == "soft_mix_tuned" or self.approx_softmax == "ecoc_mix_tuned":
                decoded = self.decoder(output)
                return decoded, hidden

        elif self.approx_softmax == "relaxed_softmax":
            decoded = self.decoder(output.view(-1, output.size(2)))
            decoded = self._relaxed_softmax(decoded)
            return decoded, hidden

        elif self.approx_softmax == "sampled_softmax":
            if target is None:
                decoded = self._sampled_softmax(output)
                return decoded, hidden
            else:
                decoded, new_targets = self._sampled_softmax(output, target)
                return decoded, hidden, new_targets

        elif self.approx_softmax == "relaxed_ecoc":
            decoded = self._relaxed_logit(self.decoder(output.view(-1, output.size(2))))
            return decoded, hidden

        elif self.drop_position in [2, 3]:
            if self.rnn_type.lower() == 'highway' and self.drop_method != 'concrete':
                decoded = self.decoder(output.view(output.size(0) *
                                                   output.size(1), output.size(2)))
            elif self.drop_method == 'concrete':
                """no self.decoder since it is apart of self.drop_out when concrete used.
                Also haven't flattened dim 0 and 1 like below here """
                decoded = self.drop_out(output)
                self.reg_loss += self.drop_out.regularisation()
                # print("second")
                return decoded, hidden
            else:
                decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        else:
            # output.size(2)
            output = output.view(output.size(0) * output.size(1), output.size(2))
            decoded = self.decoder(output)

        if self.nptb_tags is not None and self.pos:
            ptb_decoded = self.ptb_decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            return (ptb_decoded.view(output.size(0), output.size(1), decoded.size(1)),
                    decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden)

        return decoded, hidden

    def forward_seqtoseq(self, x_source, hidden_enc, x_target):
        emb = self.encoder(x_source)
        if self.drop_in:
            emb = self.drop_in(emb)
        enc_output, enc_hidden = self.rnn(emb, hidden_enc)
        if self.drop_out:
            enc_hidden = self.drop_out(enc_hidden)
        output, hidden = self.rnn_decoder(x_target, enc_hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def regularisation_loss(self, clear=False):
        reg_loss = self.forward_main[0].regularisation()\
                   +self.forward_main[1].regularisation()\
                   +self.forward_main[2].regularisation()
        return reg_loss

    def nce_loss(self, pred, target):
        #if hasattr(RNNModel, 'criterion'):
        loss = self.criterion(target, pred)
        return loss.mean()
