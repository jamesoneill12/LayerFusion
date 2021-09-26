import torch
import torch.nn as nn
import numpy as np
import math
from .layers import EncoderLayer, DecoderLayer
from .normalization import LayerNorm
import torch.nn.functional as F

import copy
import json
import math
import re
import collections


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)



def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = nn.Parameter(w)
            self.b = nn.Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.b[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth,
                 total_value_depth, filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.enc = nn.Sequential(*[EncoderLayer(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        # Add input dropout
        x = self.input_dropout(inputs)
        # Project to hidden size
        x = self.embedding_proj(x)
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        y = self.enc(x)
        y = self.layer_norm(y)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf

    To understand how transformer generation works - (https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
    The decoder operates similarly, but generates one word at a time, from left to right.
    It attends not only to the other previously generated words, but also to the final representations generated by the encoder.
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, vocab_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        # why linear layer here and not an Embedding layer ??
        self.embedding_proj = nn.Embedding(embedding_size, hidden_size) # , bias=False)
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output):
        # inputs should be [batch_size, length, hidden_size]
        # if len(inputs.size()) < 3: inputs = inputs.squeeze(1)
        # Project to hidden size
        x = self.embedding_proj(inputs)
        # Add input dropout
        x = self.input_dropout(x)

        # print(x.size())
        # print(inputs.size())
        # print(self.timing_signal.size())
        # timing_signal (max_length, hidden_size)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(x.data)

        # added for image captioning because I was getting errors otherwise
        if encoder_output.size() != x.size():
            encoder_output = encoder_output.unsqueeze(1)

        # Run decoder
        y, _ = self.dec((x, encoder_output))
        # Final layer normalization
        y = self.layer_norm(y)
        # decode to size of output
        y = self.out_layer(y)
        return y

    def generate(self, encoder_output, seq_len, beam=False):
        """
        inputs should be [batch_size, length] (hidden_size when passed embedding_proj)
        need to figure out how to do generation with just start token
        look at the openai transformer LMHead used in LMModel for reference
        """
        input = torch.zeros((encoder_output.size(0), 1)).type(torch.cuda.LongTensor)
        ys = []
        for i in range(seq_len):
            x = self.embedding_proj(input)
            # Add input dropout
            x = self.input_dropout(x)
            # Add timing signal
            x += self.timing_signal[:, i, :].type_as(x.data)
            # Run decoder
            y, _ = self.dec((x, encoder_output.unsqueeze(1)))
            # Final layer normalization
            y = self.layer_norm(y)
            # decode to size of output
            x = self.out_layer(y)
            # need to return probs so this line removed
            input = self._beam(x) if beam else x.argmax(2)
            ys.append(x)
        ys = torch.stack(ys, 1).squeeze().type(torch.cuda.FloatTensor)
        return ys

    def beam_(self, per):
        return per.argmax(1)


class LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, n_embd, trunc_and_reshape=True):
        super(LMHead, self).__init__()
        self.n_embd = n_embd
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight # Tied weights
        self.trunc_and_reshape = trunc_and_reshape  # XD

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd) \
            if self.trunc_and_reshape else h  # XD
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(TransformerModel, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        x = x.view(-1, x.size(-2), x.size(-1))
        h = self.drop(self.embed(x)).squeeze()
        # print(h.size())
        # Add the position information to the input embeddings
        # h = e.sum(dim=2)
        for block in self.h:
            h = block(h)
        return h


class LMModel(nn.Module):
    """ Transformer with language model head only """
    def __init__(self, cfg, vocab_size, n_ctx=512):
        super(LMModel, self).__init__()

        # embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
        #                  filter_size, vocab_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
        #                  attention_dropout=0.0, relu_dropout=0.0, n_ctx=512, return_probs=False

        # self.decoder = Decoder(embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
        #         filter_size, vocab_size, max_length=max_length, input_dropout=input_dropout, layer_dropout=layer_dropout,
        #         attention_dropout=attention_dropout, relu_dropout=relu_dropout)

        self.transformer = TransformerModel(cfg, vocab=vocab_size, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, n_embd=cfg.d_embed, trunc_and_reshape=False)
        self.return_probs = cfg.return_probs
        if self.return_probs:
            pos_emb_mask = torch.zeros(1, 1, vocab_size)
            pos_emb_mask[:, :, -n_ctx:] = -1e12
            self.register_buffer('pos_emb_mask', pos_emb_mask)

    def forward(self, x):
        h = self.transformer(x)
        # print("h {}".format(h.size()))
        lm_logits = self.lm_head(h)
        # print("lm logit {}".format(lm_logits.size()))
        if self.return_probs:
            lm_logits = F.softmax(lm_logits + self.pos_emb_mask, dim=-1)
        return lm_logits


def get_transformer(tf_type, args):
    tf_type = tf_type.upper()
    if tf_type == "TRANSFORMER_ENCODER":
        model = Encoder(args.ninp, args.nhid, args.nlayers,
                                  args.num_heads, args.total_key_depth,
                                  args.total_value_depth, args.filter_size, max_length=args.max_len,
                                  input_dropout=args.dropout, layer_dropout=args.dropout,
                                  attention_dropout=args.dropout)
    elif tf_type == "TRANSFORMER_DECODER":

        model = Decoder(args.ninp, args.ninp, args.nlayers,
                                  args.num_heads, args.total_key_depth,
                                  args.total_value_depth, args.filter_size, vocab_size=args.vocab_size,
                                  max_length=args.max_len, input_dropout=args.in_dropr,
                                  layer_dropout=args.hid_dropr, attention_dropout=args.att_dropr)
    elif tf_type == "TRANSFORMER_LM":
        model = LMModel(cfg=args, vocab_size=args.vocab_size, n_ctx=args.n_ctx)
        """
        recurrent = transformer.LMModel(args.ninp, args.ninp, args.nlayers,
                                  args.num_heads, args.total_key_depth,
                                  args.total_value_depth, args.filter_size, vocab_size=args.vocab_size,
                                  max_length=args.max_len, input_dropout=args.in_dropr,
                                  layer_dropout=args.hid_dropr, attention_dropout=args.att_dropr,
                                  n_ctx=512, return_probs=False)        
        """

    elif tf_type == "TRANSFORMER_XL":
        from pytorch_pretrained_bert import TransfoXLLMHeadModel
        model = TransfoXLLMHeadModel(args)
        # Load a pre-trained model
        # model = TransfoXLLMHeadModel.from_pretrained(args.model_name)
        model = model.cuda()

    elif tf_type == "BERT_LM":
        from pytorch_pretrained_bert import BertForMaskedLM
        BertForMaskedLM()

    return model


if __name__ == "__main__":
    pass
    # enc = Encoder()
    # dec = Decoder()