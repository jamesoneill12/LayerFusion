from torch import nn
import torch
from models.networks.generative.vae.made import MADE


class AFLayer(nn.Module):
    def __init__(self, data_dim, n_hidden_layers, n_hidden_units, nonlinearity, transform_function,
                 hidden_order='sequential', swap_trngen_dirs=False,
                 input_order=None, conditional_inp_dim=None, dropout=[0, 0], coupling_level=0):
        super().__init__()

        self.made = MADE(data_dim, n_hidden_layers, n_hidden_units, nonlinearity, hidden_order,
                         out_dim_per_inp_dim=transform_function.num_params, input_order=input_order,
                         conditional_inp_dim=conditional_inp_dim,
                         dropout=dropout)

        self.train_func = transform_function.standard if swap_trngen_dirs else transform_function.reverse
        self.gen_func = transform_function.reverse if swap_trngen_dirs else transform_function.standard
        self.output_order = self.made.end_order
        self.data_dim = data_dim

        self.use_cond_inp = conditional_inp_dim is not None

    def forward(self, inputs):
        """
            Defines the reverse pass which is used during training
            logdet means log det del_y/del_x
        """

        if self.use_cond_inp:
            y, logdet, cond_inp = inputs
            nn_outp = self.made([y, cond_inp])  # [B, D, 2]
        else:
            y, logdet = inputs
            nn_outp = self.made(y)  # [B, D, 2]

        x, change_logdet = self.train_func(y, nn_outp)

        return x, logdet + change_logdet, cond_inp

    def generate(self, inputs):
        """
            Defines the forward pass which is used during testing
            logdet means log det del_y/del_x
        """
        if self.use_cond_inp:
            x, logdet, cond_inp = inputs
        else:
            x, logdet = inputs

        y = torch.tensor(x)
        for idx in range(self.data_dim):
            t = (self.output_order == idx).nonzero()[0][0]

            if self.use_cond_inp:
                nn_outp = self.made([y, cond_inp])
            else:
                nn_outp = self.made(y)

            y[..., t:t + 1], new_partial_logdet = self.gen_func(x[..., t:t + 1], nn_outp[..., t:t + 1, :])
            logdet += new_partial_logdet

        return y, logdet, cond_inp