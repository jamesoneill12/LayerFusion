from models.layers.ff import FeedForwardNet
import torch
from torch import nn


class SCFLayer(nn.Module):
    def __init__(self, data_dim, n_hidden_layers, n_hidden_units, nonlinearity, transform_function, hidden_order=None,
                 swap_trngen_dirs=False,
                 input_order=None, conditional_inp_dim=None, dropout=[0, 0]):
        super().__init__()

        self.net = FeedForwardNet(data_dim // 2 + conditional_inp_dim, n_hidden_units,
                                  (data_dim - (data_dim // 2)) * transform_function.num_params, n_hidden_layers,
                                  nonlinearity, dropout=dropout[1])

        self.train_func = transform_function.standard if swap_trngen_dirs else transform_function.reverse
        self.gen_func = transform_function.reverse if swap_trngen_dirs else transform_function.standard
        self.input_order = input_order

        self.use_cond_inp = conditional_inp_dim is not None

    def forward(self, inputs):
        """
            Defines the reverse pass which is used during training
            logdet means log det del_y/del_x
        """

        data_dim = len(self.input_order)
        assert data_dim == inputs[0].shape[-1]

        first_indices = torch.arange(len(self.input_order))[
            self.input_order <= data_dim // 2]  # This is <= because input_order goes from 1 to data_dim+1
        second_indices = torch.arange(len(self.input_order))[self.input_order > data_dim // 2]

        if self.use_cond_inp:
            y, logdet, cond_inp = inputs
            net_inp = torch.cat([y[..., first_indices], cond_inp], -1)
        else:
            y, logdet = inputs
            net_inp = y[..., first_indices]

        nn_outp = self.net(net_inp).view(*net_inp.shape[:-1], data_dim - (data_dim // 2),
                                         -1)  # [..., ~data_dim/2, num_params]

        x = torch.tensor(y)
        x[..., second_indices], change_logdet = self.train_func(y[..., second_indices], nn_outp)

        return x, logdet + change_logdet, cond_inp

    def generate(self, inputs):
        """
            Defines the forward pass which is used during testing
            logdet means log det del_y/del_x
        """

        data_dim = len(self.input_order)
        assert data_dim == inputs[0].shape[-1]

        first_indices = torch.arange(len(self.input_order))[
            self.input_order <= data_dim // 2]  # This is <= because input_order goes from 1 to data_dim+1
        second_indices = torch.arange(len(self.input_order))[self.input_order > data_dim // 2]

        if self.use_cond_inp:
            x, logdet, cond_inp = inputs
            net_inp = torch.cat([x[..., first_indices], cond_inp], -1)
        else:
            x, logdet = inputs
            net_inp = x[..., first_indices]

        nn_outp = self.net(net_inp).view(*net_inp.shape[:-1], data_dim - (data_dim // 2),
                                         -1)  # [..., ~data_dim/2, num_params]

        y = torch.tensor(x)
        y[..., second_indices], change_logdet = self.gen_func(x[..., second_indices], nn_outp)

        return y, logdet + change_logdet, cond_inp