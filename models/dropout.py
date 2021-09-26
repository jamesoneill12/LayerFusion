"""https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def get_dropout(drop_position, drop_rate, drop_dim, drop_method, fixed_dropout=False):
    if drop_position == 1 or drop_position == 3:
        drop_in = dropout(drop_rate, drop_dim,
                          drop_method, fixed=fixed_dropout)
    else:
        drop_in = False
    if drop_position == 2 or drop_position == 3:
        drop_out = dropout(drop_rate, drop_dim,
                           drop_method, fixed=fixed_dropout)
    else:
        drop_out = False
    return drop_in, drop_out


def dropout(p=None, dim=None, method='standard', fixed=False):
    if method == 'standard':
        return LockedDropout(p) if fixed else nn.Dropout(p)
    elif method == 'gaussian':
        return GaussianDropout(p/(1-p), fixed=fixed)
    elif method == 'locked':
        return LockedDropout(p)
    elif method == 'variational':
        """This is specifically gaussian variational dropout
        and doesn't converge for either fixed time steps or non-fixed"""
        return VariationalDropout(p/(1-p), dim, locked=fixed)
    elif method == 'concrete':
        # takes  layer, input_shape
        return ConcreteDropout
    # elif method == 'zoneout':
        # return Zoneout(p, fixed)
    elif method == 'curriculum':
        """Not required, can just change nn.Dropout() param p"""
        # return CurriculumDropout()
        return nn.Dropout(p)
    elif method == 'standout':
        return Standout
    elif method == 'fraternal':
        return FraternalDropout()
    elif method == 'eld':
        pass
        # left for separate train script (awd_lstm_train.py) to test
        # expectation linear dropout


"""
class Zoneout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.p = dropout

    def forward(self, recurrent, p=None):
        if p is not None:
            self.p = p

        if self.training:
            mask = Variable(recurrent['weight_hh_l0'].data.new(*recurrent['weight_hh_l0'].size())
                            .bernoulli_(1 - self.zoneout), requires_grad=False)
            F = F * mask
        else:
            F *= 1 - self.zoneout
"""


class LockedDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.p = dropout

    def forward(self, x, p=None):
        if p is not None:
            self.p = p
        if not self.training or not p:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)
        mask = Variable(m, requires_grad=False) / (1 - self.p)
        mask = mask.expand_as(x)
        return mask * x


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0, fixed=False):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])
        self.fixed = fixed

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            if self.fixed:
                epsilon = torch.randn((1, x.size(1), x.size(2))) * self.alpha + 1
            else:
                epsilon = torch.randn(x.size()) * self.alpha + 1
            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class VariationalDropout(nn.Module):
    """
    Variational Gaussian Dropout is not Bayesian so read this paper:
    https://arxiv.org/abs/1711.02989
    """

    # max alpha is used for clamping and should be small
    def __init__(self, alpha=0.01, dim=None, locked=True):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        self.locked = locked
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        alpha = self.log_alpha.exp()
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3
        kl = -negative_kl
        return kl.mean()

    def forward(self, x):
        """Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e"""
        if self.train():
            # N(0,1)

            if self.locked:
                epsilon = Variable(torch.randn(size=(x.size(0), x.size(2))))
                epsilon = torch.cat([epsilon] * x.size(1)).view(x.size())
            else:
                epsilon = Variable(torch.randn(x.size()))

            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x


"""https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb"""


class ConcreteDropout(nn.Module):
    def __init__(self, layer, input_shape, weight_regularizer=1e-6, locked = True,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        # Post drop out layer
        self.layer = layer
        # Input dim for regularisation scaling
        self.input_dim = np.prod(input_shape[1:])
        # Regularisation hyper-parameters
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.locked = locked
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)

    def forward(self, x):
        return self.layer(self._concrete_dropout(x))

    def regularisation(self):
        """Computes weights and dropout regularisation for the layer, has to be
        extracted for each layer within the model and added to the total loss
        """
        weights_regularizer = self.weight_regularizer * self.sum_n_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer

    def _concrete_dropout(self, x):
        """Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 0.1
        self.p = nn.functional.sigmoid(self.p_logit)

        # Check if batch size is the same as unif_noise, if not take care
        if self.locked:
            noise = np.random.uniform(size=(x.size(0), x.size(2)))
            noise = np.repeat(noise[:, np.newaxis, :], x.size(1), axis=1)
        else:
            noise = np.random.uniform(size=tuple(x.size()))

        unif_noise = Variable(torch.FloatTensor(noise)).cuda()

        drop_prob = (torch.log(self.p + eps)
                     - torch.log(1 - self.p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

    def sum_n_square(self):
        """Helper function for paramater regularisation
        """
        sum_of_square = 0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square


class Linear_relu(nn.Module):

    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class CurriculumDropout(nn.Module):
    """
    :param
    gamma : temperature I think ??
    p : scheduled probability throughout training, reust ss_prob func
    """
    def __init__(self):
        super(CurriculumDropout, self).__init__()

    def forward(self, x, gamma, p):
        if self.train():
            return (1.-p) * np.exp(-gamma * x) + p
        else:
            return x


class Standout(nn.Module):

    def __init__(self, last_layer, alpha, beta):
        print("<<<<<<<<< THIS IS DEFINETLY A STANDOUT TRAINING >>>>>>>>>>>>>>>")
        super(Standout, self).__init__()
        self.pi = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid()

    def forward(self, previous, current, p=0.5, deterministic=False):
        # Function as in page 3 of paper: Variational Dropout
        self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)
        self.mask = sample_mask(self.p)

        # Deterministic version as in the paper
        if (deterministic or torch.mean(self.p).data.cpu().numpy() == 0):
            return self.p * current
        else:
            return self.mask * current


def sample_mask(p):
    """Given a matrix of probabilities, this will sample a mask in PyTorch."""

    if torch.cuda.is_available():
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1).cuda())
    else:
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1))
    mask = uniform < p

    if torch.cuda.is_available():
        mask = mask.type(torch.cuda.FloatTensor)
    else:
        mask = mask.type(torch.FloatTensor)

    return mask


"""https://github.com/kondiz/fraternal-dropout/blob/master/model.py"""

class FraternalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(FraternalDropout, self).__init__()
        self.dropout = dropout
        self.mask = None

    def forward(self, draw_mask, input):
        if self.training == False:
            return input
        if self.mask is None or draw_mask == True:
            self.mask = input.data.new().resize_(input.size()).bernoulli_(1 - self.dropout) / (1 - self.dropout)
        mask = Variable(self.mask)
        masked_input = mask * input
        return masked_input


class FraternalEmbeddedDropout(nn.Module):
    def __init__(self, embed, dropout=0.5):
        super(FraternalEmbeddedDropout, self).__init__()
        self.dropout = dropout
        self.e = embed
        w = getattr(self.e, 'weight')
        del self.e._parameters['weight']
        self.e.register_parameter('weight_raw', nn.Parameter(w.data))

    def _setweights(self):
        raw_w = getattr(self.e, 'weight_raw')
        if self.training:
            mask = raw_w.data.new().resize_((raw_w.size(0), 1)).bernoulli_(1 - self.dropout).expand_as(raw_w) / (
                        1 - self.dropout)
            w = Variable(mask) * raw_w
            setattr(self.e, 'weight', w)
        else:
            setattr(self.e, 'weight', Variable(raw_w.data))

    def forward(self, draw_mask, *args):
        if draw_mask or self.training == False:
            self._setweights()
        return self.e.forward(*args)


# https://github.com/ucla-vision/information-dropout
def sample_lognormal(mean, sigma=None, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backprogpagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)
    """
    e = torch.random_normal(mean.size(), mean = 0., stddev = 1.)
    return torch.exp(mean + sigma * sigma0 * e)


def information_dropout(inputs, stride = 2, max_alpha = 0.7, sigma0 = 1.):
    """
    An example layer that performs convolutional pooling
    and information dropout at the same time.
    """
    num_outputs = inputs.get_shape()[-1]
    # Creates a convolutional layer to compute the noiseless output
    network = F.conv2d(inputs,
        num_outputs=num_outputs,
        kernel_size=3,
        activation_fn=F.relu,
        stride=stride)
    # Computes the noise parameter alpha for the new layer based on the input
    alpha = max_alpha * F.conv2d(inputs, num_outputs=num_outputs, kernel_size=3,
                        stride=stride, activation_fn=F.sigmoid, scope='alpha')
    # Rescale alpha in the allowed range and add a small value for numerical stability
    alpha = 0.001 + max_alpha * alpha
    # Similarly to variational dropout we renormalize so that
    # the KL term is zero for alpha == max_alpha
    kl = - torch.log(alpha/(max_alpha + 0.001))
    e = sample_lognormal(mean=torch.zeros_like(network), sigma=alpha, sigma0=sigma0)
    # Noisy output of Information Dropout
    return network * e


def show_drop_probs(model, dropout_position):
    if dropout_position == 1:
        print("drop-in {}".format(model.drop_in.p))
    elif dropout_position == 2:
        print("drop-out {}".format(model.drop_out.p))
    elif dropout_position == 3:
        print("drop-in {} \t drop-out {}".format(model.drop_in.p, model.drop_out.p))


if __name__ == "__main__":

    V = 50
    h = 4
    bptt = 10
    batch_size = 2

    e = nn.Embedding(V, h)
    f = nn.Embedding(V, h)
    f.weight.data = e.weight.data.clone()
    embed_drop = FraternalEmbeddedDropout(f)

    words = np.random.random_integers(low=0, high=V - 1, size=(batch_size, bptt))
    words = torch.LongTensor(words)
    words = Variable(words)

    print("0")
    print(e(words))
    embed_drop.eval()
    print("1 - should be the same as 0")
    print(embed_drop(True, words))
    print("2 - should be the same as 1")
    print(embed_drop(False, words))
    embed_drop.train()
    print("3 - should be different than 2")
    print(embed_drop(True, words))
    print("4 - should be different than 3")
    print(embed_drop(True, words))
    print("5 - should be the same as 4")
    print(embed_drop(False, words))
    embed_drop.eval()
    print("6 - should be the same as 0")
    print(embed_drop(False, words))
    print("7 - should be the same as 0")
    print(embed_drop(True, words))
