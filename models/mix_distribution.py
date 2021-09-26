from torch.distributions import Categorical
from torch.autograd import Variable
from models.dropout import ConcreteDropout, Standout
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch

scale = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
transform = transforms.Compose([scale])


t_normalize = lambda x, z: z.mean() * (x - x.mean()) / (x.max() - x.min())
t_ratio = lambda x, z:  x / z


def mix_dist(x, z, mixture='linear', p=None,  disc_rep=False, eps=0.01):
    if mixture in ['exp', 'linear', 'sigmoid', 'static']:
        if p is None:
            raise Exception('p arguement in None, should be provided when using a {} schedule'.format(mixture))
        x_n, z_n = random_mix(x, z, p, disc_rep=disc_rep, eps=eps)

    elif mixture == ['gumbel', 'logits', 'standout']:
        x_n, z_n = 1, 1

    # print("{:.2f} change in x at rate {}".format(float((x == x_n).sum())/x.numel(), p))
    return x_n, z_n


def random_mix(x, z, p, disc_rep=False, eps=0, scale=True):

    mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(p))).type(torch.ByteTensor)
    # https://discuss.pytorch.org/t/use-torch-nonzero-as-index/33218 for having to use split()
    mask_inds = (mask == 1).nonzero().split(1, dim=1)
    if eps != 0: u = torch.randn(z.data[mask_inds].size()) * eps
    if disc_rep:
        x[mask],z[mask] = z[mask], x[mask]
        # assert int(torch.all(torch.eq(x, z))) == 0
        if eps != 0:
            x.data[mask_inds] += u
            z.data[mask_inds] += u
    else:
        z.data[mask_inds] = t_normalize(x.data[mask_inds], z.data[mask_inds])
        if eps != 0:
            z.data[mask_inds] += u
    return x, z


def test_random_mix():
    x = torch.randn((2, 3))
    z = torch.randn((2, 3))
    x_new, z_new = random_mix(x, z, p=0.5, disc_rep=True)
    #assert int(torch.all(torch.eq(x, x_new))) == 0


class MixConcrete(nn.Module):

    def __init__(self, layer, input_shape, w_reg=1e-6, d_reg=1e-5):
        super(MixConcrete, self).__init__()

        self.conc = ConcreteDropout(layer, input_shape, weight_regularizer=w_reg, locked = True,
                 dropout_regularizer=d_reg, init_min=0.1, init_max=0.1)

    def forward(self, x, z):
        x_c = self.conc(x)


class MixStandout(nn.Module):

    def __init__(self, l_shape, alpha, beta=1):
        super(MixStandout, self).__init__()

        self.linear = nn.Linear(l_shape)
        self.standout = Standout(self.linear, alpha, beta)

    def forward(self, x, z, deterministic = False):
        x_sig = F.relu(self.linear(x))
        # this would be (x, x_sig)
        z_tilde = self.standout(x_sig, z)
        return z_tilde


class InfoMix(nn.Module):

    def __init__(self, lam=1e-3):
        super(InfoMix, self).__init__()
        self.lam = lam

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def forward(self, x, z, l = 0.5):
        """Minimizes the Shannon-divergence between """
        o12 = F.kl_div(F.log_softmax(x), z)
        o21 = F.kl_div(F.log_softmax(z), x)
        out = self.lam * torch.mean((o12 * l + o21 * (1-l)))
        return out


def mix_z(x, z, p):
    ":params x: data input, z: random input,  p: mix prob vector for each column "
    print(p)
    print(Categorical(p).sample())
    x.view()


if __name__ == "__main__":

    test_random_mix()

    def ex():
        x = torch.randn((2, 3, 4, 4))
        sf = 0.5
        # print(x)
        x_mean = x.view(x.size(0), x.size(1), -1).mean(2)
        x_std = x.view(x.size(0), x.size(1), -1).std(2)
        print(x_mean.size())
        print(x_std.size())

        x = (x.view(x.size(0), x.size(1), -1) / x_mean) * sf
        print(x)
