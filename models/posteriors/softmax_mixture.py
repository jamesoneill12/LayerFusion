import torch
from torch import nn
import torch.nn.functional as F


class SoftmaxMixture(nn.Module):

    """
    Assigns a mixture of linear layers to a softmax by
    looking up seperate linear layers from an nn.Embedding layer where index of the embedding
    depends on the term frequency. This means there is a different linear layer depdending
    on the word frequency.

    :param

    """

    def __init__(self, latent_size, vocab_size, mix_num = 5, tune=False):
        super().__init__()

        self.mix_num = mix_num
        self.vsize = vocab_size
        self.linear = nn.ModuleList([nn.Linear(latent_size, vocab_size) for _ in range(mix_num)])
        if tune:
            self.p = nn.Parameter(torch.ones(mix_num))
        else:
            self.p = None

    def forward(self, xs):
        # My implementation of softmax mixtures
        x = torch.cat([self.linear[i](xs) for i in range(self.mix_num)], 2)
        # x = F.softmax(x, dim=2)
        x = x.view(x.size(0), x.size(1), int(x.size(2)/self.mix_num), self.mix_num)
        if self.p is not None:
            x = x * F.softmax(self.p)
            x = x.sum(dim=3)
        else:
            x = torch.mean(x, 3)
        # weighted average here is tunable controls the weights
        return x

    def log_forward(self, context):
        log_priors = F.log_softmax(context[:, -self.mix_num:]).unsqueeze(2)
        print(log_priors.size())
        log_mixes = []
        for i in range(self.mix_num):
            print(i)
            print(i * self.vsize)
            print((i + 1) * self.vsize)
            print(context.size())
            scale =  F.softmax(context[:, i * self.vsize: (i + 1) * self.vsize])
            print(i)
            log_mixes.append(log_priors[:, i].unsqueeze(1) * scale)
            print(i)

        """
        indiv_log_mixtures =  [log_priors[:, i].unsqueeze(1) * F.softmax(context[:, i * self.vsize : (i + 1) * self.vsize])
                                          for i in range(self.mix_num)]
         """
        log_mixtures = torch.stack(log_mixes, 1)
        out = torch.log(torch.exp(log_priors + log_mixtures).sum(1))
        return out


if __name__ == "__main__":

    latent_size = 200
    vocab_size = 1000
    mix_num = 5
    dims = (35, 20, 200)

    x = torch.randn(*dims)

    sm = SoftmaxMixture(latent_size, vocab_size, mix_num = mix_num, tune=False)

    #sm.log_forward(x)
    sm.forward(x)