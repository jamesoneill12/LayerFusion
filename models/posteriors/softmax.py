"""Retrieves the chosen approximate softmax"""

from torch import nn
from models.posteriors.adasoftmax import AdaptiveSoftmax
from models.posteriors.hsoftmax import HierarchicalSoftmax, HierarchicalSoftmaxMixture
from models.samplers.sampled_softmax import SampledSoftmax
from models.posteriors.dsoftmax import DifferentiatedSoftmax
from models.posteriors.softmax_mixture import SoftmaxMixture
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from models.loss.nce import NCELoss


def get_softmax(approx_softmax, nhid, ntoken, temp = 2.2, softmax_nsampled=100,
                cutoff=[2000, 10000], mix_num=5, tune=True, noise=None):
    if approx_softmax == "adasoftmax":
        decoder = AdaptiveSoftmax(nhid, [*cutoff, ntoken + 1])
    elif approx_softmax == "soft_mix":
        decoder = SoftmaxMixture(nhid, ntoken, mix_num=mix_num, tune=tune)
    elif approx_softmax == "hsoftmax":
        decoder = HierarchicalSoftmax(ntoken, nhid)
    elif approx_softmax == "relaxed_hsoftmax":
        decoder = HierarchicalSoftmax(ntoken, nhid, rs=True, temp=temp)
    elif approx_softmax == "relaxed_softmax":
        decoder = RelaxedOneHotCategorical
    elif approx_softmax == "sampled_softmax":
        decoder = SampledSoftmax(ntokens=ntoken, nsampled=softmax_nsampled,
                                 nhid=nhid, tied_weight=None).cuda()

    elif approx_softmax == "hsoft_mix":
        decoder = HierarchicalSoftmaxMixture(ntoken, nhid, mix_num=mix_num, tune=False)
    elif approx_softmax == "hsoft_mix_tuned":
        decoder = HierarchicalSoftmaxMixture(ntoken, nhid, mix_num=mix_num, tune=tune)
    elif approx_softmax == "dsoftmax":
        decoder = DifferentiatedSoftmax()
    elif approx_softmax == "dsoft_mix":
        decoder = DifferentiatedSoftmax(ntoken, nhid, mix_num=mix_num, tune=tune)
    elif approx_softmax == "relaxed_ecoc":
        # FIX TOMORROW MORNING
        decoder = LogitRelaxedBernoulli
    elif approx_softmax == "ecoc_mix":
        # token in this case is the dimension of the latent code
        decoder = SoftmaxMixture(nhid, ntoken, mix_num, tune=False)
    elif approx_softmax == "ecoc_mix_tuned":
        # token in this case is the dimension of the latent code
        decoder = SoftmaxMixture(nhid, ntoken, mix_num, tune=True)
    elif approx_softmax == "nce":
        if noise is None:
            import torch
            print("Adding uniform noise as no noise provided")
            noise = torch.randn(nhid, ntoken)
        decoder = NCELoss(noise)
    else:
        decoder = nn.Linear(nhid, ntoken)
    return decoder


def perform_softmax(approx_softmax, decoder, target=None):
    if approx_softmax == "adasoftmax":
        if target is None:
            ValueError("You must pass targets to forward when using adaptive softmax")
        return decoder.set_target(target.data)
    elif approx_softmax == "hsoftmax":
        decoder = HierarchicalSoftmax()
    elif approx_softmax == "dsoftmax":
        decoder = DifferentiatedSoftmax()
    elif approx_softmax == "nce":
        decoder = NCELoss()
    return decoder

