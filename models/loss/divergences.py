import torch
from torch import nn
from models.regularizers.dropout import VariationalDropout

def mmd(x, y, alpha=0.1):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    K = torch.exp(- alpha * (rx.t() + rx - 2 * xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2 * yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2 * zz))
    beta = (1. / (x * (y - 1)))
    gamma = (2. / (y * y))
    return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)


# can choose amount of symmetry too, doesnt have to 1/2 and 1/2
def js_loss(p, q, soft_norm=True, square=False, bptt=35,
            w=0.5, vocab_size=10000, bsize=20):
    """
    p - true, q - prediction
    Idea is to approximate the probability distribution by choosing k-top softmax args,
    finding the corresponding true target in k-positions and normalizing their values.

    OR ALTERNATIVELY (probably better solution)
    select elements of Q for where elemnents of P are non-zero and then softmax that Q subset.
    Why ? Minimizing JS between all of P and Q is very costly in practical settings of NLM i.e large vocabularies.
    """

    if soft_norm:
        p = F.softmax(p.type(torch.FloatTensor))
        # Dummy input that HAS to be 2D for the scatter
        # (you can use view(-1,1) if needed)
        y = torch.LongTensor(bsize * bptt, 1).random_() % vocab_size
        # One hot encoding buffer that you create out of the loop
        # and just keep reusing
        y_onehot = torch.FloatTensor(bsize * bptt, vocab_size)
        y_onehot.zero_()
        q = y_onehot.scatter_(1, y, 1)
    else:
        p, q = p / p.sum(), q / q.sum()
    m = 1. / 2 * (p + q)
    # stats.entropy = sum(pk * log(pk / qk), axis=0)
    pmls = p * torch.log((p / m).clamp(min=1e-8))
    qmls = q * torch.log((q / m).clamp(min=1e-8))
    b = torch.mean(qmls.sum(1) * (1 - w) + qmls.sum(1) * w)
    if square:
        b = torch.sqrt(b)
    return b


def adversarial_loss(gen_output, disc_output):
    pass


"""reg controls the amount of regularization, set to 0.01 as default
Because of the large initial learning rate, Variational Dropout can be volatile and lead
to many errors hence need to be careful when updating the values.
"""
def KL(model, reg = 0.01):
    kl = 0
    for name, module in model.named_modules():
        if isinstance(module, VariationalDropout):
            kl += module.kl().sum()
    print(kl)
    return kl * reg

