import torch


def get_jacobian(net, x, noutputs):
    x = x.squeeze()
    n = x.size()[0]
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    print(x.size())
    y = net(x)
    y.backward(torch.eye(noutputs))
    return x.grad.data


if __name__ == "__main__":

    from torch import nn
    net, x, noutputs = nn.Linear(100, 100), torch.randn(100), 100
    out_j = get_jacobian(net, x, noutputs)