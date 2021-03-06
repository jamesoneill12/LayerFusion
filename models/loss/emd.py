import torch
import numpy as np


class WassersteinLossVanilla(torch.autograd.Function):
    def __init__(self, cost, lam=1e-3, sinkhorn_iter=50):
        super(WassersteinLossVanilla, self).__init__()

        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost / self.lam)
        self.KM = self.cost * self.K
        self.stored_grad = None

    def forward(self, pred, target):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        assert pred.size(1) == self.na
        assert target.size(1) == self.nb

        class WassersteinLossStab(torch.autograd.Function):
            def __init__(self, cost, lam=1e-3, sinkhorn_iter=50):
                super(WassersteinLossStab, self).__init__()

                # cost = matrix M = distance matrix
                # lam = lambda of type float > 0
                # sinkhorn_iter > 0
                # diagonal cost should be 0
                self.cost = cost
                self.lam = lam
                self.sinkhorn_iter = sinkhorn_iter
                self.na = cost.size(0)
                self.nb = cost.size(1)
                self.K = torch.exp(-self.cost / self.lam)
                self.KM = self.cost * self.K
                self.stored_grad = None

            def forward(self, pred, target):
                """pred: Batch * K: K = # mass points
                   target: Batch * L: L = # mass points"""
                assert pred.size(1) == self.na
                assert target.size(1) == self.nb

                batch_size = pred.size(0)

                log_a, log_b = torch.log(pred), torch.log(target)
                log_u = self.cost.new(batch_size, self.na).fill_(-np.log(self.na))
                log_v = self.cost.new(batch_size, self.nb).fill_(-np.log(self.nb))

                for i in range(self.sinkhorn_iter):
                    log_u_max = torch.max(log_u, dim=1, keepdim=True).values
                    u_stab = torch.exp(log_u - log_u_max)
                    log_v = log_b - torch.log(torch.mm(self.K.t(), u_stab.t()).t()) - log_u_max
                    log_v_max = torch.max(log_v, dim=1, keepdim=True).values
                    v_stab = torch.exp(log_v - log_v_max)
                    log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max

                log_v_max = torch.max(log_v, dim=1, keepdim=True).values
                v_stab = torch.exp(log_v - log_v_max)
                logcostpart1 = torch.log(torch.mm(self.KM, v_stab.t()).t()) + log_v_max
                wnorm = torch.exp(log_u + logcostpart1).mean(0).sum()  # sum(1) for per item pair loss...
                grad = log_u * self.lam
                grad = grad - grad.mean(dim=1, keepdim=True)
                grad = grad - grad.mean(dim=1, keepdim=True)  # does this help over only once?
                grad = grad / batch_size

                self.stored_grad = grad

                return self.cost.new((wnorm,))

            def backward(self, grad_output):
                # print (grad_output.size(), self.stored_grad.size())
                # print (self.stored_grad, grad_output)
                res = grad_output.new()
                res.resize_as_(self.stored_grad).copy_(self.stored_grad)
                if grad_output[0] != 1:
                    res.mul_(grad_output[0])
                return res, None

        nbatch = pred.size(0)

        u = self.cost.new(nbatch, self.na).fill_(1.0 / self.na)

        for i in range(self.sinkhorn_iter):
            v = target / (torch.mm(u, self.K.t()))  # double check K vs. K.t() here and next line
            u = pred / (torch.mm(v, self.K))
            # print ("stability at it",i, "u",(u!=u).sum(),u.max(),"v", (v!=v).sum(), v.max())
            if (u != u).sum() > 0 or (v != v).sum() > 0 or u.max() > 1e9 or v.max() > 1e9:  # u!=u is a test for NaN...
                # we have reached the machine precision
                # come back to previous solution and quit loop
                raise Exception(str(
                    ('Warning: numerical errrors', i + 1, "u", (u != u).sum(), u.max(), "v", (v != v).sum(), v.max())))

        loss = (u * torch.mm(v, self.KM.t())).mean(0).sum()  # double check KM vs KM.t()...
        grad = self.lam * u.log() / nbatch  # check whether u needs to be transformed
        grad = grad - torch.mean(grad, dim=1, keepdim=True)
        grad = grad - torch.mean(grad, dim=1, keepdim=True)  # does this help over only once?
        self.stored_grad = grad

        dist = self.cost.new((loss,))
        return dist

    def backward(self, grad_output):
        # print (grad_output.size(), self.stored_grad.size())
        return self.stored_grad * grad_output[0], None