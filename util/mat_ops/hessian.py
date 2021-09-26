import torch
from torch.autograd import Variable
import numpy as np


def mean_kl_divergence(observations, model):
    """
    Returns an estimate of the average KL divergence between a given model and self.policy_model
    """
    observations_tensor = torch.cat(
        [Variable(torch.Tensor(observation)).unsqueeze(0) for observation in observations])
    actprob = model(observations_tensor).detach() + 1e-8
    old_actprob = model(observations_tensor)
    return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()


def hessian_vector_product(model, vector, cg_damping):
    """
    Returns the product of the Hessian of the KL divergence and the given vector
    """
    model.zero_grad()
    mean_kl_div = mean_kl_divergence(model)
    kl_grad = torch.autograd.grad(
        mean_kl_div, model.parameters(), create_graph=True)
    kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
    grad_vector_product = torch.sum(kl_grad_vector * vector)
    grad_grad = torch.autograd.grad(
        grad_vector_product, model.parameters())
    fisher_vector_product = torch.cat(
        [grad.contiguous().view(-1) for grad in grad_grad]).data
    return fisher_vector_product + (cg_damping * vector.data)


def conjugate_gradient(b, cg_iters, residual_tol):
    """
    Returns F^(-1)b where F is the Hessian of the KL divergence
    """
    p = b.clone().data
    r = b.clone().data
    x = np.zeros_like(b.data.cpu().numpy())
    rdotr = r.double().dot(r.double())
    for _ in range(cg_iters):
      z = hessian_vector_product(Variable(p)).squeeze(0)
      v = rdotr / p.double().dot(z.double())
      x += v * p.cpu().numpy()
      r -= v * z
      newrdotr = r.double().dot(r.double())
      mu = newrdotr / rdotr
      p = r + mu * p
      rdotr = newrdotr
      if rdotr < residual_tol: break
    return x