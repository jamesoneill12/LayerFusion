import torch
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from torch.optim.optimizer import Optimizer


# decoder_optim = torch.optim.Adam(model.decoder.parameters(), amsgrad=True)
# decoder_optim = torch.optim.Adam(model.decoder.parameters())
# decoder_optim = torch.optim.SGD(model.encoder.parameters(), lr=args.lr)


def get_optimizer(params, optimizer, lr):
    if optimizer == 'amsgrad': return torch.optim.Adam(params, lr=lr, amsgrad=True)
    elif optimizer == 'adam': return torch.optim.Adam(params, lr=lr)
    elif optimizer == 'adamw': return AdamW(params, lr=lr, weight_decay=1e-5)
    elif optimizer == 'sgd' or optimizer is None: return torch.optim.SGD(params, lr=lr)


def get_scheduler(args, optim, train_data=None):
    # if None, do the original annealing from lr = 20.
    if args.scheduler == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20000, eta_min=0, last_epoch=-1)
    elif args.scheduler == 'cosine_restart':
        scheduler = CosineLRWithRestarts(optim, args.batch_size, args.epochs, restart_period=5, t_mult=1.2)
    elif args.scheduler == 'lro':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min')
    elif args.scheduler == 'multi_step':
        max_epochs = args.epochs * train_data.size(0) / args.bptt
        mstones = list(range(0, max_epochs, max_epochs / 10))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=mstones, gamma=0.1)
    return scheduler


"""
def get_scheduler(optim, args, train_size = None):
    if args.scheduler == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20000, eta_min=0, last_epoch=-1)
    elif args.scheduler == 'lro':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min')
    elif args.scheduler == 'multi_step':
        if train_size is None:
            ValueError("Need to provide train data as argument")
        max_epochs = args.epochs * train_size / args.bptt
        mstones = list(range(0, max_epochs, max_epochs / 10))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=mstones, gamma=0.1)
    return scheduler
"""

def sgd_update(pretrained, params, lr):
    if pretrained is not None:
        for name, p in params:
            if name is not 'encoder.weight':
                p.data.add_(-lr, p.grad.data)
    else:
        for p in params:
            p.data.add_(-lr, p.grad.data)


"""Optimizer for Actor Critic Training"""
class Optim(object):
    def __init__(self, params, lr, is_pre,
                 grad_clip, new_lr=0.0, weight_decay=0.):
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(
            0.9, 0.98), eps=1e-09, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.params = params
        if is_pre:
            self.step = self.pre_step
        else:
            assert new_lr != 0.0

            self.n_current_steps = 0
            self.new_lr = new_lr
            self.step = self.train_step

    def train_step(self):
        self.optimizer.step()

        self.n_current_steps += 1
        if self.n_current_steps == 1e6:
            self.update_learning_rate()

    def pre_step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm(self.params, self.grad_clip)

    def update_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.new_lr


class Policy_optim(Optim):
    def __init__(self, params, lr, grad_clip, new_lr):
        super().__init__(params, lr, False, grad_clip, new_lr)

    def train_step(self, reward):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.grad = p.grad.mul(reward)

        self.optimizer.step()

        self.n_current_steps += 1
        if self.n_current_steps == 1e6:
            self.update_learning_rate()


# Non-centered RMSprop update with shared statistics (without momentum)
class SharedRMSprop(torch.optim.RMSprop):
  def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
    super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)

    # State initialisation (must be done before step, else will not be shared between threads)
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = p.data.new().resize_(1).zero_()
        state['square_avg'] = p.data.new().resize_as_(p.data).zero_()

  def share_memory(self):
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'].share_memory_()
        state['square_avg'].share_memory_()

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        state = self.state[p]

        square_avg = state['square_avg']
        alpha = group['alpha']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], p.data)

        # g = αg + (1 - α)Δθ^2
        square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
        # θ ← θ - ηΔθ/√(g + ε)
        avg = square_avg.sqrt().add_(group['eps'])
        p.data.addcdiv_(-group['lr'], grad, avg)

    return loss

  
class SharedAdam(torch.optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss







# ------------------ Adam With Warm Restarts ----------------------- #
class AdamW(Optimizer):
    """Implements Adam algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        # super(AdamW, self).__init__(params, defaults)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class CosineLRWithRestarts(_LRScheduler):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will extend/shrink


    Example:
        >>> scheduler = CosineLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, eta_threshold=1000, verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'],
                                 optimizer.param_groups))

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.iteration = 0
        self.epoch_size = epoch_size
        self.eta_threshold = eta_threshold
        self.t_mult = t_mult
        self.verbose = verbose
        self.base_weight_decays = list(map(lambda group: group['weight_decay'],
                                           optimizer.param_groups))
        self.restart_period = restart_period
        self.restarts = 0
        self.t_epoch = -1
        self.batch_increments = []
        self._set_batch_increment()

    def _schedule_eta(self):
        """
        Threshold value could be adjusted to shrink eta_min and eta_max values.
        """
        eta_min = 0
        eta_max = 1
        if self.restarts <= self.eta_threshold:
            return eta_min, eta_max
        else:
            d = self.restarts - self.eta_threshold
            k = d * 0.09
            return (eta_min + k, eta_max - k)

    def get_lr(self, t_cur):
        eta_min, eta_max = self._schedule_eta()

        eta_t = (eta_min + 0.5 * (eta_max - eta_min)
                 * (1. + math.cos(math.pi *
                                  (t_cur / self.restart_period))))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))
        lrs = [base_lr * eta_t for base_lr in self.base_lrs]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart at epoch {}".format(self.last_epoch))
            self.restart_period *= self.t_mult
            self.restarts += 1
            self.t_epoch = 0

        return zip(lrs, weight_decays)

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = list(np.linspace(0, 1, batches_in_epoch))

    def step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        self.batch_step()

    def batch_step(self):
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self.iteration += 1
        except (IndexError):
            raise RuntimeError("Epoch size and batch size used in the "
                               "training loop and while initializing "
                               "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay


'''A wrapper class for optimizer '''


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr