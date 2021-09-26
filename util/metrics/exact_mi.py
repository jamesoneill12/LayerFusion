import torch
import numpy as np

eps = 1e-07

# sampling with replacement, without setting batch dimension
def random_indices(n, d): return  (0 - n) * torch.rand(n * d, ) + n

def gather_nd_reshape(t, indices, final_shape):
    h = torch.gather(t, indices)
    return h.reshape(final_shape)


#
# Produce an index tensor that gives a permuted matrix of other samples in
# batch, per sample.
#
# Parameters
# ----------
# batch_size : int
#     Number of samples in the batch.
# d_max : int
#     The number of blocks, or the number of samples to generate per sample.
#
# Deps:
#   numpy
def permute_neighbor_indices(
        batch_size,
        d_max=-1, replace=False, pop=True,
):
    if d_max < 0:
        d_max = batch_size + d_max

    inds = []
    if not replace:
        for i in range(batch_size):
            sub_batch = list(range(batch_size))

            # pop = False includes training sample for echo
            # (i.e. dmax = batch instead of dmax = batch - 1)
            if pop:
                sub_batch.pop(i)
            np.random.shuffle(sub_batch)
            inds.append(list(enumerate(sub_batch[:d_max])))
        return inds

    else:
        for i in range(batch_size):
            inds.append(list(enumerate(
                np.random.choice(batch_size, size=d_max, replace=True)
            )))
        return inds


#
# This function implements the Echo Noise distribution specified in:
#   Exact Rate-Distortion in Autoencoders via Echo Noise
#   Brekelmans et al. 2019
#   https://arxiv.org/abs/1904.07199
#
# Parameters
# ----------
# inputs should be specified as list:
#   [ f(X), s(X) ] with s(X) in log space if calc_log = True
# the flag plus_sx should be:
#   True if logsigmoid activation for s(X)
#   False for softplus (equivalent)
#
# Deps:
#   numpy
#   tensorflow
#   permute_neighbor_indices (above)
#   gather_nd_reshape (above)
#   random_indices (above)
def echo_sample(
        inputs,
        clip=None, d_max=100, batch=100, multiplicative=False, echo_mc=False,
        replace=False, fx_clip=None, plus_sx=True, calc_log=True,
        return_noise=False, **kwargs
):
    # kwargs unused

    if isinstance(inputs, list):
        fx = inputs[0]
        sx = inputs[-1]
    else:
        fx = inputs

    # TO DO : CALC_LOG currently determines both whether to do log space calculations AND whether sx is a log

    fx_shape = fx.get_shape()
    sx_shape = sx.get_shape()

    # clip is multiplied times s(x) to ensure that sum of truncated terms < machine precision
    # clip should be calculated numerically according to App C in paper
    # M (r ^ dmax / 1-r ) < precision, SOLVE for r (clipping factor), with M = max magnitude of f(x)

    # calculation below is an approximation (ensuring only term d_max + 1 < precision)
    if clip is None:
        max_fx = fx_clip if fx_clip is not None else 1.0
        clip = (2 ** (-23) / max_fx) ** (1.0 / d_max)

    # fx_clip can be used to restrict magnitude of f(x), not used in paper
    # defaults to no clipping and M = 1 (e.g. with tanh activation for f(x))
    if fx_clip is not None:
        torch._clamp(fx, -fx_clip, fx_clip)

    if not calc_log:
        sx = torch.mul(clip, sx)
        sx = torch.where(torch.abs(sx) < eps, eps * torch.sign(sx), sx)
    else:
        # plus_sx based on activation for sx = s(x):
        #   True for log_sigmoid
        #   False for softplus
        sx = torch.log(clip) + (-1 * sx if not plus_sx else sx)

    if echo_mc is not None:
        # use mean centered fx for noise
        fx = fx - fx.mean(dim=0, keepdim=True)

    z_dim = fx.size()[-1]

    if replace:  # replace doesn't set batch size (using permute_neighbor_indices does)
        batch = fx.size[0]
        sx = sx.flatten() if len(sx_shape) > 2 else sx
        fx = fx.flatten() if len(fx_shape) > 2 else fx
        inds = torch.reshape(random_indices(batch, d_max), (-1, 1))
        select_sx = gather_nd_reshape(sx, inds, (-1, d_max, z_dim))
        select_fx = gather_nd_reshape(fx, inds, (-1, d_max, z_dim))

        if len(sx_shape) > 2:
            select_sx = torch.expand_dims(torch.expand_dims(select_sx, 2), 2)
            sx = torch.expand_dims(torch.expand_dims(sx, 1), 1)
        if len(fx_shape) > 2:
            select_fx = torch.expand_dims(torch.expand_dims(select_fx, 2), 2)
            fx = torch.expand_dims(torch.expand_dims(fx, 1), 1)

    else:
        # batch x batch x z_dim
        # for all i, stack_sx[i, :, :] = sx
        repeat = torch.mul(torch.ones_like(torch.expand_dims(fx, 0)), torch.ones_like(torch.expand_dims(fx, 1)))
        stack_fx = torch.mul(fx, repeat)
        stack_sx = torch.mul(sx, repeat)

        # select a set of dmax examples from original fx / sx for each batch entry
        inds = permute_neighbor_indices(batch, d_max, replace=replace)

        # note that permute_neighbor_indices sets the batch_size dimension != None
        # this necessitates the use of fit_generator, e.g. in training to avoid 'remainder' batches if data_size % batch > 0

        select_sx = torch.gather(stack_sx, inds)
        select_fx = torch.gather(stack_fx, inds)

    if calc_log:
        sx_echoes = torch.cumsum(select_sx, axis=1, exclusive=True)
    else:
        sx_echoes = torch.cumprod(select_sx, axis=1, exclusive=True)

    # calculates S(x0)S(x1)...S(x_l)*f(x_(l+1))
    sx_echoes = torch.exp(sx_echoes) if calc_log else sx_echoes
    fx_sx_echoes = torch.mul(select_fx, sx_echoes)

    # performs the sum over dmax terms to calculate noise
    noise = fx_sx_echoes.sum(dim=1)

    if multiplicative:
        # unused in paper, not extensively tested
        sx = sx if not calc_log else torch.exp(sx)
        output = torch.exp(fx + torch.mul(sx, noise))  # tf.multiply(fx, tf.multiply(sx, noise))
    else:
        sx = sx if not calc_log else torch.exp(sx)
        output = fx + torch.mul(sx, noise)

    sx = sx if not calc_log else torch.exp(sx)

    if multiplicative:  # log z according to echo
        output = torch.exp(fx + torch.mul(sx, noise))
    else:
        output = fx + torch.mul(sx, noise)

    return output if not return_noise else noise


#
# This function implements the Mutual Information penalty (via Echo Noise)
# which was specified in:
#
#   Exact Rate-Distortion in Autoencoders via Echo Noise
#   Brekelmans et al. 2019
#   https://arxiv.org/abs/1904.07199
#
# Parameters
# ----------
# inputs: tensor (or list of tensors [f(x), s(x)])
#   The sigmoid outputs from an encoder (don't include the mean outputs).
#
# clip : scales s(x) to ensure that sum of truncated terms < machine precision
#   clip should be calculated numerically according to App C in paper
#   Solve for r:  M (r ^ dmax / 1-r ) < machine precision, with M = max magnitude of f(x)
#
# calc_log : bool whether inputs are in log space
# plus_sx : bool whether inputs measure s(x) (vs. -s(x) if parametrized with softplus, e.g.)

def echo_loss(inputs, clip=0.8359, calc_log=True, plus_sx=True, multiplicative=False, **kwargs):
    if isinstance(inputs, list):
        z_mean = inputs[0]
        z_scale = inputs[-1]
    else:
        z_scale = inputs

    mi = -torch.log(torch.abs(clip * z_scale) + torch.epsilon()) if not calc_log else -(
                torch.log(clip) + (z_scale if plus_sx else -z_scale))

    return mi

