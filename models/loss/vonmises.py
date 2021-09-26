from torch.autograd import Variable
import torch
import scipy as sp
import numpy as np

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


# def ratio(v, z):
#     # return z/(v-0.5+torch.sqrt((v-0.5)*(v-0.5) + z*z))
#     return z/(v-1+torch.sqrt((v+1)*(v+1) + z*z))

class Logcmk(torch.autograd.Function):
    """
    The exponentially scaled modified Bessel function of the first kind
    """
    @staticmethod
    def forward(ctx, k):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        m = 300
        ctx.save_for_backward(k)
        k = k.double()
        answer = (m/2-1)*torch.log(k) - torch.log(sp.special.ive(m/2-1, k)).cuda() - k - (m/2)*np.log(2*np.pi)
        answer = answer.float()
        return answer

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        k, = ctx.saved_tensors
        m = 300
        # x = -ratio(m/2, k)
        k = k.double()
        x = -((sp.special.ive(m/2, k))/(sp.special.ive(m/2-1,k))).cuda()
        x = x.float()

        return grad_output*Variable(x)


def NLLvMF(outputs, targets, target_embeddings, generator, opt, eval=False):

    #approximation of LogC(m, k)
    def logcmkappox(d, z):
      v = d/2-1
      return torch.sqrt((v+1)*(v+1)+z*z) - (v-1)*torch.log(v-1 + torch.sqrt((v+1)*(v+1)+z*z))

    loss = 0
    cosine_loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)

    logcmk = Logcmk.apply

    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        out_vec_t = generator(out_t)

        kappa_times_mean = out_vec_t
        tar_vec_t = target_embeddings(targ_t)
        tar_vec_t = tar_vec_t.view(-1, tar_vec_t.size(2))

        kappa = out_vec_t.norm(p=2, dim=-1)#*tar_vec_t.norm(p=2,dim=-1)

        tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
        out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

        cosine_loss_t = (1.0-(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1)).masked_select(targ_t.view(-1).ne(PAD)).sum()

        lambda2 = 0.1
        lambda1 = 0.02
        # nll_loss = - logcmk(kappa) + kappa*(lambda2-lambda1*(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1))
        nll_loss = - logcmk(kappa) + torch.log(1+kappa)*(0.2-(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1))
        # nll_loss = logcmkappox(opt.output_emb_size, kappa) + torch.log(1+kappa)*(0.2-(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1))

        loss_t = nll_loss.masked_select(targ_t.view(-1).ne(PAD)).sum()
        loss += loss_t.data[0]
        cosine_loss += cosine_loss_t.data[0]

        if not eval: loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, cosine_loss


def MaxMarginLoss(outputs, targets, target_embeddings, generator, opt, eval=False):

    # compute generations one piece at a time
    loss = 0
    cosine_loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)

    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        out_vec_t = generator(out_t)
        tar_vec_t = target_embeddings(targ_t)
        tar_vec_t = tar_vec_t.view(-1, tar_vec_t.size(2))

        tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
        out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

        target_embeddings.weight.data.copy_(torch.nn.functional.normalize(target_embeddings.weight.data, p=2, dim=-1))
        # target_embeddings_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=-1)
        cos_ihat_j = out_vec_norm_t.matmul(target_embeddings.weight.t())
        # cos_i_j = tar_vec_norm_t.matmul(target_embeddings.weight.t())

        # s_j = cos_ihat_j - tar_vec_norm_t.matmul(target_embeddings.weight.t())
        maxvalues, jmax = torch.max(cos_ihat_j - tar_vec_norm_t.matmul(target_embeddings.weight.t()), dim=-1)

        cos1 = cos_ihat_j.gather(1, targ_t.view(-1, 1)).view(-1)
        cos2 = cos_ihat_j.gather(1, jmax.view(-1, 1)).view(-1)

        lamd = 0.5
        # diff = lamd + torch.acos(cos1) - torch.acos(cos2)
        diff = lamd + cos2 - cos1

        cosine_loss_t = (1-cos1).masked_select(targ_t.view(-1).ne(PAD)).sum()
        loss_t = (diff.max(torch.autograd.Variable(torch.zeros(1).cuda()))).masked_select(targ_t.view(-1).ne(PAD)).sum()

        cosine_loss += cosine_loss_t.data[0]
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, cosine_loss


def CosineLoss(outputs, targets, target_embeddings, generator, opt, eval=False):
    # compute generations one piece at a time
    loss = 0
    true_loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)
    # print outputs
    # print outputs.size()
    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    targets_ones = torch.ones(targets.size()).cuda()
    targets_split_ones = torch.split(targets_ones, opt.max_generator_batches)
    last=False
    for i, (out_t, targ_t, targ_ones) in enumerate(zip(outputs_split, targets_split, targets_split_ones)):
        out_t = out_t.view(-1, out_t.size(2))
        out_vec_t = generator(out_t)
        tar_vec_t = target_embeddings(targ_t)
        tar_vec_t = tar_vec_t.view(-1, tar_vec_t.size(2))

        tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
        out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

        # true_loss_t = (1.0-(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1)).masked_select(targ_t.view(-1).ne(onmt.Constants.PAD)).sum()
        loss_t = (1-(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1)).masked_select(targ_t.view(-1).ne(PAD)).sum()

        # true_loss += true_loss_t.data[0]
        loss += loss_t.data[0]
        #
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, loss

def L2Loss(outputs, targets, target_embeddings, generator, opt, eval=False):
    loss = 0
    other_loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)

    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        out_vec_t = generator(out_t)
        tar_vec_t = target_embeddings(targ_t)
        tar_vec_t = tar_vec_t.view(-1, tar_vec_t.size(2))

        diff = out_vec_t - tar_vec_t
        # abs_loss = torch.abs(diff)
        crit_loss = diff*diff
        loss_t = (crit_loss.sum(dim=-1)).masked_select(targ_t.view(-1).ne(PAD)).sum()
        # abs_loss_t = (abs_loss.sum(dim=-1)).masked_select(targ_t.view(-1).ne(onmt.Constants.PAD)).sum()
        loss += loss_t.data[0]
        # other_loss += abs_loss_t.data[0]

        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, loss


def CrossEntropy(outputs, targets, generator, crit, opt, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = generator(out_t)
        loss_t = crit(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(PAD).data).sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct



def NormalizedMSELoss(outputs, targets, target_embeddings, generator, opt, eval=False):
    loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)
    # print outputs
    # print outputs.size()
    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)

    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        out_vec_t = generator(out_t)
        tar_vec_t = target_embeddings(targ_t)
        tar_vec_t = tar_vec_t.view(-1, tar_vec_t.size(2))

        tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
        out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

        loss_t = torch.sqrt(crit(out_vec_norm_t, tar_vec_norm_t)).sum(dim=1).masked_select(targ_t.view(-1).ne(PAD)).sum()
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, 0.0
