"""A generic NCE wrapper which speedup the training and inferencing"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.loss.alias_multinomial import AliasMultinomial

# https://github.com/Stonesjtu/Pytorch-NCE/blob/master/nce/nce_loss.py


class NCELoss(nn.Module):
    """Noise Contrastive Estimation

    NCE is to eliminate the computational cost of softmax
    normalization.

    There are two modes in this NCELoss module:
        - nce: enable the NCE approximtion
        - ce: use the original cross entropy as default loss
    They can be switched by calling function `enable_nce()` or
    `disable_nce()`, you can also switch on/off via `nce_mode(True/False)`

    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.pg.uk/download/pdf/42338485.pdf

    Attributes:
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        size_average: average the loss by batch size
        reduce: returned the loss for each target_idx if True,
        this will ignore the value of `size_average`
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported

    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: :math:`(B, N)` if `reduce=True`

    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module

    Return:
        loss: if `reduce=False` the scalar NCELoss Variable ready for backward,
        else the loss matrix for every individual targets.

    Shape:
    """

    def __init__(self,
                 noise,
                 noise_ratio=100,
                 norm_term='auto',
                 reduction='elementwise_mean',
                 per_word=False,
                 loss_type='nce',
                 ):
        super(NCELoss, self).__init__()

        self.register_buffer('noise', noise)
        self.alias = AliasMultinomial(noise)
        self.noise_ratio = noise_ratio
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term
        self.reduction = reduction
        self.per_word = per_word
        self.bce = nn.BCELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type

    def forward(self, target, *args, **kwargs):
        """compute the loss with output and the desired target

        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """

        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':

            noise_samples = self.get_noise(batch, max_len)

            # B,N,Nr
            prob_noise = Variable(
                self.noise[noise_samples.data.view(-1)].view_as(noise_samples)
            )
            prob_target_in_noise = Variable(
                self.noise[target.data.view(-1)].view_as(target)
            )

            # (B,N), (B,N,Nr)
            prob_model, prob_noise_in_model = self._get_prob(target, noise_samples, *args, **kwargs)

            if self.loss_type == 'nce':
                if self.training:
                    loss = self.nce_loss(
                        prob_model, prob_noise_in_model,
                        prob_noise, prob_target_in_noise,
                    )
                else:
                    # directly output the approximated posterior
                    loss = - prob_model.log()
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )

            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError('loss type {} not implemented at {}'.format(self.loss_type, current_stage))

        else:
            # Fallback into conventional cross entropy
            loss = self.ce_loss(target, *args, **kwargs)

        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        if self.per_word:
            noise_samples = self.alias.draw(
                batch_size,
                max_len,
                self.noise_ratio,
            )
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(batch_size, max_len, self.noise_ratio)

        noise_samples = Variable(noise_samples).contiguous()
        return noise_samples

    def _get_prob(self, target_idx, noise_idx, *args, **kwargs):
        """Get the NCE estimated probability for target and noise

        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        MIN_PROB = 1e-9  # a minimal probability for numerical stability
        target_score, noise_score = self.get_score(target_idx, noise_idx, *args, **kwargs)

        target_prob = target_score.sub(self.norm_term).exp()
        target_prob.data.clamp_(MIN_PROB, 1)
        noise_prob = noise_score.sub(self.norm_term).exp()
        noise_prob.data.clamp_(MIN_PROB, 1)
        return target_prob, noise_prob

    def get_score(self, target_idx, noise_idx, *args, **kwargs):
        """Get the target and noise scores given input

        This method should be override by inherit classes

        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        raise NotImplementedError()

    def ce_loss(self, target_idx, *args, **kwargs):
        """Get the conventional CrossEntropyLoss

        The returned loss should be of the same size of `target`

        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class

        Returns:
            - loss: the estimated loss for each target
        """
        raise NotImplementedError()

    def nce_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """Compute the classification loss given all four probabilities

        Args:
            - prob_model: probability of target words given by the model (RNN)
            - prob_noise_in_model: probability of noise words given by the model
            - prob_noise: probability of noise words given by the noise distribution
            - prob_target_in_noise: probability of target words given by the noise distribution

        Returns:
            - loss: a mis-classification loss for every single case
        """

        p_model = torch.cat([prob_model.unsqueeze(2), prob_noise_in_model], dim=2)
        p_noise = torch.cat([prob_target_in_noise.unsqueeze(2), prob_noise], dim=2)

        # predicted probability of the word comes from true data distribution
        p_true = p_model / (p_model + self.noise_ratio * p_noise)
        label = torch.cat(
            [torch.ones_like(prob_model).unsqueeze(2),
             torch.zeros_like(prob_noise)], dim=2
        )

        loss = self.bce(p_true, label).sum(dim=2)

        return loss

    def sampled_softmax_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        logits = torch.cat([prob_model.unsqueeze(2), prob_noise_in_model], dim=2).log()
        q_logits = torch.cat([prob_target_in_noise.unsqueeze(2), prob_noise], dim=2).log()
        # subtract Q for correction of biased sampling
        logits = logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()
        loss = self.ce(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ).view_as(labels)

        return loss


"""An index linear class for generic NCE module"""
class IndexLinear(NCELoss):
    """A linear layer that only decodes the results of provided indices

    Args:
        target_idx: indices of target words
        noise_idx: indices of noise words
        input: input matrix

    Shape:
        - target_idx :math:`(B, N)` where `max(M) <= N` B is batch size
        - noise_idx :math:`(B, N, N_r)` where `max(M) <= N`
        - Input :math:`(B, N, in\_features)`

    Return:
        - target_score :math:`(N, 1)`
        - noise_score :math:`(N, N_r)` the un-normalized score
    """

    def __init__(self, embedding_dim, num_classes, *args, **kwargs):
        super(IndexLinear, self).__init__(*args, **kwargs)
        # use Embedding to store the output embedding
        # it's efficient when it comes sparse update of gradients
        self.emb = nn.Embedding(num_classes, embedding_dim)
        # self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.bias = nn.Embedding(num_classes, 1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.emb.embedding_dim)
        self.emb.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # initialize the bias with unigram instead of uniform
            self.bias.weight.data = torch.log(self.noise + 1e-10) + self.norm_term
            self.bias.weight.data.unsqueeze_(1)

    def get_score(self, target_idx, noise_idx, input):
        """
        Shape:
            - target_idx: :math:`(B, L)` where `B` is batch size
            `L` is sequence length
            - noise_idx: :math:`(B, L, N_r)` where `N_r is noise ratio`
            - input: :math:`(B, L, E)` where `E = output embedding size`
        """

        if self.per_word:
            return self._compute_sampled_logit(
                target_idx, noise_idx, input
            )
        else:
            return self._compute_sampled_logit_batched(
                target_idx, noise_idx, input
            )

    def _compute_sampled_logit(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector

        Args:
            - target_idx: :math:`B, L, 1`
            - noise_idx: :math:`B, L, N_r` target_idx and noise_idx are
            concatenated into one single index matrix for performance
            - input: :math:`(B, L, E)` where `E = vector dimension`

        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """

        # the size will be used to pack the output of indexlinear
        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, 1, input.size(-1))  # N,1,E
        target_idx = target_idx.view(-1).unsqueeze(-1)  # N,1
        noise_idx = noise_idx.view(-1, noise_idx.size(-1))  # N,Nr
        indices = torch.cat([target_idx, noise_idx], dim=-1)

        # the pytorch's [] operator can't BP correctly with redundant indices
        # before version 0.2.0
        # [] operator is much slower than index_select in pytorch-0.4.0

        # index_select is faster than pure embedding look-up which is weird
        # 20it/s vs. 14 it/s
        # target_batch = self.emb(indices)
        # bias = self.bias(indices).squeeze(2)
        target_batch = self.emb.weight.index_select(0, indices.view(-1)).view(*indices.size(), -1)
        bias = self.bias.weight.index_select(0, indices.view(-1)).view_as(indices)
        # the element-wise multiplication is automatically broadcasted
        logits = torch.sum(input * target_batch, dim=2) + bias
        logits = logits.view(*original_size, -1)

        target_score, noise_score = logits[:, :, 0], logits[:, :, 1:]
        return target_score, noise_score

    def _compute_sampled_logit_batched(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector

        A batched version, it speeds up computation and puts less burden on
        sampling methods.

        Args:
            - target_idx: :math:`B, L, 1` flatten to `(N)` where `N=BXL`
            - noise_idx: :math:`B, L, N_r`, noises at the dim along B and L
            should be the same, flatten to `N_r`
            - input: :math:`(B, L, E)` where `E = vector dimension`

        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """

        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)

        target_batch = self.emb(target_idx)
        # target_bias = self.bias.index_select(0, target_idx)  # N
        target_bias = self.bias(target_idx).squeeze(1)  # N
        target_score = torch.sum(input * target_batch, dim=1) + target_bias # N X E * N X E

        noise_batch = self.emb(noise_idx)  # Nr X H
        # noise_bias = self.bias.index_select(0, noise_idx).unsqueeze(0)  # Nr
        noise_bias = self.bias(noise_idx)  # 1, Nr
        noise_score = torch.matmul(
            input, noise_batch.t()
        ) + noise_bias.t()  # N X Nr
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def ce_loss(self, target_idx, input):
        score = F.linear(input, self.emb.weight, self.bias.weight.squeeze(1))  # (N, V)
        loss = self.ce(
            score.view(-1, score.size(-1)),
            target_idx.view(-1)
        ).view_as(target_idx)
        return loss


if __name__ == "__main__":

    emb_dim = 400
    noise = torch.nn.Softmax()
    noise_in = noise(torch.randn(20, 100))
    target = torch.randint(2, (20, ))
    # nce = NCELoss(noise_in)
    # nce(target)

    IndexLinear(emb_dim, noise_in.size(1))