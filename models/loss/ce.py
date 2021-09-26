import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.distributions import Categorical
from util.eval.bleu import get_bleu


class CrossEntropyLoss(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.

    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    log_softmax = nn.LogSoftmax()

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = Variable(torch.FloatTensor(class_weights).cuda())

    def forward(self, logits, target):
        log_probabilities = self.log_softmax(logits)
        # NLLLoss(x, class) = -weights[class] * x[class]
        return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()


class RewardSmoother(nn.Module):
    """
    Reward Smoother smoothens maximum likelihood
    """

    def __init__(self, class_weights, batch_size=None,
                 lm=True, vocab_vectors=None, temp=10):
        """class weights should be ntokens length"""
        super(RewardSmoother, self).__init__()

        self.ce_loss = CrossEntropyLoss(class_weights)
        self.lm = lm
        # use for flattening out decayed exponent
        self.temp = temp
        self.bs = batch_size
        # torch_vocab_vectors = torch.cuda.FloatTensor(list(corpus.dictionary.wv.values()))

        if vocab_vectors is None:
            ValueError("You need to pass torch_vocab_vectors when using this loss to compute rewards")
        else:
            self.vectors = vocab_vectors

    def compute_bleu_reinforce(self, output, predicted, target):
        """
            Computes reward using the reinforce algorithm
        """
        m = Categorical(output)
        action = m.sample()
        # predict_tensor = predicted.view(x_trg.size(0), -1)
        reshape_predicted = predicted.view(output.size(0), output.size(1))
        reshape_target = target.view(output.size(0), output.size(1))
        reward = get_bleu(reshape_predicted.cpu().numpy(), reshape_target.cpu().numpy())
        loss = - m.log_prob(action) * reward
        return loss

    def compute_cosine_reward(self, loss, predicted, target):
        """
        If batch_size is passed then we can normalize reward
        for each sequence before averaging
        """
        predict_vector = self.vectors[predicted]
        target_vector = self.vectors[target]
        reward = F.cosine_similarity(predict_vector, target_vector)
        if self.bs is not None:
            reward = reward.view(self.bs, -1)
            """So here we weight the importance of earlier tokens more than later ones"""
            weights = torch.cuda.FloatTensor(list(reversed(range(reward.size(1)))))
            """temp > 1, otherwise little gain in correct predictions after 3-4 tokens per sequence"""
            norm_weights = F.softmax(torch.exp(weights) / self.temp)
            loss = loss.view(self.bs, -1) * norm_weights
            print("Loss")
            print(loss)
            print("Reward")
            print(reward)
            loss = torch.mean(loss * reward, 1)
        else:
            loss = torch.mean(loss * reward)
        return loss

    def forward(self, output, target):
        _, predicted = torch.max(output.data, 1)
        loss = self.ce_loss(output, target)
        if self.lm is False:
            loss = self.compute_bleu_reinforce(output, target)
        else:
            loss = self.compute_cosine_reward(loss, predicted, target)

        loss = torch.mean(loss)
        return loss


class BinaryCrossEntropyLoss(nn.Module):
    """
    This criterion (`BinaryCrossEntropyLoss`) combines `LogSigmoid` and `NLLLoss` in one single class.
    Therefore, no Softmax used, but instead a Sigmoid for each unit in the output.
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """

    def __init__(self, uni_prob, num_neighbors=10):
        """

        :param class_weights: weights certain classes more than others
        :param uni_prob: counts for getting negative samples OR
                        an int that is the length of the vocabulary
        :param num_neighbors:
        """
        super().__init__()
        if type(uni_prob) == int:
            self.class_weights = Variable(torch.ones(uni_prob)).cuda()
        else:
            if self.samp_prob is not None:
                usamp = uni_prob.pow(0.75)
                self.samp_prob = usamp / usamp.sum()
                assert self.samp_prob.sum().data == 1.0
            else:
                self.samp_prob = Variable(torch.ones(uni_prob).cuda())
                self.samp_prob /= self.samp_prob.sum()

        self.sig = nn.LogSigmoid()
        # uniprob should be a tensor that sums to 1
        self.num_neighbors = num_neighbors

    def _neg_samples(self, pos_targets):
        """randomly samples negative labels that are not already positive labels
           if sample prob is not None then samples are drawn according to prob dist.

           :param
           pos_targets: Shape (batch_size, seq_length, num_neighbors)

           :return
           default                                                     10    +      10
           positive_neg_targets: Shape (batch_size, seq_length, num_neighbors + num_neg_samples)
        """

        # pos_targets = targets .view(targets.size(0), -1)
        num_samps = self.num_neighbors * pos_targets.size(0) * pos_targets.size(1)

        if self.samp_prob is not None:
            # (batch_size, seq_len * num_neighbors)
            neg_targets = torch.multinomial(self.samp_prob,
                                          num_samps, replacement=False)
        else:
            neg_targets = torch.FloatTensor(num_samps).\
                uniform_(1, len(self.class_weights))

        neg_targets = neg_targets.view(pos_targets.size()).type(torch.LongTensor)
        if pos_targets.is_cuda:
            neg_targets = neg_targets.cuda()

        new_targets = torch.cat([pos_targets, neg_targets], 2)
        return new_targets

    def forward(self, logits, target):
        """Logits should be (batch_size, seq_length,  num_neighbors+num_neg_samples)"""
        log_probabilities = self.sig(logits)
        target = self._neg_samples(target)
        target, logits, = target.view(target.size(0),-1), logits.view(logits.size(0), -1)
        # NLLLoss(x, class) = -weights[class] * x[class]
        return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()


def test_ce_loss():
    x = torch.FloatTensor([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.2, 0.1, 0.7], [0.4, 0.3, 0.3]])
    y = torch.LongTensor([0, 0, 2, 2])
    class_weights = [1] * len(y)
    ce_loss = CrossEntropyLoss(class_weights)
    loss = ce_loss(x.cuda(), y.cuda())
    print(loss)

def test_bce_loss():

    bsize = 30
    seq_length = 20
    num_neighbors = 10 # num_neighbors is the num of positive classes
    neg_samps = 10 # negative samples
    nclasses = num_neighbors + neg_samps
    # num_classes = 200

    yhat = F.softmax(torch.randn((bsize, seq_length, nclasses)), dim=2)
    y = torch.ones(bsize, seq_length, num_neighbors).type(torch.cuda.LongTensor)
    yhat, y = yhat.cuda(), y.cuda()

    bce_loss = BinaryCrossEntropyLoss(nclasses)
    targets = bce_loss.forward(yhat, y)

def test_power_bce_loss( bsize = 2, code_size = 3, seq_len = 1):
    yhat = torch.FloatTensor([[[0.9, 0.2, 0.8]],[[0.9, 0.9, 0.9]]]).cuda()
    y = torch.FloatTensor([[[0.0, 0., 0.]],[[1., 1., 1.]]]).cuda()
    #yhat = torch.randn((bsize, seq_len, code_size)) # logit in the loss itself
    #y = torch.ones(bsize, seq_len, code_size).type(torch.cuda.FloatTensor)
    yhat, y = yhat.cuda(), y.cuda()
    loss = PowerBCEWithLogitsLoss(code_size)

    l1 = loss(yhat, y)
    l2 = loss.f(yhat, y)
    print("Power loss is {}".format(l1.item()))
    print("new power loss is {}".format(l2.item()))


class PowerBCEWithLogitsLoss(nn.Module):
    """
    Same as BCE with logit losses only that each loss given
    by bits in a codeword is weighted by the power correspond to binary position.
    """
    log_sigmoid = nn.LogSigmoid()

    def __init__(self, code_size, temp = 1.):
        super().__init__()

        self.powers = (2 ** (torch.FloatTensor(list(reversed(range(code_size))))/temp)).cuda()
        self.class_weights = Variable(torch.FloatTensor().cuda())
        self.nll = nn.BCEWithLogitsLoss(reduce= False)

    def forward(self, logits, target):
        nll_loss = self.nll(logits, target)
        power_nll_loss = (nll_loss * self.powers) / torch.sum(self.powers)
        return power_nll_loss.mean()

    def f(self, logits, target):
        # expects both logit and targets to be FloatTensors
        assert target.type() == 'torch.cuda.FloatTensor'
        log_probabilities = self.log_sigmoid(logits)
        # NLLLoss(x, class) = -weights[class] * x[class]
        # compute the binary cross entropy loss given logit input
        nll_loss = - target * log_probabilities
        power_nll_loss = (nll_loss * self.powers) / torch.sum(self.powers)
        # -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
        return power_nll_loss.mean()


if __name__ == "__main__":

    # test_bce_loss()
    test_power_bce_loss()

