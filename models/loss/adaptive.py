from torch import nn
import torch.nn.functional as F
import torch


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    print(- (soft_targets * logsoftmax(pred)).size())
    return torch.sum(- soft_targets * logsoftmax(pred), 1)


class AdaptiveLoss(nn.Module):
    """Loss layer for Adaptive Softmax

    Args:
        cutoff: indexes of words that splited into each bucket

    Shape:
        - input: [(N, cutoff[0] + len(cutoff) - 1), (N_1, cutoff[1] - cutoff[0]), ...]
        - target: (N)

    Examples::

        >>> cutoff = [2000, 10000]
        >>> m = AdaptiveSoftmax(20, cutoff)
        >>> criterion = AdaptiveLoss(cutoff)
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(low=0, high=10000, size=[128])
        >>> output = m(input, target)
        >>> loss = criterion(output, target)
        >>> loss.backward()
    """

    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def remap_target(self, target):

        new_target = [target.clone()]
        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i
            if mask.any():
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                new_target.append(None)
        return new_target

    def forward(self, input, target):

        batch_size = input[0].size(0)
        target = self.remap_target(target.data)

        output = 0.0

        for i in range(len(input)):
            if input[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= input[i].size(1)
                """
                if target[i].size(0) > input[i].size(0):
                    pad_t = torch.zeros(abs(target[i].size(0) - input[i].size(0)),
                                        input[i].size(1)).cuda()
                    input[i] = torch.cat([input[i], pad_t])
                elif target[i].size(0) < input[i].size(0):
                    pad_t = torch.zeros(abs(target[i].size(0) - input[i].size(0))).type(torch.cuda.LongTensor)
                    target[i] = torch.cat([target[i], pad_t])
                """

                if target[i].size(0) > input[i].size(0):
                    target[i] = target[i][:int(input[i].size(0))]
                elif target[i].size(0) < input[i].size(0):
                    input[i] = input[i][:int(target[i].size(0)), :]

                output = output + F.cross_entropy(
                    input[i], target[i], size_average=False
                )

        output /= batch_size
        return output

    def forward_all(self, input, target):
        """for when you want to save individual losses for adaptive softmax"""
        batch_size = input[0].size(0)
        target = self.remap_target(target.data)
        output = 0.0
        for i in range(len(input)):
            if input[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= input[i].size(1)
                print(input[i].size())
                print(target[i].size())
                output = output + cross_entropy(input[i], target[i])
        output /= batch_size
        return output


if __name__ == "__main__":
    """
        m = AdaptiveSoftmax(20, [2000, 10000])
        input = torch.randn(128, 20)
        target = torch.randint(low=0, high=10000, size=[128])
        output = m(input, target)
        log_prob = m.log_prob(input)

        approx_softmax = choose_approx_softmax(args.approx_softmax)
        asoft = approx_softmax(ntokens, args.nhid, ntokens_per_class = None)
        yhat = output.view(-1, ntokens)
        yhat = asoft(yhat, targets)                        
    """
