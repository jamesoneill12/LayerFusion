import torch


class CodeAccuracy:
    def __init__(self, ecoc_nbits):
        bin_conv = [pow(2, i) for i in reversed(range(ecoc_nbits))]
        self.bin2int = torch.Tensor(bin_conv).unsqueeze(1).cuda()

    def get_code_accuracy(self, pred_code, target):
        pred = torch.mm(pred_code, self.bin2int).type(torch.cuda.LongTensor).t()
        acc = get_accuracy(pred, target)
        return acc


def get_accuracy(pred, target):
    """gets accuracy either by single prediction
    against target or comparing their codes """
    if len(pred.size()) > 1:
        pred = pred.max(1)[1]
    #pred, target = pred.flatten(), target.flatten()
    accuracy = round(float((pred == target).sum())/float(pred.numel()) * 100, 3)
    return accuracy


"""
# DON'T USE - OLD and SLOW
def get_code_accuracy(pred, target):
    # gets accuracy given predicted codes and target codes Far too slow atm !
    correct = 0.
    num_samps = pred.size(0)
    for i in range(num_samps):
        correct += float((pred[i, :] == target[i, :]).all())
    accuracy = round(correct/float(num_samps) * 100, 3)
    return accurac

"""