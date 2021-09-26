import numpy as np
import math
from collections import Counter
from util.batchers import get_pred_target_split


def bleu(stats):
    """Compute BLEU given n-gram statistics"""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_perc = sum([math.log(float(x)/y) for x, y in zip(stats[2::2], stats[3::2])])/4.
    return math.exp(min([0, 1 -float(r) / c]) + log_bleu_perc)


def bleu_stats(hypo, ref, bleu_len=5):
    """Compute statistics for BLEU."""
    stats = list()
    stats.append(len(hypo))
    stats.append(len(ref))
    for n in range(1, bleu_len):
        s_ngrams = Counter([tuple(hypo[i:i+n]) for i in range(len(hypo)+1-n)])
        r_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref)+1-n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypo) + 1-n, 0]))
    return stats


def get_bleu(hypo, ref, seq_len=10, bleu_len=5):
    """
    Get validation BLEU score
    shuffle=true when (hypo,hypo) when evaluating bleu diversity (compares predictions with each other)
    """
    stats = np.array([0.]*seq_len)
    for h, r in zip(hypo, ref):
        h = np.trim_zeros(h)
        r = np.trim_zeros(r)
        stats += np.array(bleu_stats(h, r, bleu_len))
    return 100 * bleu(stats)


def convert2bleu(outputs, targets, lengths, seq_len=10, bleu_len=5):
    output_pind, targets = get_pred_target_split(outputs, targets, lengths)
    # print(output_pind, targets)
    bleu_score = get_bleu(output_pind, targets, seq_len, bleu_len)  # , args.bleu_num)
    return bleu_score


def get_bleus(hypos, ref):
    total_bleu = 0.
    for i in range(len(hypos)):
        total_bleu += get_bleu(hypos[i], ref[i], len(hypos[i]))
    return total_bleu


if __name__ == "__main__":

    import torch
    torch.manual_seed(1)
    hypo = torch.randint(10000,(50, 35)).numpy()
    ref = torch.randint(10000, (50, 35)).numpy()
    print(hypo[0,:])
    print(get_bleu(hypo, ref, seq_len=4, bleu_len=2))

