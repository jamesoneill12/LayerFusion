import math
from collections import Counter
import numpy as np

def cider(stats):
    """implement cider here """
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_perc = sum([math.log(float(x)/y) for x, y in zip(stats[2::2], stats[3::2])])/4.
    return math.exp(min([0, 1 -float(r) / c]) + log_bleu_perc)


def cider_stats(hypo, ref, id2count, bleu_len=5):
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


def get_cider(hypo, ref, id2count, seq_len=10, bleu_len = 5):
    """Get validation cider scores"""
    stats = np.array([0.]*seq_len)
    for h, r in zip(hypo, ref):
        cstat = cider_stats(h, r, bleu_len, id2count)
        stats += np.array(cstat)
    return 100 * cider(stats)