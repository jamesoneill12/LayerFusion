# refer to (they use 0.5 by default as well) -> https://github.com/bdusell/rougescore
from __future__ import division
import collections
import numpy as np
import torch
import six


def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)


def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))


def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)


def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result


def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def _safe_f1(matches, recall_total, precision_total, alpha=0.5):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0


def rouge_n(peer, models, n, alpha):
    """
    Compute the ROUGE-N score of a peer with respect to one or more mods, for
    a given value of `n`.
    """
    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    model_counter = _ngram_counts(models, n)
    matches += _counter_overlap(peer_counter, model_counter)
    recall_total += _ngram_count(models, n)
    precision_total = len(models) * _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total, alpha)


def get_rouge_n(peers, models, n, alpha):
    total = 0.
    for peer, model in zip(peers, models):
        total += rouge_n(peer, model, n, alpha)
    return (total * 10)/len(models)


def rouge_1(peer, models, alpha):
    """
    Compute the ROUGE-1 (unigram) score of a peer with respect to one or more
    mods.
    """
    return rouge_n(peer, models, 1, alpha)


def rouge_2(peer, models, alpha):
    """
    Compute the ROUGE-2 (bigram) score of a peer with respect to one or more
    mods.
    """
    return rouge_n(peer, models, 2, alpha)


def rouge_3(peer, models, alpha):
    """
    Compute the ROUGE-3 (trigram) score of a peer with respect to one or more
    mods.
    """
    return rouge_n(peer, models, 3, alpha)


def lcs(a, b):
    """
    Compute the length of the longest common subsequence between two sequences.

    Time complexity: O(len(a) * len(b))
    Space complexity: O(min(len(a), len(b)))
    """
    # This is an adaptation of the standard LCS dynamic programming algorithm
    # tweaked for lower memory consumption.
    # Sequence a is laid out along the rows, b along the columns.
    # Minimize number of columns to minimize required memory
    if len(a) < len(b):
        a, b = b, a
    # Sequence b now has the minimum length
    # Quit early if one sequence is empty
    if len(b) == 0:
        return 0
    # Use a single buffer to store the counts for the current row, and
    # overwrite it on each pass
    row = [0] * len(b)
    for ai in a:
        left = 0
        diag = 0
        for j, bj in enumerate(b):
            up = row[j]
            if ai == bj:
                value = diag + 1
            else:
                value = max(left, up)
            row[j] = value
            left = value
            diag = up
    # Return the last cell of the last row
    return left


def rouge_l(peer, models, alpha=0.5):
    """
    Compute the ROUGE-L score of a peer with respect to one or more mods.
    """
    matches = 0
    recall_total = 0

    if type(peer) == torch.Tensor:
        peer = peer.cpu().numpy()
    if type(models) == torch.Tensor:
        models = models.cpu().numpy()

    if peer.shape == models.shape:
        scores = []
        for i in range(peer.shape[0]):
            mod = np.trim_zeros(models[i, :])
            per = np.trim_zeros(peer[i, :])
            matches = lcs(mod, per)
            score = _safe_f1(matches, len(mod), len(per), alpha)
            scores.append(score)
        return torch.autograd.Variable(torch.Tensor(scores), requires_grad=False).cuda()

    else:
        for model in models:
            matches += lcs(model, peer)
            recall_total += len(model)
            precision_total = len(models) * len(peer)
    return _safe_f1(matches, recall_total, precision_total, alpha)


def mask_score(props, words, scores):
    assert words.size() == scores.size()
    mask = (words > 0).float()
    return props * scores * mask