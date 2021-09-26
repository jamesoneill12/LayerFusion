import numpy as np


def get_wmd(output, targets, corpus, pre_emb):
    """corpus.dictionary is from Dictionary class"""
    pred_sents = []
    targ_sents = []
    # cnt = 0
    for i in range(output.shape[0]):
        p_temp = [];
        t_temp = [];
        for j in range(output.shape[1]):
            p_word = corpus.dictionary.idx2word[output[i, j]]
            # int(cnt+j)
            t_word = corpus.dictionary.idx2word[targets[i, j]]
            p_temp.append(p_word)
            t_temp.append(t_word)
        # cnt += output.shape[1]
    pred_sents.append(p_temp)
    targ_sents.append(t_temp)
    av_wmd_sim = np.ma.masked_invalid(([pre_emb.wmdistance(pred_sent, targ_sent) for
                                            (pred_sent, targ_sent) in zip(pred_sents, targ_sents)])).mean()
    return av_wmd_sim