import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def clean_bert(w_name):
    return w_name.replace(".weight", "").replace("embeddings.", "").\
        replace("bert.encoder.layer.", "").replace("transformer.", "")


def get_y_title(metric):
    if metric == 'emd':
        cbar = 'Wasserstein distance ($10^{-5}$)'
    elif metric == 'cka':
        cbar = 'Centered Kernel Alignment ($10^{-2}$)'
    elif metric == 'cov':
        cbar = 'Normalized Covariance distance ($10^{-2}$)'
    elif metric == 'kl':
        cbar = 'Kullbeck-Leibler Divergence ($10^{-2}$)'
    elif metric == 'cos':
        cbar = 'Cosine Similarity ($10^{-2}$)'
    elif metric == 'euclidean':
        cbar = 'Euclidean Distance'
    elif metric == 'manhattan':
        cbar = 'Manhattan Distance'
    return cbar


def create_nlist(x_len, sub=0):
    pos = []
    for i in range(x_len-1):
        for j in range(x_len - sub):
            pos.append((i, j))
    return pos


def plot_layer_sims(
        sims,
        param_by_type,
        mask=True,
        metric='cov',
        mod_name='Transformer',
        annotate=False
):
    """

    :param sims: a numpy similarity matrix
    :param param_by_type:
    :param mask: masks out symmetric similarity measures when true
    (only view the lower triangular of similarity matrix in the heatmap)
    :param metric: how to measure the layer similarity
    :param mod_name: the name of the model type
    :param annotate: bool that adds annotations to the plot if True
    :return:
    """

    print(param_by_type.keys(), sims.shape)

    # remove layer norm
    num_removed = 0
    for i, p_name in enumerate(list(param_by_type)):
        if 'ln' not in p_name:
            del param_by_type[p_name]
            sims.pop(i)
            num_removed += 1

    # don't need mask atm
    sns.set()
    sim_len = len(sims)
    gs = [None] * sim_len
    x_len = int(np.sqrt(sim_len)) + 1
    # print(x_len)

    f, ax = plt.subplots(sim_len- num_removed, 1, sharey=False)
    f.tight_layout()  # Or equivalently,  "plt.tight_layout()"

    # check aggregate_layers, we only use 4 keys for gpt-2
    #if 'gpt' in mod_name:
    #    pos = create_nlist(x_len, 1)
    #else:
    pos = create_nlist(x_len)

    #try:
    assert len(pos) == len(param_by_type)
    #except:
    #    pos = create_nlist(x_len, 1)
    #    assert len(pos) == len(param_by_type)

    for k, ptype in zip(range(len(sims)), param_by_type.keys()):
        s = sims[k].detach().numpy()
        if mask:
            m = np.zeros_like(s)
            tri_l = np.tril_indices_from(m)
            m[tri_l] = True
            m = np.rot90(m)

        # xn, yn = x_names[k], y_names[k]
        pt_names = list(param_by_type[ptype].keys())
        labels = [clean_bert(tag) for tag in pt_names]

        cbar = get_y_title(metric)

        with sns.axes_style("white"):
            # if metric == 'euclidean': s = ((s - s.mean()) / s.std())
            g = sns.heatmap(s, xticklabels=list(labels),   yticklabels=labels, ax=ax[pos[k]],
                        cbar_kws={'label': cbar}, annot=annotate, annot_kws={"size": 6}, fmt='.1f', mask=m)
            #  cmap='BrBG'
            g.set_title(pt_names[0].replace("0.", ""))

        if '_embeddings' in labels[0]: labels = range(len(labels))
        y_labels = list(range(len(labels)))
        x_labels = y_labels

        g.set_xticklabels(x_labels, rotation=0)
        g.set_yticklabels(y_labels, rotation=0)

    # plt.title(mod_name)
    save_fn = metric.lower() + '_' + mod_name.lower() + '.pdf'
    if os.path.isfile(save_fn):
        os.remove(save_fn)

    plt.savefig(save_fn, format='pdf', dpi = f.dpi * 2)
    plt.show(block=False)


def plot_sim(sims, xlabels=None, ylabels=None, mask=None):
    """ use to analyse the similariity between two matrices (of most interest covariance matrices)"""
    sns.set()
    ax = sns.heatmap(sims, mask=mask, xticklabels=xlabels, yticklabels=ylabels)
    plt.show()


def plot_sim_by_type(sims, xlabels=None, ylabels=None, mask=None):
    """ use to analyse the similariity between two matrices (of most interest covariance matrices)"""
    sns.set()
    ax = sns.heatmap(sims, mask=mask)
    # , xticklabels=xlabels, yticklabels=ylabels)
    plt.show()


if __name__ == "__main__":

    metric = 'euclidean' # 'cov'
    from prune.transformer_prune import get_bert, layer_merge_by_type

    model = get_bert()
    sims, param_by_type, x_names, y_names = layer_merge_by_type(model, perc=0.5, metric='cov', cn=False)
    plot_layer_sims(sims, param_by_type, mask=True,
                    metric=metric, mod_name='Transformer', annotate=False)

