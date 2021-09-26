from helpers import get_n_layers
from aggregate_layers import get_n_layers_per_type,\
    get_layers_by_type, clean_param_name
from layer_similarity import calculate_sim_by_type
from merge.plotter import plot_sim
import random
from operator import itemgetter
import torch.nn as nn
import math
from models.mix_distribution import random_mix
from util.metrics.getter import get_sim_measure
import numpy as np
import torch


def layer_merge(model, perc=0.5, metric='cov'):
    """
    Merge Layers Based on the similarity between unit normalized layer pairs Covariance matrices (when using cov as dist)
       Note: The amount of merging is decided by perc, however I plan on automating this based on some
       performance metric such as validation perplexity or allow the user to specify a scheduled rate.
    """

    mod_name = model.__class__.__name__.lower()
    if perc > 1: perc /= 100
    n_layers = get_n_layers(model)
    n_merge = int(n_layers * perc)
    sims = torch.zeros(n_layers, n_layers)
    mask = np.ones((n_layers, n_layers))
    met = get_sim_measure(metric)
    print("{} % of the network will be pruned ({}/{})".format(int(perc * 100), n_merge, n_layers))
    x_names, y_names = [], []
    for i, (f_name, f_param) in enumerate(model.named_parameters()):
        f_name = clean_param_name(f_name, mod_name)
        x_names.append(f_name)
        for j, (s_name, s_param) in enumerate(list(model.named_parameters())[i+1:n_layers]):
            s_name = clean_param_name(s_name, mod_name)
            y_names.append(s_name)
            #  in other words, ignore the bias
            if len(f_param.size()) > 1 and len(s_param.size()) > 1:
                sim = met(f_param, s_param)
                sims[i, j] = sim
                print("{} - {} sim = {}".format(f_name, s_name, sim))
            else:
                mask[i, j] = 0
    sims = sims.detach().numpy() # .reshape(len(sims), 1)
    print(plot_sim(sims, xlabels=x_names, ylabels=y_names, mask=mask))


# TODO: Parallelize wherever possible here, computing similarity is
#  slow especially for Tranformer-XL. Needs to compute similarity faster.
def layer_merge_by_type(model, perc=0.5, metric='cov', cn=False):

    """
    Merge Layers Based on the Similarity between unit normalized layer pairs of covariance matrices
       Note: The amount of merging is decided by perc, however I plan on automating this based on some
       performance metric.
    """
    mod_name = model.__class__.__name__.lower()
    if perc > 1: perc /= 100
    print("1 ", mod_name)
    n_layers = get_n_layers_per_type(model)
    print("2 ",  model.__class__.__name__.lower())
    param_by_type = get_layers_by_type(model, clean_names=cn)
    met = get_sim_measure(metric)

    # creates the m x m similarity matrix
    n_merges, sims = [], []
    for layer_name, n_layer in n_layers.items():
        n_merge = int(n_layer * perc)
        n_merges.append(n_merge)
        sims.append(torch.zeros(n_layer, n_layer))

    print("{} % of the network with params will be pruned ({}/{})"
          .format(int(perc * 100), sum(n_merges), sum(n_layers.values())))

    # chooses the metric for computing similarity
    if metric == 'cov': factor = 10e2
    elif metric == 'emd': factor = 10e5
    elif metric == 'cka': factor = 10e2
    elif metric == 'kl': factor = 10e2
    elif metric == 'euclidean': factor = 10e1
    elif metric == 'cos': factor = 1

    # calculates the similarity by type of weights
    sims, x_names, y_names = calculate_sim_by_type(
        param_by_type, n_layers,
        met, metric, sims, mod_name,
        factor=factor, clean_names=cn
    ) # names=False for plotting
    # print("hello simmy ", sims[0].shape)
    return sims, param_by_type, x_names, y_names


def compute_normal_all_pair_sim(param_ss):
    'computes all_pair_sim for tuples of normal means and standard devs.'
    'desc: if the similarity is asymmetric, meaning it is not a proper distance' \
    'metric, well then both directions need to be computed. This is important when ' \
    'we freeze a layer because we now have a way of deciding which of the pair gets' \
    'frozen based on the asymmetric property.'
    all_pair_sim = {}
    for i, (f_name, f_p) in enumerate(param_ss.items()):
        for j, (s_name, s_p) in enumerate(param_ss.items()):
            # only considering sim between weight matrices of same size
            if f_name != s_name:
                # differences between mean and stds
                all_pair_sim[(f_name, s_name)] = math.sqrt(abs((f_p[0] - s_p[0]) + abs(f_p[1] - s_p[1])))
    layers_to_fuse = dict(sorted(all_pair_sim.items(), key=itemgetter(1)))
    return layers_to_fuse


def layer_stat(param_ss, perc):
    '''

    :param param_ss:
    :param perc:
    :return: list of tuples with names of module pairs to fuse
    '''
    pair_vals = compute_normal_all_pair_sim(param_ss)
    # don't pick top ones yet, need to filter out matches of different tensor size
    return pair_vals


def compute_mod_ss(model, layer_type):

    """returns a dictionary of tuples (mean, std) for each weight"""
    if 'lin' in layer_type: tag = nn.Linear
    elif 'conv' in layer_type: tag = nn.Conv2d
    param_ss, sum_means, sum_stds, cnt = {}, 0, 0, 0
    for i, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, tag):
            param_ss[name] = (m.weight.mean(), m.weight.std())
            sum_means += m.weight.mean()
            sum_stds += m.weight.std()
            cnt += 1

    if cnt == 0:
        print(f"No {layer_type} found for {model.__class__.__name__}")
        param_ss = None
    #else:
    #    param_ss['all'] = (sum_means / cnt, sum_stds / cnt)
    # print(f'{int(len(cnt)*)/cnt} pruned layers for {layer_type}')
    return param_ss


def fuse_weight(f_m, s_m, fuse_method):
    if fuse_method == 'mix':
        s_m_weight, f_m_weight = random_mix(s_m.weight.data, f_m.weight.data, p=0.5)
    elif fuse_method == 'mean':
        s_m_weight= (s_m.weight + f_m.weight) / 2
        f_m_weight = s_m_weight
    elif fuse_method == 'max':
        #  change to max
        s_m_weight = (s_m.weight+ f_m.weight) / 2
        f_m_weight = s_m_weight  # bias is always the average
    f_m.weight, s_m.weight = nn.Parameter(f_m_weight), nn.Parameter(s_m_weight)
    if s_m.bias is not None and f_m.bias is not None:
        fused_bias = nn.Parameter((s_m.bias + f_m.bias) / 2)
        f_m.bias, s_m.bias = fused_bias, fused_bias
        f_m.bias.grad = None

    f_m.weight.grad = None
    return f_m, s_m


def init_layer_fuser(model, layers = ['linear', 'conv'], perc=30,
                       dist='cosine', fuse_method='mix'):
    """
    where: mainly used in the lenet_iterative.py thus far

    what: takes a model linear and/or conv layers and returns mean and variance
    and removes percentage of layers furthest from the global mean
    mainly
    :param
    fuse_method: 'mix' mixes weights between the layers
                 'mean' takes average of layer pair
                 'max' ...
    """
    if type(layers) == list:
        param_ss = []
        for ltype in layers:
            # ss = sufficient stats
            cm = compute_mod_ss(model, ltype)
            if cm is not None:
                param_ss.append(cm)
            else:
                layers.remove(ltype)
        layer_to_fuse = [layer_stat(p_ss, perc) for p_ss in param_ss]

    modules = dict(model.named_modules())
    for ltofuse in layer_to_fuse:
        for pair in list(ltofuse):
            if modules[pair[0]].weight.size() != modules[pair[1]].weight.size():
                # print(f"Removing {pair}, tensor sizes do not match")
                del ltofuse[pair]

    num_linear_pairs = len(layer_to_fuse[0])
    num_conv_pairs = len(layer_to_fuse[1])

    if int(num_linear_pairs * (perc/100)) < num_linear_pairs:
        layer_to_fuse[0] = list(layer_to_fuse[0])[:int(num_linear_pairs * (perc / 100))]
    if len(layer_to_fuse) > 1:
        if int(num_conv_pairs * (perc / 100)) < num_conv_pairs:
            top_perc_conv_pairs = int(num_conv_pairs * (perc / 100))
            layer_to_fuse[1] = list(layer_to_fuse[1])[:top_perc_conv_pairs]

    fused_layers_name = {}
    for layer_type in layer_to_fuse:
        for l1_name, l2_name in layer_type:
            if l1_name not in fused_layers_name:
                fused_layers_name[(l1_name, l2_name)] = 1
            else:
                fused_layers_name[(l1_name, l2_name)] +=1
            new_w1, new_w2 =\
                random_mix(modules[l1_name].weight.data, modules[l2_name].weight.data, p=0.5, disc_rep=True)
            modules[l1_name].weight = nn.Parameter(new_w1)
            modules[l2_name].weight = nn.Parameter(new_w2)

    return model, fused_layers_name


def random_grads_off(model, opt, fused_layer_names):
    '''randomly chooses a layer from fuse pair to turn grads off for'''
    modules = dict(model.named_modules())
    for l1_name, l2_name in fused_layer_names:
        name = l1_name if random.random() > 0.5 else l2_name
        modules[name].weight.grad = None
    opt.step()
    #for l1_name, l2_name in fused_layer_names:
    #    modules[l1_name].weight.data.copy_(modules[l2_name].weight.data)
    return model, opt, fused_layer_names


def grads_off(model, opt, fused_layer_names):
    '''makes sure grads turned off after layer fusion carried out'''
    modules = dict(model.named_modules())
    for l1_name, l2_name in fused_layer_names:
        modules[l1_name].weight.grad.data.add_(modules[l2_name].weight.grad.data)
        modules[l2_name].weight.grad = None
    opt.step()
    for l1_name, l2_name in fused_layer_names:
        modules[l1_name].weight.data.copy_(modules[l2_name].weight.data)
    return model, opt, fused_layer_names


def mix_and_grads(model, opt, fused_layer_names):
    'repeatedly mixes the gradients during fusing'
    modules = dict(model.named_modules())
    for l1_name, l2_name in fused_layer_names:
        print(modules[l1_name].weight.data.grad)
        new_w1, new_w2 = random_mix(modules[l1_name].weight.data,
                                    modules[l2_name].weight.data, p=0.5, disc_rep=True)
        modules[l1_name].weight = nn.Parameter(new_w1)
        modules[l2_name].weight = nn.Parameter(new_w2)
        modules[l1_name].weight.grad.data.add_(modules[l2_name].weight.grad.data)
        modules[l2_name].weight.grad = None
    opt.step()
    for l1_name, l2_name in fused_layer_names:
        modules[l1_name].weight.data.copy_(modules[l2_name].weight.data)
    return model, opt, fused_layer_names


def test_mix_and_grads_off():
    from torchvision import models
    import torch.optim as optim
    model = models.resnet18(pretrained=True)
    model, fused_layer_names = init_layer_fuser(model)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model, opt, fused_layer_names = grads_off(model, opt, fused_layer_names)


def test_init_fuse_layer():
    from torchvision import models
    mod = models.resnet18(pretrained=True)
    mod, named_layers_fused = init_layer_fuser(mod, layers=['linear', 'conv'], perc=20,
                     dist='cosine', fuse_method='mix')
    for nlf in named_layers_fused: print(nlf)


if __name__ == "__main__":

    test_init_fuse_layer()

    # check if grads are turned off after mixing layer weights
    #test_mix_and_grads_off()
