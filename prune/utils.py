import numpy as np


def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix


def get_wc(mod_name):
    if 'transfoxl' in mod_name:
        def wc(p_name):
            return 'ff.CoreNet' in p_name and 'weight' in p_name
    elif 'openai' in mod_name:
        def wc(p_name):
            return 'mlp' in p_name and 'weight' in p_name
    elif 'gpt2' in mod_name:
        def wc(p_name):
            return 'mlp' in p_name and 'weight' in p_name
    elif 'trans' in mod_name:
        def wc(p_name):
            return 'dense' in p_name and 'weight' in p_name
    elif 'bert' in mod_name:
        def wc(p_name):
            return 'dense' in p_name and 'weight' in p_name
    return wc


def check_prune_perc(pruning_perc):
    if 0 > pruning_perc < 1:
        pruning_perc *= 100
    elif pruning_perc < 0:
        ValueError(f"{pruning_perc} cannot be less than 0, must be in [0, 100]")
    elif pruning_perc > 100:
        ValueError(f"{pruning_perc} cannot be more than 100, must be in [0, 100]")
    return pruning_perc
