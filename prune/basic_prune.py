import numpy as np
from models.networks.compress.prune.utils import prune_rate, arg_nonzero_min


"""might have to use this for merging layers as well !"""
def prune_attention(model, threshold=0.2):
    print("Pruning with a threshold of {} weight magnitude.".format(threshold))
    pruned_inds_by_layer = []
    for p_name, p in model.named_parameters():
        # print(p_name)
        pruned_inds = p.data.abs() < threshold
        p.data[pruned_inds] = 0.
        model.state_dict()[p_name].data.copy_(p)
        pruned_inds_by_layer.append(pruned_inds)
        # count += 1
    return model


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


def weight_prune(model, pruning_perc, return_masks=False, all_weights = False):

    if pruning_perc < 1 and pruning_perc !=0: pruning_perc *= 100
    mod_name = model.__class__.__name__.lower()

    if all_weights:
        # this means keys, values and even layer norm is considered to be pruned
        def wc(p_name): return True
    else:
        wc = get_wc(mod_name)

    if return_masks: masks = []

    for p_name, p in model.named_parameters():

        # including wc, now threshold set correctly only for dense layers
        if len(p.data.size()) != 1 and wc(p_name):

            all_weights = list(p.cpu().data.abs().numpy().flatten())
            threshold = np.percentile(np.array(all_weights), pruning_perc)
            pruned_inds = p.data.abs() < threshold
            # print((p_name, (pruned_inds == 0).sum()/len(pruned_inds)))
            s = p.clone()
            p.data[pruned_inds] = 0.
            # they are not equal
            assert (s == p).all() == 0
            if return_masks: masks.append(pruned_inds.float())
    return masks if return_masks else model


def global_weight_prune(model, pruning_perc,
                        return_masks=False, all_weights = False):
    ''' Prune pruning_perc% weights globally (not layer-wise) arXiv: 1606.09274 '''

    if pruning_perc < 1 and pruning_perc !=0: pruning_perc *= 100
    mod_name = model.__class__.__name__.lower()
    if all_weights:
        # this means keys, values and even layer norm is considered to be pruned
        def wc(p_name): return True
    else:
        wc = get_wc(mod_name)

    all_weights = []
    for p_name, p in model.named_parameters():
        # including wc, now threshold set correctly only for dense layers
        if len(p.data.size()) != 1 and wc(p_name):
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)
    # generate mask
    if return_masks: masks = []
    for p_name, p in model.named_parameters():
        if len(p.data.size()) != 1 and wc(p_name):
            pruned_inds = p.data.abs() < threshold
            if return_masks: masks.append(pruned_inds.float())
            p.data[pruned_inds] = 0.
            # model.state_dict()[p_name].data.copy_(p)
    return masks if return_masks else model


def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind,
        to_prune_layer_ind))

    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        print('{:.2f} pruned'.format(current_pruning_perc))

    return masks