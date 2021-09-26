from models.networks.compress.aggregate_layers import clean_param_name
from util.metrics.getter import compute_similarity
import numpy as np


def calculate_sim_by_type(
        param_by_type,
        n_layers,
        met,
        metric,
        sims,
        mod_name,
        factor=10e5,
        clean_names=False
):
    """
        computes the similarity aggregated by the type e.g sim between dense layers only or attention
        heads only but not similarity between these types
    """

    x_names, y_names = [], []
    sim_dict = dict(zip(param_by_type.keys(), [{}] * len(param_by_type)))
    for k, (param_name, param_type) in enumerate(param_by_type.items()):
        for i, (f_name_raw, f_param) in enumerate(param_type.items()):
            f_name = clean_param_name(f_name_raw, mod_name)
            x_names.append(f_name) if clean_names else x_names.append(f_name_raw)
            print(f_name)
            # print
            # print(n_layers.keys())

            for lname in n_layers.keys():
                if lname in f_name:
                    nlayer = n_layers[lname]

            for j, (s_name_raw, s_param) in enumerate(list(param_type.items())[i+1:nlayer]):
                s_name = clean_param_name(s_name_raw, mod_name)
                y_names.append(s_name) if clean_names else y_names.append(s_name_raw)
                #  in other words, ignore the bias
                if len(f_param.size()) > 1 and len(s_param.size()) > 1:
                    sim = compute_similarity(f_param, s_param, met, metric)
                    if metric == 'cos': sim = sim.mean()
                    # for the k-th sim measure
                    sims[k][i, j] = sim * factor
                    sim_dict[param_name] = {f_name + "_" + s_name: sim * factor}
                    print("{} - {} sim = {}".format(f_name, s_name, sim))
                    if 'norm' in f_name:
                        print("{} range: min {} \t max {}".format(f_name, f_param.min(), f_param.max()))
                        print("{} range: min {} \t max {}".format(s_name, s_param.min(), s_param.max()))

    return sims, x_names, y_names


def compute_merge(model, sims, x_names, perc=0.3):
    """takes output of calculate_sim_by_type and returns merged layers at perc rate"""
    if perc > 1: perc /= 100
    indices, Ns = [], []

    model_name = model.__class__.__name__.lower().split('for')[0]

    for sim_mat in sims:
        sim_mat = sim_mat.detach().numpy()  # to allow for negative striding below
        # print("similarity mat ", sim_mat.shape)
        N = int(perc * sim_mat.shape[0])
        Ns.append(N)
        # refer to https://stackoverflow.com/questions/6910641/
        # how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array/23734295#23734295
        ind = np.argpartition(sim_mat, -N, axis=1)[:, -N:]
        indices.append(ind)

    """for the k-th set of indices for k-similarity mat, we choose the i-th row that keeps top-N highest similarities
    x_names is 72 which is made up 12 x 6 (6 unique type of weight (see nlayer in get_layers_by_type)) """

    lind, as_id = 0, 0
    # lind carries over 12 when moving to different weight type and as_id is used when asserting that
    # the number of merges equals the sum 30% for each different weight type

    state = model.state_dict()

    sim_len = len(sims)
    deleted = [] # keeps track of already deleted params
    for k in range(sim_len):
        # for each cohort of similarity matrices
        for i in range(sim_len):
            # loop through each row in a give k sim matrix
            for j in range(i, sim_len):
                # and each column in a give j sim matrix
                for l in range(N):
                    # accounts for top N indices for each (k, j) sim matrix k
                    # if column j of row i matches index l
                    # i != j ensures same layers not checked
                    if j == indices[k][i, l] and i != j:
                        f_id, s_id = lind + i, lind + j
                        # added 2nd condition due to errors
                        if x_names[s_id] not in deleted and x_names[f_id] not in deleted:
                            f_name = clean_param_name(x_names[f_id], model_name)
                            s_name = clean_param_name(x_names[s_id], model_name)
                            print("merging {} and {}".format(f_name, s_name))
                            new_weight = (state[x_names[f_id]].data + state[x_names[s_id]].data) /2
                            fid_bias = x_names[f_id].replace("weight", "bias")
                            if fid_bias in state:
                                new_bias = (state[fid_bias].data
                                        + state[x_names[s_id].replace("weight","bias")].data) /2
                            # add merged layer to first weight and delete second weight
                            state[x_names[f_id]].data.copy_(new_weight)
                            sid_bias = x_names[f_id].replace("weight", "bias")
                            if sid_bias in state: state[sid_bias].data.copy_(new_bias)
                            print("deleting {}".format(x_names[s_id]))
                            del state[x_names[s_id]]
                            if sid_bias in state: del state[x_names[s_id].replace("weight", 'bias')]
                            deleted.append(x_names[s_id])
                            # this was unindented when 2nd condition was not necesary to get over bug
                        as_id += 2
        lind += indices[k].shape[0]

    print(Ns, as_id)
    # assert sum(Ns) * 2 == as_id
    return model


def old_compute_merge(sims, perc, x_names, y_names, model):

    for sim_mat in sims:
        sim_mat = sim_mat.numpy()  # to allow for negative striding below
        N = int(perc * sim_mat.shape[0])
        m, n = sim_mat.shape[0], sim_mat.shape[1]
        idx = np.argpartition(-sim_mat, N, axis=1)[N - 1::-1]
        sim_inds = sim_mat[np.arange(m)[:, None], idx, np.arange(n)]
        print(N)
        # print(sim_mat.shape)
        sorted_sim = np.argsort(sim_mat.ravel())[:N]
        sp = np.unravel_index(sorted_sim, (N, N))
        sim_inds = np.dstack(sp)
        #rint(sim_inds)

    for sim_mat in sims:
        for i, x_name in enumerate(x_names):
            for j, y_name in enumerate(y_names):
                if (i, j) in i:
                    print("hello")

                    model.weight[i+j].data = (model.weight[i].data + model.weight[j].data)/ 2
                    # +1 because bias should be straight after
                    model[i+j+1].data = (model.layer[i+1].data + model.layer[j+1].data)/ 2
                    # then get rid of first layer (could be j either)
                    model.layer[i].data = None
                    model.layer[i].data = None


def test_sim_by_type():

    pass
