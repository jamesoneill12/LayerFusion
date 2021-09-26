"""edited from https://github.com/mightydeveloper/Deep-Compression-PyTorch"""
import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, DBSCAN
# from sklearn.cluster import OPTICS
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=10, c_name ='kmeans'):
    """
    Applies weight sharing to the given model
    """
    for name, param in model.named_parameters():
        if 'weight' in name.lower() and 'layernorm' not in name.lower() \
                and 'word_emb' not in name.lower()\
                and 'att' not in name.lower()\
                and 'corenet.3' in name.lower()\
                and len(param.data.size()) != 1:
            dev = param.device
            weight = param.data.cpu().numpy()
            shape = weight.shape
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**bits)
            print(name)
            if  c_name == 'kmeans':
                if len(mat.data) < 10e5:
                    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=100,
                             precompute_distances=True, algorithm="full")
                else:
                    kmeans = MiniBatchKMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=100)
                kmeans.fit(mat.data.reshape(-1, 1))
                new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                new_weight = new_weight.reshape(param.size())
            elif c_name == 'digitize':
                new_weight = space[np.digitize(mat.data.reshape(-1, 1), bins=space, right=True)]\
                    .reshape(param.size()).astype('float32')
                # print(new_weight[:100])
            elif c_name == 'dbscan':
                dbscan = DBSCAN(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1)
                dbscan
            elif 'affinity' in c_name:
                ap = AffinityPropagation(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1)
                ap.fit(mat.data.reshape(-1, 1))
                new_weight = ap.cluster_centers_[kmeans.labels_].reshape(-1)
                new_weight = new_weight.reshape(param.size())
            else:
                new_weight = param.data.cpu().numpy()
            param.data = torch.from_numpy(new_weight).to(dev)
    return model
