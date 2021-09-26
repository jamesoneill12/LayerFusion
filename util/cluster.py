"""
Purpose of clustering is to build vocab embedding hierarchy can then be used to
during langauge modelling where if next word is particularly difficult to predict
we instead predict the parent embeddings in the vocab hierarchy.
"""

from macros import *
import hdbscan
import numpy as np
from sklearn.decomposition import IncrementalPCA
from scipy.spatial import KDTree
import fastcluster
from loaders.helpers import load_subword_embeddings, load_vocab, save_vocab


def get_dim_reduction(vectors, reduce_dim, dim):
    if reduce_dim == 'svd':
        vectors = np.linalg.svd(vectors)
    if reduce_dim == 'pca':
        ipca = IncrementalPCA(n_components=dim, batch_size=10000)
        print("Fitting PCA...")
        ipca.fit(vectors)
        print("Performing Transform ...")
        vectors = ipca.transform(vectors)
    return (vectors)


def hdbscan_hierarchy(vectors, distance):
    """
    HDBSCAN

    ::param
        algorithm='best',
        alpha=1.0,
        approx_min_span_tree=True
        gen_min_span_tree=True
        leaf_size=40
        memory=Memory(cachedir=None)
        metric='euclidean'
        min_cluster_size=5
        min_samples=None
        p=None
    """

    clusterer = hdbscan.HDBSCAN(metric = distance, gen_min_span_tree=True)
    print("Fitting HDBSCAN")
    clusterer.fit(vectors)
    """
    print("Building Minimum Spanning Tree ..")
    clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size= 80, edge_linewidth=2)
    print("Run single linkage ..")
    clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    print("Plot Single Linkage")
    clusterer.condensed_tree_.plot()
    """
    return clusterer


# http://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
def get_hdbscan_hierarchy(word2vec, distance = 'manhattan',
                          reduce_dim = 'pca', dim = 10, save=None):
    word2vec = {k: v.reshape(300, ) for k, v in word2vec.items()}
    vectors = np.array(list(word2vec.values()))
    print("Vocabulary Shape: {}".format(vectors.shape))
    if reduce_dim!=None: vectors = get_dim_reduction(vectors, reduce_dim, dim = dim)
    hh = hdbscan_hierarchy(vectors, distance)
    if save!=None: save_vocab(hh, path=save, show_len=False)
    return (hh)


def get_kd_tree(word2vec, reduce_dim = 'pca', lsize = 100000, dim = 10, save = None):
    word2vec = {k: v.reshape(300, ) for k, v in word2vec.items()}
    vectors = np.array(list(word2vec.values()))
    print("Vocabulary Shape: {}".format(vectors.shape))
    if reduce_dim!=None: vectors = get_dim_reduction(vectors, reduce_dim, dim = dim)
    print("Vector shape {}".format(vectors.shape))
    kdt = KDTree(vectors, leafsize=lsize)
    if save!=None: save_vocab(kdt, path=save, show_len=False)
    return (kdt)


# returns dictionary with neighbors and corresponding neighbor names and
# truncated similarity probabilities. This allows us to sample with a probability


if __name__ == "__main__":

    # sentences = wikitext103()
    # vocab = get_vocab(sentences)
    #word2ind = load_vocab()
    #id2word = load_vocab(IND2WORD_PATH)

    import sys
    sys.setrecursionlimit(10000)
    word2vec = load_vocab(WIKI3_WORD2VEC_VOCAB_PATH)
    #kdt = get_kd_tree(word2vec, save=WIKI2_KD_TREE_PATH)
    hdbscan = get_hdbscan_hierarchy(word2vec, save=WIKI3_HDBSCAN_PATH, dim = 20)#20

    #nearest_neighbor_lookup = get_nearest_neighbors(word2vec)
    #save_vocab(nearest_neighbor_lookup, WIKI2_NEIGHBOR2VEC_PATH)
