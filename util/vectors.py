"""
Purpose: Build character embeddings, NOT for building cluster hierarchy
"""

import numpy as np
from macros import *
from sklearn.decomposition import PCA
import pickle


# lm terms converts the word into the correctly spelled word
def dict_convert2id(siam_list, tick_terms, term_ids, sent_idx):
    # print_ids(tick_terms, term_ids)
    siam_pairs = []
    for sp in siam_list:
        sent_list = []
        for s in sp[sent_idx].lower().split():
            if s in term_ids:
                # tick_terms:<term, vec>, term_ids:<correct_term, index>
                sent_list.append(term_ids[s])
        siam_pairs.append(sent_list)
    return siam_pairs


def list_convert2id(siam_list, term_ids, sent_idx):
    siam_pairs = []
    for sp in siam_list:
        sent_list = []
        for s in sp[sent_idx].lower().split():
            if s in term_ids:
                correct_term = s
                sent_list.append(term_ids[correct_term])
        siam_pairs.append(sent_list)
    return siam_pairs


def read_data(filename = "annotated_data.p"):
    with open(ROOT_PATH+filename, "rb") as f:
        obj = pickle.load(f)
    return obj


class Text2CharVectors:

    def __init__(self, embedding_dim = 100, use_pca = False):

        self.maxlen = 200
        self.step = 3
        self.char2inds = None
        self.inds2char = None
        self.embedding_vector = None
        self.embedding_matrix = None
        self.embeddings_path = '' # GLOVE_CHAR_VECTOR_PATH
        self.use_pca = use_pca
        self.embedding_dim = embedding_dim

    def get_char_indices(self, chars):
        print('total chars:', len(chars))
        char2inds = dict((c, i) for i, c in enumerate(chars))
        inds2char = dict((i, c) for i, c in enumerate(chars))
        return char2inds, inds2char

    def vectorize_chars(self, text):

        # cut the text in semi-redundant sequences of maxlen characters
        chars = sorted(list(set(text)))
        self.char2inds, self.inds2char = self.get_char_indices(chars)

        sentences = []
        next_chars = []
        for i in range(0, len(text) - self.maxlen, self.step):
            sentences.append(text[i: i + self.maxlen])
            next_chars.append(text[i + self.maxlen])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        X = np.zeros((len(sentences), self.maxlen), dtype=np.int)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t] = self.char2inds[char]
            y[i, self.char2inds[next_chars[i]]] = 1
        return X, y

    def random_subset(self, X, y, p=0.2):
        idx = np.random.randint(X.shape[0], size=int(X.shape[0] * p))
        X = X[idx, :]
        y = y[idx]
        return X, y

    def get_embedding_matrix(self):
        embedding_matrix = np.zeros((len(self.chars), 300))
        # embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))

        if self.embedding_vector == None:
            pass
            # self.embedding_vector = load_char_embedding_vectors(self.embeddings_path)

        for char, i in self.char2inds.items():
            # print ("{}, {}".format(char, i))
            embedding_vector = self.embedding_vectors.get(char)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return (embedding_matrix)

    def get_pca_embedding_matrix(self):
        # Use PCA from sklearn to reduce 300D -> 50D
        pca = PCA(n_components= self.embedding_dim)
        pca.fit(self.embedding_matrix)
        embedding_matrix_pca = np.array(pca.transform(self.embedding_matrix))
        print(embedding_matrix_pca)
        print(embedding_matrix_pca.shape)
        return embedding_matrix_pca


if __name__ == "__main__":

    create = False

    if create:
        data = read_data()
        t2cv = Text2CharVectors()
        X, y = t2cv.vectorize_chars(data["item_text"])
        #save_char_vectors()
    else:
        pass #char_vectors = load_char_embedding_vectors()

    #print(char_vectors.keys())