# -*- coding: utf-8 -*-
import os
import sys
from scipy import stats
import torch
from torch import nn
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy as np
import random
import warnings


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
testdir = os.path.dirname(__file__)

def symbolize(X, m):
    """
    Converts numeric values of the series to a symbolic version of it based
    on the m consecutive values.

    Parameters
    ----------
    X : Series to symbolize.
    m : length of the symbolic subset.

    Returns
    ----------
    List of symbolized X

    """

    X = np.array(X)

    if m >= len(X):
        raise ValueError("Length of the series must be greater than m")

    dummy = []
    for i in range(m):
        l = np.roll(X, -i)
        dummy.append(l[:-(m - 1)])

    dummy = np.array(dummy)

    symX = []

    for mset in dummy.T:
        rank = stats.rankdata(mset, method="min")
        symbol = np.array2string(rank, separator="")
        symbol = symbol[1:-1]
        symX.append(symbol)

    return symX


def symbolic_mutual_information(symX, symY):
    """
    Computes the symbolic mutual information between symbolic series X and
    symbolic series Y.

    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.

    Returns
    ----------
    Value for mutual information

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)

    symbols = np.unique(np.concatenate((symX, symY))).tolist()

    jp = symbolic_joint_probabilities(symX, symY)
    pX = symbolic_probabilities(symX)
    pY = symbolic_probabilities(symY)

    MI = 0

    for yi in list(pY.keys()):
        for xi in list(pX.keys()):
            a = pX[xi]
            b = pY[yi]

            try:
                c = jp[yi][xi]
                MI += c * np.log(c / (a * b)) / np.log(len(symbols));
            except KeyError:
                continue
            except:
                print("Unexpected Error")
                raise

    return MI


def symbolic_transfer_entropy(symX, symY):
    """
    Computes T(Y->X), the transfer of entropy from symbolic series Y to X.

    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.

    Returns
    ----------
    Value for mutual information

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)

    cp = symbolic_conditional_probabilities_consecutive(symX)
    cp2 = symbolic_conditional_probabilities_consecutive_external(symX, symY)
    jp = symbolic_joint_probabilities_consecutive_external(symX, symY)

    TE = 0

    for yi in list(jp.keys()):
        for xi in list(jp[yi].keys()):
            for xii in list(jp[yi][xi].keys()):

                try:
                    a = cp[xi][xii]
                    b = cp2[yi][xi][xii]
                    c = jp[yi][xi][xii]
                    TE += c * np.log(b / a) / np.log(2.);
                except KeyError:
                    continue
                except:
                    print("Unexpected Error")
                    raise
    del cp
    del cp2
    del jp

    return TE


def symbolic_probabilities(symX):
    """
    Computes the conditional probabilities where M[A][B] stands for the
    probability of getting B after A.

    Parameters
    ----------
    symX : Symbolic series X.
    symbols: Collection of symbols. If "None" calculated from symX

    Returns
    ----------
    Matrix with conditional probabilities

    """

    symX = np.array(symX)

    # initialize
    p = {}
    n = len(symX)

    for xi in symX:
        if xi in p:
            p[xi] += 1.0 / n
        else:
            p[xi] = 1.0 / n

    return p


def symbolic_joint_probabilities(symX, symY):
    """
    Computes the joint probabilities where M[yi][xi] stands for the
    probability of ocurrence yi and xi.

    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symbols: Collection of symbols. If "None" calculated from symX

    Returns
    ----------
    Matrix with joint probabilities

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)

    # initialize
    jp = {}
    n = len(symX)

    for yi, xi in zip(symY, symX):
        if yi in jp:
            if xi in jp[yi]:
                jp[yi][xi] += 1.0 / n
            else:
                jp[yi][xi] = 1.0 / n
        else:
            jp[yi] = {}
            jp[yi][xi] = 1.0 / n

    return jp


def symbolic_conditional_probabilities(symX, symY):
    """
    Computes the conditional probabilities where M[A][B] stands for the
    probability of getting "B" in symX, when we get "A" in symY.

    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.

    Returns
    ----------
    Matrix with conditional probabilities

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)

    # initialize
    cp = {}
    n = {}

    for xi, yi in zip(symX, symY):
        if yi in cp:
            n[yi] += 1
            if xi in cp[yi]:
                cp[yi][xi] += 1.0
            else:
                cp[yi][xi] = 1.0
        else:
            cp[yi] = {}
            cp[yi][xi] = 1.0

            n[yi] = 1

    for yi in list(cp.keys()):
        for xi in list(cp[yi].keys()):
            cp[yi][xi] /= n[yi]

    return cp


def symbolic_conditional_probabilities_consecutive(symX):
    """
    Computes the conditional probabilities where M[A][B] stands for the
    probability of getting B after A.

    Parameters
    ----------
    symX : Symbolic series X.
    symbols: Collection of symbols. If "None" calculated from symX

    Returns
    ----------
    Matrix with conditional probabilities

    """

    symX = np.array(symX)

    cp = symbolic_conditional_probabilities(symX[1:], symX[:-1])

    return cp


def symbolic_double_conditional_probabilities(symX, symY, symZ):
    """
    Computes the conditional probabilities where M[y][z][x] stands for the
    probability p(x|y,z).

    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symZ : Symbolic series Z.

    Returns
    ----------
    Matrix with conditional probabilities

    """

    if (len(symX) != len(symY)) or (len(symY) != len(symZ)):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)
    symZ = np.array(symZ)

    # initialize
    cp = {}
    n = {}

    for x, y, z in zip(symX, symY, symZ):
        if y in cp:
            if z in cp[y]:
                n[y][z] += 1.0
                if x in cp[y][z]:
                    cp[y][z][x] += 1.0
                else:
                    cp[y][z][x] = 1.0
            else:
                cp[y][z] = {}
                cp[y][z][x] = 1.0
                n[y][z] = 1.0
        else:
            cp[y] = {}
            n[y] = {}

            cp[y][z] = {}
            n[y][z] = 1.0

            cp[y][z][x] = 1.0

    for y in list(cp.keys()):
        for z in list(cp[y].keys()):
            for x in list(cp[y][z].keys()):
                cp[y][z][x] /= n[y][z]

    return cp


def symbolic_conditional_probabilities_consecutive_external(symX, symY):
    """
    Computes the conditional probabilities where M[yi][xi][xii] stands for the
    probability p(xii|xi,yi), where xii = x(t+1), xi = x(t) and yi = y(t).

    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symbols: Collection of symbols. If "None" calculated from symX

    Returns
    ----------
    Matrix with conditional probabilities

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)

    cp = symbolic_double_conditional_probabilities(symX[1:], symY[:-1], symX[:-1])

    return cp


def symbolic_joint_probabilities_triple(symX, symY, symZ):
    """
    Computes the joint probabilities where M[y][z][x] stands for the
    probability of coocurrence y, z and x p(y,z,x).

    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symZ : Symbolic series Z.

    Returns
    ----------
    Matrix with joint probabilities

    """

    if (len(symX) != len(symY)) or (len(symY) != len(symZ)):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)
    symZ = np.array(symZ)

    # initialize
    jp = {}
    n = len(symX)

    for x, y, z in zip(symX, symY, symZ):
        if y in jp:
            if z in jp[y]:
                if x in jp[y][z]:
                    jp[y][z][x] += 1.0 / n
                else:
                    jp[y][z][x] = 1.0 / n
            else:
                jp[y][z] = {}
                jp[y][z][x] = 1.0 / n
        else:
            jp[y] = {}
            jp[y][z] = {}
            jp[y][z][x] = 1.0 / n

    return jp


def symbolic_joint_probabilities_consecutive_external(symX, symY):
    """
    Computes the joint probabilities where M[yi][xi][xii] stands for the
    probability of ocurrence yi, xi and xii.

    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symbols: Collection of symbols. If "None" calculated from symX

    Returns
    ----------
    Matrix with joint probabilities

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)

    jp = symbolic_joint_probabilities_triple(symX[1:], symY[:-1], symX[:-1])

    return jp


def tens2num(X):
    if type(X) == torch.Tensor:
        if X.is_cuda: X = X.cpu()
        X = X.numpy()
    return X


def compute_te(X, Y):
    """for a 1d tensor"""
    X = tens2num(X)
    Y = tens2num(Y)
    symX = symbolize(X, 3)
    symY = symbolize(Y, 3)
    print(symX)
    print(len(symX))
    print(len(symY))
    MI = symbolic_mutual_information(symX, symY)

    TXY = symbolic_transfer_entropy(symX, symY)
    TYX = symbolic_transfer_entropy(symY, symX)
    TE = TYX - TXY

    print("---------------------- Random Case ----------------------")
    print("Mutual Information = " + str(MI))
    print("T(Y->X) = " + str(TXY) + "    T(X->Y) = " + str(TYX))
    print("Transfer of Entropy = " + str(TE))
    return TE


def compute_te_net(net):
    tes = []
    # count = 0
    for i, (p_name, p) in enumerate(net.named_parameters()):
        if i == 0:
            temp = p.data.cpu().numpy()
            te_val = compute_te(temp, temp)
        else:
            temp_n = p.data.cpu().numpy()
            te_val = compute_te(temp, temp_n)
            temp = temp_n
        tes.append(te_val)
    return torch.cuda.Tensor(tes)


def test_te_net():

    net = nn.Sequential(nn.Linear(100, 30), nn.Linear(30, 10), nn.Linear(10, 5))
    te_vals = compute_te_net(net)
    print(te_vals)


def main():
    X = np.random.randint(10, size=3000)
    Y = np.random.randint(10, size=3000)

    # Uncomment this for an example of a time series (Y) clearly anticipating values of X
    # Y = np.roll(X,-1)

    symX = symbolize(X, 3)
    symY = symbolize(Y, 3)

    MI = symbolic_mutual_information(symX, symY)

    TXY = symbolic_transfer_entropy(symX, symY)
    TYX = symbolic_transfer_entropy(symY, symX)
    TE = TYX - TXY

    print("---------------------- Random Case ----------------------")
    print("Mutual Information = " + str(MI))
    print("T(Y->X) = " + str(TXY) + "    T(X->Y) = " + str(TYX))
    print("Transfer of Entropy = " + str(TE))

    # Shifted Values
    X = np.random.randint(10, size=3000)
    Y = np.roll(X, -1)

    symX = symbolize(X, 3)
    symY = symbolize(Y, 3)

    MI = symbolic_mutual_information(symX, symY)

    TXY = symbolic_transfer_entropy(symX, symY)
    TYX = symbolic_transfer_entropy(symY, symX)
    TE = TYX - TXY

    print("------------------ Y anticipates X Case -----------------")
    print("Mutual Information = " + str(MI))
    print("T(Y->X) = " + str(TXY) + "    T(X->Y) = " + str(TYX))
    print("Transfer of Entropy = " + str(TE))



"""---------------- https://raw.githubusercontent.com/gregversteeg/NPEET/master/npeet/entropy_estimators.py ------------"""

#!/usr/bin/env python
# Written by Greg Ver Steeg
# See readme.pdf for documentation
# Or go to http://www.isi.edu/~gregv/npeet.html


# CONTINUOUS ESTIMATORS

def entropy(x, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = ss.cKDTree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


def centropy(x, y, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    entropy_union_xy = entropy(xy, k=k, base=base)
    entropy_y = entropy(y, k=k, base=base)
    return entropy_union_xy - entropy_y


def tc(xs, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropy(col, k=k, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropy(xs, k, base)


def ctc(xs, y, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropy(col, y, k=k, base=base) for col in xs_columns]
    return np.sum(centropy_features) - centropy(xs, y, k, base)


def corex(xs, ys, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [mi(col, ys, k=k, base=base) for col in xs_columns]
    return np.sum(cmi_features) - mi(xs, ys, k=k, base=base)


def mi(x, y, z=None, k=3, base=2):
    """ Mutual information of x and y (conditioned on z if z is not None)
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(yz, dvec), avgdigamma(z, dvec), digamma(k)
    return (-a - b + c + d) / log(base)


def cmi(x, y, z, k=3, base=2):
    """ Mutual information of x and y, conditioned on z
        Legacy function. Use mi(x, y, z) directly.
    """
    return mi(x, y, z=z, k=k, base=base)


def kldiv(x, xp, k=3, base=2):
    """ KL Divergence between p and q for x~p(x), xp~q(x)
        x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k < min(len(x), len(xp)), "Set k smaller than num. samples - 1"
    assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
    d = len(x[0])
    n = len(x)
    m = len(xp)
    const = log(m) - log(n - 1)
    tree = ss.cKDTree(x)
    treep = ss.cKDTree(xp)
    nn = query_neighbors(tree, x, k)
    nnp = query_neighbors(treep, x, k - 1)
    return (const + d * (np.log(nnp).mean() - np.log(nn).mean())) / log(base)


# DISCRETE ESTIMATORS
def entropyd(sx, base=2):
    """ Discrete entropy estimator
        sx is a list of samples
    """
    unique, count = np.unique(sx, return_counts=True, axis=0)
    # Convert to float as otherwise integer division results in all 0 for proba.
    proba = count.astype(float) / len(sx)
    # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba)) / log(base)


def midd(x, y, base=2):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    assert len(x) == len(y), "Arrays should have same length"
    return entropyd(x, base) - centropyd(x, y, base)


def cmidd(x, y, z, base=2):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    assert len(x) == len(y) == len(z), "Arrays should have same length"
    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    return entropyd(xz, base) + entropyd(yz, base) - entropyd(xyz, base) - entropyd(z, base)


def centropyd(x, y, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    return entropyd(xy, base) - entropyd(y, base)


def tcd(xs, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropyd(col, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropyd(xs, base)


def ctcd(xs, y, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropyd(col, y, base=base) for col in xs_columns]
    return np.sum(centropy_features) - centropyd(xs, y, base)


def corexd(xs, ys, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [midd(col, ys, base=base) for col in xs_columns]
    return np.sum(cmi_features) - midd(xs, ys, base)


# MIXED ESTIMATORS
def micd(x, y, k=3, base=2, warning=True):
    """ If x is continuous and y is discrete, compute mutual information
    """
    assert len(x) == len(y), "Arrays should have same length"
    entropy_x = entropy(x, k, base)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += py * entropy(x_given_y, k, base)
        else:
            if warning:
                warnings.warn("Warning, after conditioning, on y={yval} insufficient data. "
                              "Assuming maximal entropy in this case.".format(yval=yval))
            entropy_x_given_y += py * entropy_x
    return abs(entropy_x - entropy_x_given_y)  # units already applied


def midc(x, y, k=3, base=2, warning=True):
    return micd(y, x, k, base, warning)


def centropycd(x, y, k=3, base=2, warning=True):
    return entropy(x, base) - micd(x, y, k, base, warning)


def centropydc(x, y, k=3, base=2, warning=True):
    return centropycd(y, x, k=k, base=base, warning=warning)


def ctcdc(xs, y, k=3, base=2, warning=True):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropydc(col, y, k=k, base=base, warning=warning) for col in xs_columns]
    return np.sum(centropy_features) - centropydc(xs, y, k, base, warning)


def ctccd(xs, y, k=3, base=2, warning=True):
    return ctcdc(y, xs, k=k, base=base, warning=warning)


def corexcd(xs, ys, k=3, base=2, warning=True):
    return corexdc(ys, xs, k=k, base=base, warning=warning)


def corexdc(xs, ys, k=3, base=2, warning=True):
    return tcd(xs, base) - ctcdc(xs, ys, k, base, warning)


# UTILITY FUNCTIONS

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    n_elements = len(points)
    tree = ss.cKDTree(points)
    avg = 0.
    dvec = dvec - 1e-15
    for point, dist in zip(points, dvec):
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(point, dist, p=float('inf')))
        avg += digamma(num_points) / n_elements
    return avg


# TESTS

def shuffle_test(measure, x, y, z=False, ns=200, ci=0.95, **kwargs):
    """ Shuffle test
        Repeatedly shuffle the x-values and then estimate measure(x, y, [z]).
        Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs.
        'measure' could me mi, cmi, e.g. Keyword arguments can be passed.
        Mutual information and CMI should have a mean near zero.
    """
    x_clone = np.copy(x)  # A copy that we can shuffle
    outputs = []
    for i in range(ns):
        np.random.shuffle(x_clone)
        if z:
            outputs.append(measure(x_clone, y, z, **kwargs))
        else:
            outputs.append(measure(x_clone, y, **kwargs))
    outputs.sort()
    return np.mean(outputs), (outputs[int((1. - ci) / 2 * ns)], outputs[int((1. + ci) / 2 * ns)])


if __name__ == "__main__":

    print("MI between two independent continuous random variables X and Y:")
    print(mi(np.random.rand(1000, 1000), np.random.rand(1000, 300), base=2))

#    test_te_net()


    #main()


    #import torch
    #x, y = torch.randn((10, 100)).numpy(), torch.randn((10, 100)).numpy()
    #print(symbolic_mutual_information(x, y))