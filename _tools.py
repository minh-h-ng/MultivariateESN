# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)
import numbers

import numpy as np
import scipy


def permute(A):
    '''Get a permutation to diagonalize and scale a matrix
    Parameters
    ----------
    A : ndarray, shape (n_features, n_features)
        A matrix close from a permutation and scale matrix.
    Returns
    -------
    A : ndarray, shape (n_features, n_features)
        A permuted matrix.
    '''
    A = A.copy()
    n = A.shape[0]
    idx = np.arange(n)
    done = False
    while not done:
        done = True
        for i in range(n):
            for j in range(i):
                if A[i, i] ** 2 + A[j, j] ** 2 < A[i, j] ** 2 + A[j, i] ** 2:
                    A[(i, j), :] = A[(j, i), :]
                    idx[i], idx[j] = idx[j], idx[i]
                    done = False
    A /= np.diag(A)
    order_sort = np.argsort(np.sum(np.abs(A), axis=0))
    A = A[order_sort, :]
    A = A[:, order_sort]
    return A


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = scipy.linalg.eigh(np.dot(W, W.T))
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)


def _ica_par(X, fun, max_iter, w_init, verbose):
    """Parallel FastICA.
    Used internally by FastICA --main loop
    """
    if verbose:
        print('Running %d iterations of FastICA...' % max_iter)
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = fun.score_and_der(np.dot(W, X))
        g_wtx = g_wtx.mean(axis=1)
        W = _sym_decorrelation(np.dot(gwtx, X.T) / p_ -
                               g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
    if verbose:
        print('Running Picard...')
    return W