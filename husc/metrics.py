from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.spatial.distance import pdist, squareform

def sq_to_dist(i, j, n):
    """Convert coordinate of square distance matrix to condensed matrix index.

    The condensed version of a squareform, pairwise distance matrix is
    a linearisation of the upper triangular, non-diagonal coordinates
    of the squareform distance matrix. This function returns the [i, j]-th
    coordinate of the condensed array.

    eg. given a squareform matrix,

    array([[  0.        ,  10.        ,  22.36067977],
           [ 10.        ,   0.        ,  14.14213562],
           [ 22.36067977,  14.14213562,   0.        ]])

    The condensed version of this matrix is:

    array([ 10.        ,  22.36067977,  14.14213562])

    Parameters
    ----------
    i : int
        i-th coordinate.
    j : int
        j-th coordinate.
    n : int
        Dimension n of n*n distance matrix.

    Returns
    -------
    index : int
        Position of pairwise distance [i, j] in
        condensed distance matrix.

    Examples
    --------
    >>> sq_to_dist(1, 2, 3)
    2

    """
    index = i*n + j - i*(i+1)/2 - i - 1
    return index


def gene_distance_score(X, gene_list, metric='euclidean'):
    """Find intra/extra gene distance scores between samples.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data matrix.
    gene_list : array, shape (n_samples, )
        Array indicating which gene was knocked-down
        corresponding to each sample.
    metric : string, optional
        Which distance measure to use when calculating distances.
        Must be one of the options allowable in
        scipy.spatial.distance.pdist. Default euclidean.

    Returns
    -------
    all_intragene_data : array
        An 1D array with intra-gene distances (i.e. distances
        between samples with the same gene knocked down).
    all_extragene_data : array
        An 1D array with extra-gene distances (i.e. distances
        between samples with the same gene knocked down).

    Examples
    --------
    ## TODO make this doctest much prettier or make it a full test
    >>> data = np.zeros((6, 3))
    >>> data[0:2, :] = 1
    >>> data[2:4, :] = 4
    >>> data[4:6, :] = 7
    >>> genes = ['A', 'A', 'B', 'B', 'C', 'C']
    >>> intra, extra = gene_distance_score(data, genes, 'euclidean')
    >>> intra
    array([ 0.,  0.,  0.])
    >>> extra
    array([  5.19615242,   5.19615242,  10.39230485,  10.39230485,
             5.19615242,   5.19615242,  10.39230485,  10.39230485,
             5.19615242,   5.19615242,   5.19615242,   5.19615242])
    """
    all_intragene_index = []
    gene_index = defaultdict(list)

    for key, value in zip(gene_list, range(len(gene_list))):
        gene_index[key].append(value)

    for key in gene_index:
        for i, j in combinations(gene_index[key], 2):
            all_intragene_index.append(sq_to_dist(i, j, X.shape[0]))

    n = sq_to_dist(X.shape[0], X.shape[0], X.shape[0])
    all_extragene_index = np.setdiff1d(np.arange(n), all_intragene_index)

    all_intragene_data = pdist(X, 'euclidean')[all_intragene_index]
    all_extragene_data = pdist(X, 'euclidean')[all_extragene_index]

    return all_intragene_data, all_extragene_data

