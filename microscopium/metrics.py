from __future__ import absolute_import, division
from itertools import combinations
import numpy as np
from scipy.spatial.distance import pdist
from six.moves import map

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

    Reference
    ---------

    In the scipy.spatial.squareform documentation, it is shown that the
    index in the condensed array is given by
    {n choose 2} - {(n - i) choose 2} + (j - i - 1).
    Some simple arithmetic shows that this can be expanded to the formula below.
    The documentation can be found in the following link:

    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.squareform.html


    Examples
    --------
    >>> sq_to_dist(0, 1, 4)
    0
    >>> sq_to_dist(0, 3, 4)
    2
    >>> sq_to_dist(1, 2, 4)
    3

    """
    if i > j:
        i, j = j, i
    index = i * n + j - i * (i + 1) / 2 - i - 1
    return int(index)

def mongo_group_by(collection, group_by):
    """
    Group MongoDB collection according to specified field.

    Sends aggregate query to MongoDB collection to group
    all documents by a given field and returns dictionary
    mapping the field to the corresponding (plate, well)
    co-ordinate(s).

    Parameters
    ----------
    collection : pymongo collection
        Pymongo object directing to collection.
    group_by : string
        Field to group collection by.
    Returns
    -------
    query_dict : dict { string : list of tuple }
        Query dictionary mapping the specified group_by field to a list of
        (plate, well) co-ordinates.
    """
    mongo_query = collection.aggregate([{
            '$group' : {
                # groups all documents according to specified field
                '_id': '$' + group_by,
                'coords': {
                    '$addToSet': {
                        # add plate and well for each document
                        # belonging to the group
                        'plate': '$plate',
                        'well': '$well'
                    }
                }
            }
    }])['result']

    query_dict = {}
    for doc in mongo_query:
        query_dict[doc['_id']] = []
        for coord in doc['coords']:
            try:
                new_coord = (int(coord['plate']), str(coord['well']))
                query_dict[doc['_id']].append(new_coord)
            except KeyError:
                pass
    return query_dict


def gene_distance_score(X, collection, metric='euclidean'):
    """Find intra/inter gene distance scores between samples.

    Parameters
    ----------
    X : Data frame, shape (n_samples, n_features)
        Feature data frame.

    metric : string, optional
        Which distance measure to use when calculating distances.
        Must be one of the options allowable in
        scipy.spatial.distance.pdist. Default is euclidean distance.

    Returns
    -------
    all_intragene_data : array
        An 1D array with intra-gene distances (i.e. distances
        between samples with the same gene knocked down).
    all_intergene_data : array
        An 1D array with inter-gene distances (i.e. distances
        between samples with different gene knocked down).

    """
    gene_dict = mongo_group_by(collection, 'gene_name')
    nsamples = X.shape[0]
    npairs = int(nsamples * (nsamples - 1) / 2)

    all_intragene_index = []
    for key in gene_dict:
        if len(gene_dict[key]) > 1:
            indices = (X.index.get_loc(coord) for coord in gene_dict[key]
                       if coord in X.index)
            for i, j in combinations(indices, 2):
                all_intragene_index.append(sq_to_dist(i, j, X.shape[0]))

    all_intragene_index.sort()
    all_intergene_index = np.setdiff1d(np.arange(npairs), all_intragene_index,
                                       assume_unique=True)
    distance = pdist(X, metric)
    all_intragene_data = distance[all_intragene_index]
    all_intergene_data = distance[all_intergene_index]
    return all_intragene_data, all_intergene_data
