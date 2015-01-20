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

    Examples
    --------
    >>> sq_to_dist(1, 2, 3)
    1

    """
    index = n*j - j*(j+1)/2 + i - 1 - j
    return int(index)

def mongo_group_by(collection, group_by):
    """Group MongoDB collection according to specified field.

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
                        # add cell_plate_barcode and well for each document
                        # belonging to the group
                        'cell_plate_barcode': '$cell_plate_barcode',
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
                new_coord = (coord['cell_plate_barcode'], str(coord['well']))
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

    all_intragene_index = []
    for key in gene_dict:
        if len(gene_dict[key]) > 1:
            indices = map(X.index.get_loc, gene_dict[key])
            for i, j in combinations(indices, 2):
                all_intragene_index.append(sq_to_dist(i, j, X.shape[0]))

    all_intragene_index.sort()
    n = sq_to_dist(X.shape[0], X.shape[0], X.shape[0])
    all_intergene_index = np.setdiff1d(np.arange(n), all_intragene_index,
                                       assume_unique=True)
    distance = pdist(X, metric)
    all_intragene_data = distance[all_intragene_index]
    all_intergene_data = distance[all_intergene_index]
    return all_intragene_data, all_intergene_data
