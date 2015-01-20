from __future__ import absolute_import
import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import MDS

def rt_embedding(X, n_estimators=100, max_depth=10, n_jobs=-1):
    """Embed data matrix X in a random forest.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data matrix.
    n_estimators : int, optional
        The number of trees in the embedding.
    max_depth : int, optional
        The maximum depth of each tree.
    n_jobs : int, optional
        Number of compute jobs when fitting the trees. -1 means number
        of processors on the current computer.

    Returns
    -------
    rt : RandomTreesEmbedding object
        The embedding object.
    X_transformed : sparse matrix
        The transformed data.
    """
    rt = RandomTreesEmbedding(n_estimators=n_estimators, max_depth=max_depth,
                              n_jobs=n_jobs)
    X_transformed = rt.fit_transform(X)
    return rt, X_transformed


def dbscan_clustering(X, eps=0.5, min_samples=5, metric='euclidean',
                      random_state=None):
    """`DBSCAN` clustering applied to data matrix X

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data matrix.
    eps : float, optional
        The maximum distance between two samples for them to be
        considered as in the same neighborhood.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be
        considered as a core point.
    metric : string, optional
        The distance metric to use when calculating pairwise distances.
        Must be one of the options allowable in
        `sklearn.metrics.pairwise.pairwise_distances`. Default euclidean.
    random_state : int, optional
        Generator used to initialize, set fixed integer to
        reproduce results for debugging.

    Returns
    -------
    dbscan_clustered : DBSCAN object
        The clustering object.
    core_samples : array of int, shape (n_samples,)
        Row indices of core samples in data matrix X.
    membership: array of int, shape (n_samples,)
        1D array where each element represents which cluster
        each sample was assigned to. -1 represents noisy/unassigned
        sample.
    """
    dbscan_clustered = DBSCAN(X, eps=eps, min_samples=min_samples,
                              metric=metric, random_state=random_state)
    core_samples = dbscan_clustered.components_
    membership = dbscan_clustered.labels_
    return dbscan_clustered, core_samples, membership


def kmeans_clustering(X, n_clusters=None, max_iter=300, n_init=10,
                      random_state=None):
    """Mini-Batch K-Means clustering applied to data matrix X

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data matrix.
    n_clusters : int, optional
        The number of clusters. Default is floor(sqrt(n_samples/2)).
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm.
    n_init : int, optional
        Number of time the k-means algorithm will be run with different\
        centroid seeds.
    random_state : int, optional
        Generator used to initialize, set fixed integer to
        reproduce results for debugging.

    Returns
    -------
    kmeans_clustered : KMeans object
        The clustering object.
    centroids : array , shape (n_clusters, n_features)
        The centroids for each cluster.
    membership : array of int, shape (n_samples,)
        1D array where each element represents which cluster sample
        assigned to.

    Examples
    --------
    >>> data = np.random.rand(50, 5)
    >>> kmeans, centroids, membership = kmeans_clustering(data, n_clusters=6)
    >>> centroids.shape
    (6, 5)
    >>> membership.shape
    (50,)
    """
    if n_clusters is None:
        n_clusters = int(np.floor(np.sqrt(X.shape[0]/2)))
    kmeans_clustered = MiniBatchKMeans(
        n_clusters=n_clusters, max_iter=max_iter, n_init=n_init,
        random_state=random_state)
    kmeans_clustered.fit(X)
    centroids = kmeans_clustered.cluster_centers_
    membership = kmeans_clustered.labels_
    return kmeans_clustered, centroids, membership


def mds_mapping(X, n_components=2, max_iter=500, n_jobs=-1,
                random_state=None):
    """MDS scaling applied to data matrix X

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data matrix
    n_components : int, optional
        Dimensionality of the reduced mapping
    max_iter : int, optional
        Max number of iterations
    n_jobs: int, optional
        Number of compute jobs when fitting scale. -1 means number
        of processors on the current computer.
    random_state : int, optional
        Generator used to initialize, set fixed integer to
        reproduce results for debugging.

    Returns
    -------
    mds_embedding: MDS object
        The embedding object.
    X_transformed : array, shape (n_samples, n_components)
        The transformed data.

    Examples
    --------
    >>> data = np.random.rand(5, 10)
    >>> MDS_reduced, transformed_data = mds_mapping(data, n_components=3)
    >>> transformed_data.shape
    (5, 3)
    """
    mds_embedding = MDS(n_components=n_components, max_iter=max_iter,
                        n_jobs=n_jobs, random_state=random_state)
    mds_embedding.fit_transform(X)
    X_transformed = mds_embedding.embedding_

    return mds_embedding, X_transformed

