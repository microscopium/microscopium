from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import MDS

def rt_embedding(X, **kwargs):
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
    **kwargs : dict
        Keyword arguments to be passed to
        `sklearn.ensemble.RandomTreesEmbedding`

    Returns
    -------
    rt : RandomTreesEmbedding object
        The embedding object.
    X_transformed : sparse matrix
        The transformed data.
    """
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'n_jobs': -1, }
    params.update(**kwargs)
    rt = RandomTreesEmbedding().set_params(**params)
    X_transformed = rt.fit_transform(X)
    return rt, X_transformed


def mds_mapping(X, **kwargs):
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
    **kwargs : dict
        Keyword arguments to be passed to
        `sklearn.manifold.MDS`

    Returns
    -------
    mds_embedding: MDS object
        The embedding object.
    X_transformed
        error function of embedding - sum of difference
        between points in original space and new space.
    """
    params = {
        'n_components': 2,
        'max_iter': 500,
        'n_jobs': -1,
        'random_state': None, }
    params.update()
    mds_embedding = MDS().set_params(**params)
    X_transformed = mds_embedding.fit_transform(X)
    return mds_embedding, X_transformed

