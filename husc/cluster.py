from sklearn.ensemble import RandomTreesEmbedding

def rt_embedding(X, n_estimators=100, max_depth=10, n_jobs=-1, **kwargs):
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
    rt = RandomTreesEmbedding(**kwargs)
    X_transformed = rt.fit_transform(X)
    return rt, X_transformed
