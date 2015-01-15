
def groupby(key, iterable, transform=None):
    """Group items in `iterable` in a dictionary according to `key`.

    Parameters
    ----------
    key : function
        Returns a hashable when given an item in `iterable`.
    iterable : tuple, list, generator, or array-like
        The items to be grouped.
    transform : function, optional
        Transform the items before grouping them.

    Returns
    -------
    grouped : dict
        A dictionary mapping keys to elements of iterable. This has
        the form:
            ``{key(elem): [transform(elem)] for elem in iterable}``
    """
    if transform is None:
        transform = lambda x: x
    grouped = {}
    for elem in iterable:
        grouped.setdefault(key(elem), []).append(transform(elem))
    return grouped

