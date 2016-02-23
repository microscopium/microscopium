import numpy as np
import numbers


def generate_spiral(dim, direction, clockwise=False):
    """Generate an dim * dim of sequental integers in a spiral pattern.

    Many HCS systems use a spiral pattern to capture multiple fields in
    for each well. This function generates a spiral pattern of integers
    to be used by the montaging functions.

    Parameters
    ----------
    dim : int
        The dimension of the spiral. The function will return a dim * dim
        array.
    direction : str
        The direction the first step the spiral takes when leaving the origin.
        Should be one of up, down, left or right.
    clockwise : bool, optional
        The direction of the spiral. Default clockwise.

    Returns
    -------
    spiral_array : array, shape (dim, dim)
        The spiral array.

    Examples
    --------
    >>> generate_spiral(3, "up", True)
    array([[8, 1, 2],
           [7, 0, 3],
           [6, 5, 4]], dtype=uint8)
    """
    if dim <= 0:
        raise ValueError("dim must be a positive integer.")

    if dim % 2 != 1:
        raise ValueError("dim must be an odd integer.")

    if direction not in ["down", "up", "left", "right"]:
        raise ValueError("direction must be one of down, up, left or right.")

    directions = {
        "down": [1, 0],
        "up": [-1, 0],
        "left": [0, -1],
        "right": [0, 1]
    }

    di, dj = directions[direction]
    segment_length = 1
    i = j = segment_passed = 0
    center = np.floor(dim / 2).astype(np.uint8)
    spiral_array = np.zeros((dim, dim), dtype=np.uint8)

    for k in range(1, dim ** 2):
        i += di
        j += dj
        segment_passed += 1
        spiral_array[center + i][center + j] = k

        if segment_passed == segment_length:
            segment_passed = 0
            if clockwise:
                di, dj = dj, -di
            else:
                di, dj = -dj, di
            if direction in ["left", "right"]:
                if di == 0:
                    segment_length += 1
            else:
                if dj == 0:
                    segment_length += 1

    return spiral_array


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

    Examples
    --------
    >>> g = groupby(lambda x: x % 2, range(6), lambda x: x ** 2)
    >>> sorted(g.keys())
    [0, 1]
    >>> g[0]
    [0, 4, 16]
    >>> g[1]
    [1, 9, 25]
    """
    if transform is None:
        transform = lambda x: x
    grouped = {}
    for elem in iterable:
        grouped.setdefault(key(elem), []).append(transform(elem))
    return grouped


def normalise_random_state(seed):
    """Turn seed into a numpy RandomState instance.

    Parameters
    ----------
    seed : None, int, or numpy RandomState instance
        The input seed/random state.

    Returns
    -------
    random : numpy RandomState instance.
        The resulting RandomState instance.

    Notes
    -----
    This function is copied almost verbatim from scikit-learn's
    ``sklearn.utils.validation.check_random_state``, and so is
    governed by that library's code license (New BSD at the time of
    this writing.)
    """
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError('Invalid input %s to generate random state' % seed)
