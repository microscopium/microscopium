import numpy as np
import numbers


def generate_spiral(shape, direction, clockwise=False):
    """Generate sequential integers in a spiral pattern.

    Many HCS systems use a spiral pattern to capture multiple fields in
    for each well. This function generates a corresponding spiral pattern of
    integers to be used to map and montage individual fields into a single
    image.

    Parameters
    ----------
    shape : int, or 2-tuple of int
        The final shape of the spiral. If shape is scalar, the function will
        return convert it into a 2-tuple with the same value repeated.
    direction : str in {'up', 'down', 'left', 'right'}
        The direction the first step the spiral takes when leaving the origin.
    clockwise : bool, optional
        If the spiral generated should be in a clockwise direction.

    Returns
    -------
    spiral_array : array, shape (shape, shape)
        The spiral array.

    Examples
    --------
    >>> generate_spiral(3, 'up', clockwise=True)
    array([[8, 1, 2],
           [7, 0, 3],
           [6, 5, 4]], dtype=uint8)
    >>> generate_spiral((2, 3), 'left', clockwise=True)
    array([[2, 3, 4],
           [1, 0, 5]], dtype=uint8)
    """
    if np.isscalar(shape):
        shape = (shape, shape)
    if len(shape) != 2:
        ndim = len(shape)
        mesg = (f'generate_spiral only works in 2D, but an {ndim}-D shape '
                 'was given.')
        raise ValueError(mesg)
    if shape[0] < 1 or shape[1] < 1:
        mesg = ('Shape passed to generate_spiral should always be positive, '
                f'but {shape} was given.')
        raise ValueError(mesg)

    if abs(shape[0] - shape[1]) > 1:
        mesg = ('generate_spiral requires shapes that differ from square by '
                f'at most 1, but {shape} was given. For example, (2, 3) and '
                '(4, 4) are valid shapes, but (3, 5) is not.')
        raise ValueError(mesg)

    directions = {'down': [1, 0], 'up': [-1, 0],
                  'left': [0, -1], 'right': [0, 1]}
    if direction not in directions:
        mesg = ('direction in generate_spirals should be one of up, down, '
                f'left, or right, but {direction} was given.')
        raise ValueError(mesg)

    size = np.prod(shape)

    di, dj = directions[direction]
    segment_length = 1
    i = j = segment_passed = 0
    rcenter, ccenter = shape[0] // 2, shape[1] // 2
    spiral_array = np.zeros(shape, dtype=np.min_scalar_type(size))

    for k in range(1, size):
        i += di
        j += dj
        segment_passed += 1
        spiral_array[rcenter + i, ccenter + j] = k

        if segment_passed == segment_length:
            segment_passed = 0
            if clockwise:
                di, dj = dj, -di
            else:
                di, dj = -dj, di
            if ((direction in {'left', 'right'} and di == 0)
                    or (direction in {'up', 'down'} and dj == 0)):
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


def int_or_none(n):
    """Returns input n cast as int, or None if input is none.

    This is used in parsing sample information from image filenames.

    Parameters
    ----------
    n : any value castable as int, None
        The input value.

    Returns
    -------
    The input value cast an int, or None.

    Examples
    --------
    >>> int_or_none(3.0)
    3
    >>> int_or_none(None)
    """
    return int(n) if n is not None else None
