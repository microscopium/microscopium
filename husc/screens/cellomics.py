"""Feature computations and other functions for the Cellomics myoblast
fusion screen.
"""

import os
import collections as coll

import numpy as np

# getting a start on some of these
# using a (plate, well, cell) convention
# for 3x3 imaging convention used in these
# datasets

def cellomics_semantic_filename(fn):
    """Split a Cellomics filename into its annotated components.

    Parameters
    ----------
    fn : string
        A filename from the Cellomics high-content screening system.

    Returns
    -------
    semantic : collections.OrderedDict {string: string}
        A dictionary mapping the different components of the filename.
        NOTE: the 'plate' key is converted to an int.

    Examples
    --------
    >>> fn = ('cellomics-p1-j01-110210_02490688_53caa10e-ac15-4166-9b9d-'
    ...       '4b1167f3b9c6_C04_s1_w1.TIF')
    >>> d = cellomics_semantic_filename(fn)
    >>> d
    OrderedDict([('directory', ''), ('prefix', 'cellomics'), ('pass', 'p1'), ('job', 'j01'), ('date', '110210'), ('plate', 2490688), ('barcode', '53caa10e-ac15-4166-9b9d-4b1167f3b9c6'), ('well', 'C04'), ('quadrant', 's1'), ('channel', 'w1'), ('suffix', 'TIF')])
    """
    keys = ['directory', 'prefix', 'pass', 'job', 'date', 'plate',
            'barcode', 'well', 'quadrant', 'channel', 'suffix']
    directory, fn = os.path.split(fn)
    filename, suffix = fn.split('.')[0], '.'.join(fn.split('.')[1:])
    values = filename.split('_')
    full_prefix = values[0].split('-')
    if len(full_prefix) > 4:
        head, tail = full_prefix[:3], full_prefix[3:]
        full_prefix = head + ['-'.join(tail)]
    values = [directory] + full_prefix + values[1:] + [suffix]
    semantic = coll.OrderedDict(zip(keys, values))
    try:
        semantic['plate'] = int(semantic['plate'])
    except ValueError: # Some plates are labeled "NOCODE"
        semantic['plate'] = np.random.randint(1000000)
    return semantic


def filename2coord(fn):
    """Obtain (plate, well, cell) coordinates from a filename.

    Parameters
    ----------
    fn : string
        The input filename

    Returns
    -------
    coord : (int, string, int) tuple
        The (plate, well, cell) coordinates of the image.

    Examples
    --------
    >>> fn = ('cellomics-p1-j01-110210_02490688_53caa10e-ac15-4166-9b9d-'
    ...       '4b1167f3b9c6_C04_s1_w1.TIF')
    >>> filename2coord(fn)
    (2490688, 'C04')
    """
    sem = cellomics_semantic_filename(fn)
    return (sem['plate'], sem['well'])


def dir2plate(dirname):
    """Return a Plate ID from a directory name.

    Parameters
    ----------
    dirname : string
        A directory containing export images from an HCS plate.

    Returns
    -------
    plateid : int
        The plate ID parsed from the directory name.
    """
    basedir = os.path.split(dirname)[1]
    plateid = basedir.split('_')[1]
    try:
        plateid = int(plateid)
    except ValueError:
        print("Plate ID %s cannot be converted to int, replaced with 0." %
              plateid)
        return 0
    return plateid


if __name__ == '__main__':
    import doctest
    doctest.testmod()
