"""Feature computations and other functions for Cellomics screens.
"""

import os
import collections as coll
from skimage import io
import numpy as np


def run_snail_stitch(fns):
    """Run right, anti-clockwise spiral/snail stitching of 25 Cellomics TIFs.
    """
    # TODO finish docstring
    # TODO generalise to other directions and other field sizes.
    right = [[16, 15, 14, 13, 12],
             [17, 4, 3, 2, 11],
             [18, 5, 0, 1, 10],
             [19, 6, 7, 8, 9],
             [20, 21, 22, 23, 24]]

    stitched_image = np.array([])
    for i in range(0, 5):
        stitched_row = np.array([])
        print stitched_image.shape
        for j in range(0, 5):
            print fns[right[i][j]]
            image = io.imread(fns[right[i][j]])
            stitched_row = concatenate(stitched_row, image, 1)
        stitched_image = concatenate(stitched_image, stitched_row, 0)
    return stitched_image


def concatenate(arr1, arr2, axis=0):
    """concatenate that doesn't complain when arr1 is null.
    """
    # TODO write docstring
    if arr1.shape[0] == 0:
        return arr2
    else:
        return np.concatenate((arr1, arr2), axis=axis)


def make_wellchannel2file_dict(fns):
    """Return a dictionary mapping well co-ordinates to filenames
    """
    # TODO finish docstring
    wellchannel2file = coll.defaultdict(list)
    for fn in fns:
        file_info = cellomics_semantic_filename(fn)
        tuple = (file_info['well'], file_info['channel'])
        wellchannel2file[tuple].append(fn)
    return wellchannel2file


def get_by_ext(dirname, extension, sort=True):
    """Return list of files in directory with specified extension

    Parameters
    ----------
    dirname : string
        A directory containing files.
    """
    # TODO finish docstring
    fns = os.listdir(dirname)
    print fns
    fns_ext = []
    for fn in fns:
        if fn.endswith('.' + extension):
            fns_ext.append(fn)
    if sort is True:
        fns_ext.sort()
        return fns_ext
    else:
        return fns_ext


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

    Examples
    --------
    >>> fn = ('MFGTMP_140206180002_A01f00d0.TIF')
    >>> d = cellomics_semantic_filename(fn)
    >>> d
    OrderedDict([('directory', ''), ('prefix', 'MFGTMP'), ('plate', 140206180002), ('well', 'A01'), ('field', 0), ('channel', 0), ('suffix', 'TIF')])
    """
    keys = ['directory', 'prefix', 'plate', 'well', 'field', 'channel', 'suffix']
    directory, fn = os.path.split(fn)
    filename, suffix = fn.split('.')[0], '.'.join(fn.split('.')[1:])
    prefix, plate, code = filename.split('_')
    well = code[:3]
    field = int(code[4:6])
    channel = int(code[-1])
    values = [directory, prefix, int(plate), well, field, channel, suffix]
    semantic = coll.OrderedDict(zip(keys, values))
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
    >>> fn = 'MFGTMP_140206180002_A01f00d0.TIF'
    >>> filename2coord(fn)
    (140206180002, 'A01')
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

    Examples
    --------
    >>> dir2plate('MFGTMP_140206180002')
    140206180002
    """
    basedir = os.path.split(dirname)[1]
    plateid = int(basedir.split('_')[1])
    return plateid


if __name__ == '__main__':
    import doctest
    doctest.testmod()
