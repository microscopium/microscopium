"""Feature computations and other functions for Cellomics screens.
"""

import os
import collections as coll

import numpy as np

def get_img_loc(code):
    """Convert well-field-channel string to list of values.

    Parameters
    ----------
    code : string
        well-field-channel string from Cellomics filename
        e.g. 'A01f00d0' is well co-ordinate A01, field 0, channel 0

    Returns
    -------
    img_loc : list [string, int, int]
        List containing the well co-ordinate, field and image channel
        respectively.

    Examples
    --------
    >>> get_img_loc('A01f00d0')
    ['A01', 0, 0]
    """
    img_loc = [code[:3], int(code[4:6]), int(code[-1])]
    return img_loc


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
    >>> fn = ('MFGTMP_140206180002_A01f00d0.TIF')
    >>> d = cellomics_semantic_filename(fn)
    >>> d
    OrderedDict([('directory', ''), ('prefix', 'MFGTMP'), ('plate', 140206180002), ('well', 'A01'), ('field', 0), ('channel', 0), ('suffix', 'TIF')])
    """
    keys = ['directory', 'prefix', 'plate', 'well', 'field', 'channel', 'suffix']
    directory, fn = os.path.split(fn)
    filename, suffix = fn.split('.')[0], '.'.join(fn.split('.')[1:])
    filename_split = filename.split('_')
    prefix = filename_split[0]
    plate = int(filename_split[1])
    code = filename_split[2]
    values = [directory, prefix, plate] + get_img_loc(code) + [suffix]
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
