"""Feature computations and other functions for Cellomics screens.
"""

import os
import collections as coll
from skimage import io
import numpy as np


def listdir_fullpath(dir):
    """Extended os.listdir that lists the full directory instead of just the
    filename.

    Parameters
    ----------
    dir : string
        The query directory.

    Returns
    -------
    path : list of string
        A list of files in the query directory with their full path.
    """
    path = [os.path.join(dir, fn) for fn in os.listdir(dir)]
    return path


def batch_snail_stitch(dict, dir):
    """Run snail stitch over a dictionary of filenames and output to dir.

    Parameters
    ----------
    dict : dict
        A dictionary of (plate, well) co-ordinates mapped to filenames.
        Dictionary built with ``make_wellchannel2file_dict`` function.
    dir : string
        The directory to output the files to.
    """
    for key, value in dict.iteritems():
        fn0 = value[0]
        fn0 = os.path.split(fn0)[1]
        fn0_fn, fn_ext = os.path.splitext(fn0)
        new_fn = [fn0_fn, '_stitched', fn_ext]
        new_fn = ''.join(new_fn)
        stitched_image = run_snail_stitch(value)
        io.imsave(os.path.join(dir, new_fn), rescale_12bit(stitched_image))


def rescale_12bit(image, bit='8'):
    """Rescale a 12bit image.

    Cellomics TIFs ar encoded as 12bit TIF files, which generally cannot
    be viewed in most software. This function rescales the images to either an
    8 or 16 bit TIF file.

    Parameters
    ----------
    image : array, shape (M, N)
        The image to be rescaled.

    Returns
    -------
    scale_image : array, shape (M, N)
        The rescaled image.
    """
    if bit == '8':
        scale_image = np.round((image/4095.) * 255).astype(np.uint8)
    elif bit == '16':
        scale_image = np.round((image/4095.) * 65535).astype(np.uint16)
    else:
        scale_image = np.round(image/4095.).astype(np.float)
    return scale_image


def run_snail_stitch(fns):
    """Run right, clockwise spiral/snail stitching of 25 Cellomics TIFs.

    Runs clockwise stitching of the images. The spiral begins in the
    center of the image, moves to the right and circles around in a clockwise
    manner.

    Parameters
    ----------
    fns : list of array, shape (M, N)
        The list of 25 image files to be stitched together.

    Returns
    -------
    stitched_image : array, shape (5*M, 5*N)
        The stitched image.
    """
    right = [[20, 21, 22, 23, 24],
             [19, 6, 7, 8, 9],
             [18, 5, 0, 1, 10],
             [17, 4, 3, 2, 11],
             [16, 15, 14, 13, 12]]

    stitched_image = np.array([])
    for i in range(0, 5):
        stitched_row = np.array([])
        for j in range(0, 5):
            image = io.imread(fns[right[i][j]])
            stitched_row = concatenate(stitched_row, image)
        stitched_image = stack(stitched_image, stitched_row)
    return stitched_image


def stack(arr1, arr2):
    """No docstring here as this function wil be removed soon.
    """
    if arr1.shape[0] == 0:
        return arr2
    else:
        return np.vstack((arr1, arr2))


def concatenate(arr1, arr2):
    """No docstring here as this function wil be removed soon.
    """
    if arr1.shape[0] == 0:
        return arr2
    else:
        return np.concatenate((arr1, arr2), 1)


def make_wellchannel2file_dict(fns):
    """Return a dictionary mapping well co-ordinates to filenames.

    Returns a dictionary where key are (plate, well) co-ordinates and
    values are lists of images corresponding to that plate and well.

    Parameters
    ----------
    fns : list of string
        A list of Cellomics TIF files.

    Returns
    -------
    wellchannel2file : dict {tuple : list of string}
        The dictionary mapping the (plate, well) co-ordinate to
        a list of files corresponding to that well.
    """
    wellchannel2file = coll.defaultdict(list)
    for fn in fns:
        fn_base = os.path.basename(fn)
        file_info = cellomics_semantic_filename(fn_base)
        tuple = (file_info['well'], file_info['channel'])
        wellchannel2file[tuple].append(fn)
    return wellchannel2file


def get_by_ext(dir, extension, full=True, sort=True):
    """Return list of files in directory with specified extension.

    Parameters
    ----------
    dir : string
        A directory containing files.
    extension : string
        Return only files with this extension.
    full : bool
        Whether or not to return files with the path included. Default
        true.
    sort : bool
        Whether or not to sort the list of files before returning them.
        Default true.
    """
    if full is True:
        fns = listdir_fullpath(dir)
    else:
        fns = os.listdir(dir)
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
