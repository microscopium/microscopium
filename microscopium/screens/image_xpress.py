"""Feature computations and other functions for Image Xpress screens
"""
import os
import collections as coll
import re
from .. import features as feat
from .. import _util


def ix_semantic_filename(fn):
    """Split an ImageXpress filename into its annotated components.

    Parameters
    ----------
    fn : string
        A filename from the ImageXpress high-content screening system.

    Returns
    -------
    semantic : collections.OrderedDict {string: string}
        A dictionary mapping the different components of the filename.
    """
    keys = ['directory', 'prefix', 'plate', 'well', 'field', 'channel',
            'suffix']

    directory, fn = os.path.split(fn)
    fn, suffix = fn.split('.', 1)

    fn_regex = re.search(r'([^\W_]{0,})(?:_{0,})([A-P]\d+)_s(\d+)_w(\d{1})',
                         fn)
    # finds last set of contiguous digits after underscore
    dir_regex = re.search(r'_(\d+)(?!.*_(\d+))', directory)

    prefix, well, field, channel = map(lambda x: fn_regex.group(x),
                                       range(1, 5))
    plate = int(dir_regex.group(1))

    values = [directory, prefix, int(plate), well,
              _util.int_or_none(field), _util.int_or_none(channel), suffix]

    semantic = coll.OrderedDict(zip(keys, values))

    # molecular devices 1-index their channel and field values,
    # subtract 1 if these fields exist
    for key in ["channel", "field"]:
        if semantic[key] is not None:
            semantic[key] -= 1

    return semantic


def filename2coord(fn):
    """Obtain (plate, well) coordinates from a filename.

    Parameters
    ----------
    fn : string
        The input filename. This must include a directory with the
        the plate number has these aren't coded in IX files.

    Returns
    -------
    coord : (int, string, int) tuple
        The (plate, well, cell) coordinates of the image.

    Examples
    --------
    >>> fn = "./Week1_22123/G10_s2_w11C3B9BCC-E48F-4C2F-9D31-8F46D8B5B972.tif"
    >>> filename2coord(fn)
    (22123, 'G10')
    """
    sem = ix_semantic_filename(fn)
    return sem["plate"], sem["well"]


def filename2id(fn):
    """Get a string representation of (plate, well), from filename.

    Parameters
    ----------
    fn : string
        Filename of a standard Cellomics screen image.

    Returns
    -------
    id_ : string
        The string ID.
    Examples
    --------
    >>> fn = ("./Week4_27481/Week1_22123/G10_s2_w11C3B9BCC-"
    ...       "E48F-4C2F-9D31-8F46D8B5B972.tif")
    >>> filename2id(fn)
    '22123-G10'
    """
    id_ = '-'.join(map(str, filename2coord(fn)))
    return id_


feature_map = feat.default_feature_map
