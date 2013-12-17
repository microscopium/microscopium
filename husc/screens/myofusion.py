"""Feature computations and other functions for the Marcelle myoblast
fusion screen.
"""

import os
import collections as coll

import numpy as np
from scipy import ndimage as nd
from skimage.filter import threshold_otsu, threshold_adaptive

from .. import features


def feature_vector_from_rgb(image):
    """Compute a feature vector from the composite images.

    The channels are assumed to be in the following order:
     - Red: MCF-7
     - Green: cytoplasm GFP
     - Blue: DAPI/Hoechst

    Parameters
    ----------
    im : array, shape (M, N, 3)
        The input image.

    Returns
    -------
    fs : 1D array of float
        The features of the image.
    names : list of string
        The feature names.
    """
    all_fs, all_names = [], []
    ims = np.rollaxis(image, -1, 0)
    mcf, cells, nuclei = ims
    prefixes = ['mcf', 'cells', 'nuclei']
    for im, prefix in zip(ims, prefixes):
        fs, names = features.intensity_object_features(im)
        names = [prefix + '-' + name for name in names]
        all_fs.append(fs)
        all_names.extend(names)
    nuclei_mean = nd.label(nuclei > np.mean(nuclei))[0]
    fs, names = features.nearest_neighbors(nuclei_mean)
    all_fs.append(fs)
    all_names.extend(['nuclei-' + name for name in names])
    mcf_mean = nd.label(mcf)[0]
    fs, names = features.fraction_positive(nuclei_mean, mcf_mean,
                                           positive_name='mcf')
    all_fs.append(fs)
    all_names.extend(names)
    cells_t_otsu = cells > threshold_otsu(cells)
    cells_t_adapt = cells > threshold_adaptive(cells, 51)
    fs, names = features.nuclei_per_cell_histogram(nuclei_mean, cells_t_otsu)
    all_fs.append(fs)
    all_names.extend(['otsu-' + name for name in names])
    fs, names = features.nuclei_per_cell_histogram(nuclei_mean, cells_t_adapt)
    all_fs.append(fs)
    all_names.extend(['adapt-' + name for name in names])
    return np.concatenate(all_fs), all_names


feature_map = feature_vector_from_rgb


def myores_semantic_filename(fn):
    """Split a MYORES filename into its annotated components.

    Parameters
    ----------
    fn : string
        A filename from the MYORES high-content screening system.

    Returns
    -------
    semantic : collections.OrderedDict {string: string}
        A dictionary mapping the different components of the filename.

    Examples
    --------
    >>> fn = ('MYORES-p1-j01-110210_02490688_53caa10e-ac15-4166-9b9d-'
              '4b1167f3b9c6_C04_s1_w1.TIF')
    >>> d = myores_semantic_filename(fn)
    >>> d
    OrderedDict([('directory', ''), ('prefix', 'MYORES'), ('pass', 'p1'), ('job', 'j01'), ('date', '110210'), ('plate', '02490688'), ('barcode', '53caa10e-ac15-4166-9b9d-4b1167f3b9c6'), ('well', 'C04'), ('quadrant', 's1'), ('channel', 'w1'), ('suffix', '')])
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
    return semantic


