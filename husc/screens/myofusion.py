"""Feature computations and other functions for the Marcelle myoblast
fusion screen.
"""

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
    all_fs, all_names = []
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
    cells_t_adapt = cells > threshold_adaptive(cells)
    fs, names = features.nuclei_per_cell_histogram(nuclei_mean, cells_t_otsu)
    all_fs.append(fs)
    all_names.extend(['otsu-' + name for name in names])
    fs, names = features.nuclei_per_cell_histogram(nuclei_mean, cells_t_adapt)
    all_fs.append(fs)
    all_names.extend(['adapt-' + name for name in names])
    return np.concatenate(all_fs), all_names

