import functools as fun
import numpy as np
from scipy.stats.mstats import mquantiles
from scipy import ndimage as nd
from skimage import feature, color, io as imio, img_as_float, \
    morphology as skmorph
from skimage import filter as imfilter, measure



def lab_hist(rgb_image, **kwargs):
    return np.histogram(color.rgb2lab(rgb_image), **kwargs)


# threshold and labeling number of objects, statistics about object size and
# shape
def intensity_object_features(im, adaptive_t_radius=51):
    """Segment objects based on intensity threshold and compute properties.

    Parameters
    ----------
    im : 2D np.ndarray of float or uint8.
        The input image.
    adaptive_t_radius : int, optional
        The radius to calculate background with adaptive threshold.

    Returns
    -------
    f : 1D np.ndarray of float
        The feature vector.
    """
    tim1 = im > imfilter.threshold_otsu(im)
    f1 = object_features(tim1, im)
    tim2 = imfilter.threshold_adaptive(im, adaptive_t_radius)
    f2 = object_features(tim2, im)
    f = np.concatenate([f1, f2])
    return f


def object_features(bin_im, im, erode=2):
    """Compute features about objects in a binary image.

    Parameters
    ----------
    bin_im : 2D np.ndarray of bool
        The image of objects.
    im : 2D np.ndarray of float or uint8
        The actual image.
    erode : int, optional
        Radius of erosion of objects.

    Returns
    -------
    f : 1D np.ndarray of float
        The feature vector.
    """
    selem = skmorph.disk(erode)
    if erode > 0:
        bin_im = nd.binary_erosion(bin_im, selem)
    lab_im, n_objs = nd.label(bin_im)
    if erode > 0:
        lab_im = nd.grey_dilation(lab_im, footprint=selem)
    feats = measure.regionprops(lab_im,
                                ['Area', 'Eccentricity', 'EulerNumber',
                                 'Extent', 'MinIntensity', 'MeanIntensity',
                                 'MaxIntensity', 'Solidity'],
                                intensity_image=im)
    feats = np.array([props.values() for props in feats], np.float)
    feature_quantiles = mquantiles(feats, [0.05, 0.25, 0.5, 0.75, 0.95],
                                   axis=0)
    f = np.concatenate([np.array([n_objs], np.float),
                        feature_quantiles.ravel()])
    return f


full_feature_list = \
    [fun.partial(np.histogram, bins=16, range=(0.0, 1.0)),
    fun.partial(lab_hist, bins=16, range=(0.0, 1.0)),
    feature.hog
    ]
    # TO-DO: add segmentation features


def image_feature_vector(im, feature_list=None):
    if type(im) == str:
        im = img_as_float(imio.imread(im))
    if feature_list is None:
        feature_list = full_feature_list
    features = np.concatenate([f(im) for f in feature_list])
    return features
