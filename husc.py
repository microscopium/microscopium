import functools as fun
import itertools as it
import numpy as np
import scipy.ndimage as nd
from skimage import feature, color, io as imio, img_as_float, \
    morphology as skmorph, filter as imfilter

def lab_hist(rgb_image, **kwargs):
    return np.histogram(color.rgb2lab(rgb_image), **kwargs)


full_feature_list = \
    [fun.partial(np.histogram, bins=16, range=(0.0, 1.0)),
    fun.partial(lab_hist, bins=16, range=(0.0, 1.0)),
    feature.hog
    ]
    # TO-DO: add segmentation features


def quadrant_stitch(nw, ne, sw, se):
    """Stitch four seamless quadrant images into a single big image.

    Parameters
    ----------
    nw, ne, sw, se : np.ndarray, shape (M, N)
        The four quadrant images, corresponding to the cardinal directions of
        north-west, north-east, south-west, south-east.

    Returns
    -------
    stitched : np.ndarray, shape (2*M, 2*N)
        The image resulting from stitching the four input images

    Examples
    --------
    >>> imbase = np.ones((2, 3), int)
    >>> nw, ne, sw, se = [i * imbase for i in range(4)]
    >>> quadrant_stitch(nw, ne, sw, se)
    array([[0, 0, 0, 2, 2, 2],
           [0, 0, 0, 2, 2, 2],
           [1, 1, 1, 3, 3, 3],
           [1, 1, 1, 3, 3, 3]])
    """
    s = nw.shape
    stitched = np.zeros((2 * s[0], 2 * s[1]), nw.dtype)
    stitched[:s[0], :s[1]] = nw
    stitched[:s[0], -s[1]:] = ne
    stitched[-s[0]:, :s[1]] = sw
    stitched[-s[0]:, -s[1]:] = se
    return stitched


def find_background_illumination(im_iter, radius=25, quantile=0.05):
    """Use a set of related images to find uneven background illumination.

    Parameters
    ----------
    im_iter : iterable of np.ndarray, uint8 or uint16 type, shape (M, N)
        An iterable of grayscale images. skimage's rank filter will be used on
        the images, so the max value is limited to 4095.
    radius : int, optional
        The radius of the structuring element used to find background. 
        default: 51
    quantile : float in [0, 1], optional
        The desired quantile to find background. default: 0.05

    Returns
    -------
    illum : np.ndarray, float, shape (M, N)
        The estimated illumination over the image field.
    """
    selem = skmorph.disk(radius)
    qfilter = fun.partial(imfilter.rank.percentile, selem=selem, p0=quantile)
    bg_iter = it.imap(qfilter, im_iter)
    im0 = bg_iter.next()
    accumulator = np.zeros(im0.shape, float)
    illum = reduce(lambda x, y: x + y, it.chain([im0], bg_iter), accumulator)
    return illum


def find_cells(p_background, background_threshold=0.9, min_cell_size=100,
                             watershed_merge_threshold=0):
    """Segment out cells in an nD image of the probability of background."""
    background = find_background(p_background, background_threshold)
    cells = nd.label(True - background)[0]
    cells = skmorph.remove_small_connected_components(cells, 
                                                      min_cell_size, True)
    distances = nd.distance_transform_edt(cells)
    cells = morpho.watershed(distances.max() - distances,
            mask=cells.astype(bool))
    return cells


def find_background(p, threshold=0.9):
    """Obtain the largest connected component of points above threshold."""
    b = p > threshold
    bccs = nd.label(b)[0]
    real_cc = np.argmax(np.bincount(bccs.ravel()))
    b[bccs != real_cc] = False
    return b

def expand_cell_centroids(centroids, p_background):
    pass

def image_feature_vector(im, feature_list=None):
    if type(im) == str:
        im = img_as_float(imio.imread(im))
    if feature_list is None:
        feature_list = full_feature_list
    features = np.concatenate([f(im) for f in feature_list])
    return features
