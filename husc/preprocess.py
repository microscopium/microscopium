import functools as fun
import itertools as it
import re
import numpy as np
import scipy.ndimage as nd
from skimage import feature, color, io as imio, img_as_float, \
    morphology as skmorph
import skimage.filter.rank as rank

from .io import imwrite

def lab_hist(rgb_image, **kwargs):
    return np.histogram(color.rgb2lab(rgb_image), **kwargs)


full_feature_list = \
    [fun.partial(np.histogram, bins=16, range=(0.0, 1.0)),
    fun.partial(lab_hist, bins=16, range=(0.0, 1.0)),
    feature.hog
    ]
    # TO-DO: add segmentation features


def run_quadrant_stitch(fns, re_string='(.*)_(s[1-4])_(w[1-3]).TIF',
                        re_quadrant_group=1):
    """Read images, stitched them, and write out to same directory.

    Parameters
    ----------
    fns : list of string
        The filenames to be processed.
    re_string : string, optional
        The regular expression to match the filename.
    re_quadrant_group : int, optional
        The group from the re.match object that will contain quadrant info.

    Returns
    -------
    None
    """
    qd = group_by_quadrant(fns, re_string, re_quadrant_group)
    for fn_pattern, fns in qd.items():
        new_filename = '_'.join(fn_pattern) + '_stitched.tif'
        ims = map(imio.imread, sorted(fns))
        im = quadrant_stitch(*ims)
        imwrite(im, new_filename)


def group_by_quadrant(fns, re_string='(.*)_(s[1-4])_(w[1-3]).TIF',
                      re_quadrant_group=1):
    """Group filenames by quadrant to prepare for stitching.

    Parameters
    ----------
    fns : list of string
        The filenames to be processed.
    re_string : string, optional
        The regular expression to match the filename.
    re_quadrant_group : int, optional
        The group from the re.match object that will contain quadrant info.

    Returns
    -------
    grouped : dict mapping tuple of string to tuple of string
        The filenames, grouped into tuples containing four quadrants of the
        same image. The keys are all the regular expression match groups
        *other* than the quadrant group, useful for composing a filename for
        the stitched images.

    Examples
    --------
    >>> fn_numbering = it.product(range(2), range(1, 5))
    >>> fns = ['image_%i_s%i_w1.TIF' % (i, j) for i, j in fn_numbering]
    >>> fns
    ['image_0_s1_w1.TIF',
     'image_0_s2_w1.TIF',
     'image_0_s3_w1.TIF',
     'image_0_s4_w1.TIF',
     'image_1_s1_w1.TIF',
     'image_1_s2_w1.TIF',
     'image_1_s3_w1.TIF',
     'image_1_s4_w1.TIF']
    >>> group_by_quadrant(fns)
    {('image_0', 'w1'): ['image_0_s1_w1.TIF',
      'image_0_s2_w1.TIF',
      'image_0_s3_w1.TIF',
      'image_0_s4_w1.TIF'],
     ('image_1', 'w1'): ['image_1_s1_w1.TIF',
      'image_1_s2_w1.TIF',
      'image_1_s3_w1.TIF',
      'image_1_s4_w1.TIF']}
    """
    re_match = fun.partial(re.match, re_string)
    match_objs = map(re_match, fns)
    fns = [fn for fn, match in zip(fns, match_objs) if match is not None]
    match_objs = filter(lambda x: x is not None, match_objs)
    matches = map(lambda x: x.groups(), match_objs)
    keys = map(tuple, [[m[i] for i in range(len(m)) if i != re_quadrant_group]
                                                        for m in matches])
    grouped = {}
    for k, fn in zip(keys, fns):
        grouped.setdefault(k, []).append(fn)
    return grouped


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


def find_background_illumination(im_iter, radius=51, quantile=0.05):
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
    qfilter = fun.partial(rank.percentile, selem=selem, p0=quantile)
    bg_iter = it.imap(qfilter, im_iter)
    im0 = bg_iter.next()
    accumulator = np.zeros(im0.shape, float)
    bg_iter = it.chain([im0], bg_iter)
    counter = it.count()
    illum = reduce(lambda x, y: x[0] + y[0],
                   it.izip(bg_iter, counter), accumulator)
    n_images = counter.next()
    illum /= n_images
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
