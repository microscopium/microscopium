import functools as fun
import itertools as it
import re
import numpy as np
from scipy.stats.mstats import mquantiles as quantiles
from skimage import io as imio, img_as_float, \
    morphology as skmorph, img_as_ubyte
import skimage.filter.rank as rank

from .io import imwrite


def stretchlim(im, bottom=0.01, top=0.99):
    """Stretch the image so new image range corresponds to given quantiles.

    Parameters
    ----------
    im : np.ndarray
        The input image.
    bottom : float, optional
        The lower quantile.
    top : float
        The upper quantile.

    Returns
    -------
    out : np.ndarray of float
        The stretched image.
    """
    im = im.astype(float)
    q0, q1 = quantiles(im, [bottom, top])
    out = (im - q0) / (q1 - q0)
    out[out < 0] = 0
    out[out > 1] = 1
    return out


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
    fns_out : list of string
        The output filenames
    """
    qd = group_by_quadrant(fns, re_string, re_quadrant_group)
    fns_out = []
    for fn_pattern, fns in qd.items():
        new_filename = '_'.join(fn_pattern) + '_stitched.tif'
        ims = map(imio.imread, sorted(fns))
        im = quadrant_stitch(*ims)
        imwrite(im, new_filename)
        fns_out.append(new_filename)
    return fns_out


def crop(im, slices=(slice(100, -100), slice(250, -300))):
    """Crop an image to contain only plate interior.

    Parameters
    ----------
    im : array
        The image to be cropped.
    slices : tuple of slice objects, optional
        The slices defining the crop. The default values are for
        stitched images from the Marcelle screen.

    Returns
    -------
    imc : array
        The cropped image.
    """
    return im[slices]


def group_by_channel(fns, re_string='(.*)_(w[1-3])_stitched.tif',
                      re_channel_group=1):
    """Group filenames by channel to prepare for illumination estimation.

    Intended to be run *after* quadrant stitching.

    Parameters
    ----------
    fns : list of string
        The filenames to be processed.
    re_string : string, optional
        The regular expression to match the filename.
    re_quadrant_group : int, optional
        The group from the re.match object that will contain channel info.

    Returns
    -------
    grouped : dict mapping tuple of string to list of string
        The filenames, grouped into lists containing all images of the same
        channel. The keys are the channel regular expression group, useful for
        composing a filename for the illumination image.

    Examples
    --------
    >>> fn_numbering = it.product(range(2), range(1, 5))
    >>> fns = ['image_%i_s1_w%i.TIF' % (i, j) for i, j in fn_numbering]
    >>> fns
    ['image_0_w1_stitched.tif',
     'image_0_w2_stitched.tif',
     'image_0_w3_stitched.tif',
     'image_1_w1_stitched.tif',
     'image_1_w2_stitched.tif',
     'image_1_w3_stitched.tif']
    >>> group_by_channel(fns)
    {('w1'): ['image_0_w1_stitched.tif', 'image_1_w1_stitched.tif'],
     ('w2'): ['image_0_w2_stitched.tif', 'image_1_w2_stitched.tif'],
     ('w3'): ['image_0_w3_stitched.tif', 'image_1_w3_stitched.tif']}
    """
    re_match = fun.partial(re.match, re_string)
    match_objs = map(re_match, fns)
    fns = [fn for fn, match in zip(fns, match_objs) if match is not None]
    match_objs = filter(lambda x: x is not None, match_objs)
    matches = map(lambda x: x.groups(), match_objs)
    keys = [m[re_channel_group] for m in matches]
    grouped = {}
    for k, fn in zip(keys, fns):
        grouped.setdefault(k, []).append(fn)
    return grouped


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


def find_background_illumination(im_iter, radius=51, quantile=0.05,
                                 stretch_quantile=0.0):
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
    stretch_quantile : float in [0, 1], optional
        Stretch image to full dtype limit, saturating above this quantile.

    Returns
    -------
    illum : np.ndarray, float, shape (M, N)
        The estimated illumination over the image field.
    """
    im_iter = (stretchlim(im, stretch_quantile, 1 - stretch_quantile) for
               im in im_iter)
    im_iter = (img_as_ubyte(im) for im in im_iter)
    selem = skmorph.disk(radius)
    qfilter = fun.partial(rank.percentile, selem=selem, p0=quantile)
    bg_iter = it.imap(qfilter, im_iter)
    im0 = bg_iter.next()
    accumulator = np.zeros(im0.shape, float)
    bg_iter = it.chain([im0], bg_iter)
    counter = it.count()
    illum = reduce(lambda x, y: x + y[0],
                   it.izip(bg_iter, counter), accumulator)
    n_images = counter.next()
    illum /= n_images
    return illum


def correct_image_illumination(im, illum):
    """Divide input image pointwise by the illumination field.

    Parameters
    ----------
    im : np.ndarray of float
        The input image.
    illum : np.ndarray of float, same shape as `im`
        The illumination field.

    Returns
    -------
    imc : np.ndarray of float, same shape as `im`
        The corrected image.
    """
    if im.dtype != np.float:
        imc = img_as_float(im)
    else:
        imc = im.copy()
    imc /= illum
    imc = stretchlim(imc, 0.001, 0.999)
    return imc

