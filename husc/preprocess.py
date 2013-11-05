import os
import functools as fun
import itertools as it
import collections as coll
import re
import numpy as np
import mahotas as mh
from scipy.stats.mstats import mquantiles as quantiles
from skimage import (io as imio, img_as_float, morphology as skmorph,
                     filter as imfilter)
import skimage.filter.rank as rank
from skimage.util import pad

from .io import imwrite


def basefn(fn):
    """Get the filename without the extension.

    Parameters
    ----------
    fn : string
        A filename.

    Returns
    -------
    outfn : string
        `fn` with the extension stripped.
    """
    return os.path.splitext(fn)[0]


def max_mask_iter(fns, offset=0):
    """Find masks for a set of images having brightness artifacts.

    Parameters
    ----------
    fns : list of string
        The images being examined.
    offset : int, optional
        Offset the threshold automatically found.

    Returns
    -------
    maxes : iterator of bool array
        The max mask image corresponding to each input image.
    """
    ms = maxes(fns)
    t = imfilter.threshold_otsu(ms)
    ims = it.imap(mh.imread, fns)
    masks = ((im < t + offset) for im in ims)
    return masks


def write_max_masks(fns, suffix='.mask.tif', offset=0):
    """Find a mask for images having a brightness artifact.

    This function iterates over a set of images and finds the maximum
    value of each. Then, Otsu's threshold is applied to the set of
    maxima, and any element brighter than this in *any* image is
    masked out.

    Parameters
    ----------
    fns : list of string
        The images being examined.
    suffix : string, optional
        Save an image next to the original, with this suffix.
    offset : int, optional
        Offset the threshold automatically found.

    Returns
    -------
    n, m : int
        The number of images for which a mask was created, and the
        total number of images
    """
    masks = max_mask_iter(fns, offset)
    n = 0
    m = 0
    for fn, mask in it.izip(fns, masks):
        outfn = basefn(fn) + suffix
        m += 1
        if not mask.all():
            # we multiply by 255 to make the image easy to look at
            mh.imsave(outfn, mask.astype(np.uint8) * 255)
            n += 1
    return n, m


def maxes(fns):
    """Return an array of the maximum intensity of each image.

    Parameters
    ----------
    fns : list of string
        The filenames of the images.

    Returns
    -------
    maxes : 1D array
        The maximum value of each image examined.
    """
    ims = it.imap(mh.imread, fns)
    maxes = np.array(map(np.max, ims))
    return maxes


def stretchlim(im, bottom=0.01, top=0.99, mask=None):
    """Stretch the image so new image range corresponds to given quantiles.

    Parameters
    ----------
    im : array, shape (M, N, [...,] P)
        The input image.
    bottom : float, optional
        The lower quantile.
    top : float
        The upper quantile.
    mask : array of bool, shape (M, N, [...,] P), optional
        Only consider intensity values where `mask` is ``True``.

    Returns
    -------
    out : np.ndarray of float
        The stretched image.
    """
    if mask is None:
        mask = np.ones(im.shape, dtype=bool)
    im = im.astype(float)
    q0, q1 = quantiles(im[mask], [bottom, top])
    out = (im - q0) / (q1 - q0)
    out[out < 0] = 0
    out[out > 1] = 1
    return out


def run_quadrant_stitch(fns, re_string='(.*)_(s[1-4])_(w[1-3]).*',
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


def group_by_channel(fns, re_string='(.*)_(w[1-3]).*',
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
    >>> fns = ['image_%i_w%i.TIF' % (i, j) for i, j in fn_numbering]
    >>> fns
    ['image_0_w1.tif',
     'image_0_w2.tif',
     'image_0_w3.tif',
     'image_1_w1.tif',
     'image_1_w2.tif',
     'image_1_w3.tif']
    >>> group_by_channel(fns)
    {('w1'): ['image_0_w1.tif', 'image_1_w1.tif'],
     ('w2'): ['image_0_w2.tif', 'image_1_w2.tif'],
     ('w3'): ['image_0_w3.tif', 'image_1_w3.tif']}
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


def group_by_quadrant(fns, re_string='(.*)_(s[1-4])_(w[1-3]).*',
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
    nw, ne, sw, se : np.ndarray, shape (Mi, Ni)
        The four quadrant images, corresponding to the cardinal directions of
        north-west, north-east, south-west, south-east.

    Returns
    -------
    stitched : np.ndarray, shape (M0+M2, N0+N1)
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
    x1 = nw.shape[0]
    x2 = sw.shape[0]
    y1 = nw.shape[1]
    y2 = ne.shape[1]
    stitched = np.zeros((x1 + x2, y1 + y2), nw.dtype)
    stitched[:x1, :y1] = nw
    stitched[:x1, y1:] = ne
    stitched[x1:, :y1] = sw
    stitched[x1:, y1:] = se
    return stitched


def rescale_to_11bits(im_float):
    """Rescale a float image in [0, 1] to integers in [0, 2047].

    This operation makes rank filtering much faster.

    Parameters
    ----------
    im_float : array of float in [0, 1]
        The float image. The range and type are *not* checked prior to
        conversion!

    Returns
    -------
    im11 : array of uint16 in [0, 2047]
        The converted image.
    """
    im11 = np.floor(im_float * 2047.9999).astype(np.uint16)
    return im11


def unpad(im, pad_width):
    """Remove padding from a padded image.

    Parameters
    ----------
    im : array
        The input array.
    pad_width : int or sequence of int
        The width of padding: a number for the same width along each
        dimension, or a sequence for different widths.

    Returns
    -------
    imc : array
        The unpadded image.
    """
    if not isinstance(pad_width, coll.Iterable):
        pad_width = [pad_width] * im.ndim
    slices = tuple([slice(p, -p) for p in pad_width])
    return im[slices]


def find_background_illumination(fns, radius=51, quantile=0.05,
                                 stretch_quantile=0.0, mask=True,
                                 mask_offset=0):
    """Use a set of related images to find uneven background illumination.

    Parameters
    ----------
    fns : list of string
        A list of image file names
    radius : int, optional
        The radius of the structuring element used to find background.
        default: 51
    quantile : float in [0, 1], optional
        The desired quantile to find background. default: 0.05
    stretch_quantile : float in [0, 1], optional
        Stretch image to full dtype limit, saturating above this quantile.
    mask : bool, optional
        Whether to automatically mask brightness artifacts in the images.
    mask_offset : int, optional
        Offset the mask threshold automatically found.

    Returns
    -------
    illum : np.ndarray, float, shape (M, N)
        The estimated illumination over the image field.
    """
    im0 = mh.imread(fns[0])
    im_iter = (mh.imread(fn) for fn in fns)
    if stretch_quantile > 0:
        im_iter = (stretchlim(im, stretch_quantile, 1 - stretch_quantile) for
                   im in im_iter)
    else:
        im_iter = it.imap(img_as_float, im_iter)
    if mask:
        mask_iter1 = max_mask_iter(fns, mask_offset)
        mask_iter2 = max_mask_iter(fns, mask_offset)
    im_iter = it.imap(rescale_to_11bits, im_iter)
    pad_image = fun.partial(pad, pad_width=radius, mode='reflect')
    im_iter = it.imap(pad_image, im_iter)
    mask_iter1 = it.imap(pad_image, mask_iter1)
    selem = skmorph.disk(radius)
    bg_iter = (rank.percentile(im, selem, mask=mask, p0=quantile) for
               im, mask in it.izip(im_iter, mask_iter1))
    bg_iter = (unpad(im, pad_width=radius) for im in bg_iter)
    illum = np.zeros(im0.shape, float)
    counter = np.zeros(im0.shape, float)
    for bg, mask in it.izip(bg_iter, mask_iter2):
        illum[mask] += bg[mask]
        counter[mask] += 1
    illum /= counter
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

