import os
import functools as fun
import itertools as it
import collections as coll
import re
import numpy as np
from scipy import ndimage as nd
from skimage import io
from scipy.stats.mstats import mquantiles as quantiles
from skimage import morphology as skmorph, filter as imfilter
import skimage.filter.rank as rank
import skimage
import cytoolz as tlz


def morphop(im, operation='open', radius='5'):
    """Perform a morphological operation with spherical structuring element.

    Parameters
    ----------
    im : array, shape (M, N[, P])
        2D or 3D grayscale image.
    operation : string, optional
        The operation to perform. Choices are 'opening', 'closing',
        'erosion', and 'dilation'. Imperative verbs also work, e.g.
        'dilate'.
    radius : int, optional
        The radius of the structuring element (disk or ball) used.

    Returns
    -------
    imout : array, shape (M, N[, P])
        The transformed image.

    Raises
    ------
    ValueError : if the image is not 2D or 3D.
    """
    if im.ndim == 2:
        selem = skmorph.disk(radius)
    elif im.ndim == 3:
        selem = skmorph.ball(radius)
    else:
        raise ValueError("Image input to 'morphop' should be 2D or 3D"
                         ", got %iD" % im.ndim)
    if operation.startswith('open'):
        imout = nd.grey_opening(im, footprint=selem)
    elif operation.startswith('clos'):
        imout = nd.grey_closing(im, footprint=selem)
    elif operation.startswith('dila'):
        imout = nd.grey_dilation(im, footprint=selem)
    elif operation.startswith('ero'):
        imout = nd.grey_erosion(im, footprint=selem)
    return imout


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


def max_mask_iter(fns, offset=0, close_radius=0, erode_radius=0):
    """Find masks for a set of images having brightness artifacts.

    Parameters
    ----------
    fns : list of string
        The images being examined.
    offset : int, optional
        Offset the threshold automatically found.
    close_radius : int, optional
        Perform a morphological closing of the mask of this radius.
    erode_radius : int, optional
        Perform a morphological erosion of the mask, after any closing,
        of this radius.

    Returns
    -------
    maxes : iterator of bool array
        The max mask image corresponding to each input image.
    """
    ms = maxes(fns)
    t = imfilter.threshold_otsu(ms)
    ims = it.imap(io.imread, fns)
    masks = ((im < t + offset) for im in ims)
    if close_radius > 0:
        masks = (morphop(mask, 'close', close_radius) for mask in masks)
    if erode_radius > 0:
        masks = (morphop(mask, 'erode', erode_radius) for mask in masks)
    return masks


def write_max_masks(fns, offset=0, close_radius=0, erode_radius=0,
                    suffix='.mask.tif'):
    """Find a mask for images having a brightness artifact.

    This function iterates over a set of images and finds the maximum
    value of each. Then, Otsu's threshold is applied to the set of
    maxima, and any element brighter than this in *any* image is
    masked out.

    Parameters
    ----------
    fns : list of string
        The images being examined.
    offset : int, optional
        Offset the threshold automatically found.
    close_radius : int, optional
        Perform a morphological closing of the mask of this radius.
    erode_radius : int, optional
        Perform a morphological erosion of the mask, after any closing,
        of this radius.
    suffix : string, optional
        Save an image next to the original, with this suffix.

    Returns
    -------
    n, m : int
        The number of images for which a mask was created, and the
        total number of images
    """
    masks = max_mask_iter(fns, offset, close_radius, erode_radius)
    n = 0
    m = 0
    for fn, mask in it.izip(fns, masks):
        outfn = basefn(fn) + suffix
        m += 1
        if not mask.all():
            # we multiply by 255 to make the image easy to look at
            io.imsave(outfn, mask.astype(np.uint8) * 255)
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
    ims = it.imap(io.imread, fns)
    maxes = np.array(map(np.max, ims))
    return maxes


def stretchlim(im, bottom=0.01, top=None, mask=None):
    """Stretch the image so new image range corresponds to given quantiles.

    Parameters
    ----------
    im : array, shape (M, N, [...,] P)
        The input image.
    bottom : float, optional
        The lower quantile.
    top : float, optional
        The upper quantile. If not provided, it is set to 1 - `bottom`.
    mask : array of bool, shape (M, N, [...,] P), optional
        Only consider intensity values where `mask` is ``True``.

    Returns
    -------
    out : np.ndarray of float
        The stretched image.
    """
    if mask is None:
        mask = np.ones(im.shape, dtype=bool)
    if top is None:
        top = 1. - bottom
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
        ims = map(io.imread, sorted(fns))
        im = quadrant_stitch(*ims)
        io.imsave(new_filename, im)
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
    >>> fn_numbering = it.product(range(2), range(1, 4))
    >>> fns = ['image_%i_w%i.tif' % (i, j) for i, j in fn_numbering]
    >>> fns
    ['image_0_w1.tif', 'image_0_w2.tif', 'image_0_w3.tif', 'image_1_w1.tif', 'image_1_w2.tif', 'image_1_w3.tif']
    >>> sorted(group_by_channel(fns).items())
    [('w1', ['image_0_w1.tif', 'image_1_w1.tif']), ('w2', ['image_0_w2.tif', 'image_1_w2.tif']), ('w3', ['image_0_w3.tif', 'image_1_w3.tif'])]
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
    ['image_0_s1_w1.TIF', 'image_0_s2_w1.TIF', 'image_0_s3_w1.TIF', 'image_0_s4_w1.TIF', 'image_1_s1_w1.TIF', 'image_1_s2_w1.TIF', 'image_1_s3_w1.TIF', 'image_1_s4_w1.TIF']
    >>> sorted(group_by_quadrant(fns).items())
    [(('image_0', 'w1'), ['image_0_s1_w1.TIF', 'image_0_s2_w1.TIF', 'image_0_s3_w1.TIF', 'image_0_s4_w1.TIF']), (('image_1', 'w1'), ['image_1_s1_w1.TIF', 'image_1_s2_w1.TIF', 'image_1_s3_w1.TIF', 'image_1_s4_w1.TIF'])]
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
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [2, 2, 2, 3, 3, 3],
           [2, 2, 2, 3, 3, 3]])
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

    Examples
    --------
    >>> im = np.array([0., 0.5, 1.])
    >>> rescale_to_11bits(im)
    array([   0, 1024, 2047], dtype=uint16)
    """
    im11 = np.round(im_float * 2047.).astype(np.uint16)
    return im11


def rescale_from_11bits(im11):
    """Rescale a uint16 image with range in [0, 2047] to float in [0., 1.]

    Parameters
    ----------
    im11 : array of uint16, range in [0, 2047]
        The input image, encoded in uint16 but having 11-bit range.

    Returns
    -------
    imfloat : array of float, same shape as `im11`
        The output image.

    Examples
    --------
    >>> im = np.array([0, 1024, 2047], dtype=np.uint16)
    >>> rescale_from_11bits(im)
    array([ 0.    ,  0.5002,  1.    ])

    Notes
    -----
    Designed to be a no-op with the above `rescale_to_11bits` function,
    although this is subject to approximation errors.
    """
    return np.round(im11 / 2047., decimals=4)


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


def _reduce_with_count(pairwise, iterator, accumulator=None):
    """Return both the result of the reduction and the number of elements.

    Parameters
    ----------
    pairwise : function (a -> b -> a)
        The function with which to reduce the `iterator` sequence.
    iterator : iterable
        The sequence being reduced.
    accumulator : type "a", optional
        An initial value with which to perform the reduction.

    Returns
    -------
    result : type "a"
        The result of the reduce operation.
    count : int
        The number of elements that were accumulated.

    Examples
    --------
    >>> x = [5, 6, 7]
    >>> _reduce_with_count(np.add, x)
    (18, 3)
    """
    def new_pairwise(a, b):
        (elem1, c1), (elem2, c2) = a, b
        return pairwise(elem1, elem2), c2
    new_iter = it.izip(iterator, it.count(1))
    new_acc = (0, accumulator)
    return tlz.reduce(new_pairwise, new_iter, new_acc)


def find_background_illumination(fns, radius=51, quantile=0.05,
                                 stretch_quantile=0., method='mean'):
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
    method : 'mean', 'average', 'median', or 'histogram', optional
        How to use combine the smoothed intensities of the input images
        to infer the illumination field:

        - 'mean' or 'average': Use the mean value of the smoothed
        images at each pixel as the illumination field.
        - 'median': use the median value. Since all images need to be
        in-memory to compute this, use only for small sets of images.
        - 'histogram': use the median value approximated by a
        histogram. This can be computed on-line for large sets of
        images.

    Returns
    -------
    illum : np.ndarray, float, shape (M, N)
        The estimated illumination over the image field.

    See Also
    --------
    ``correct_image_illumination``.
    """
    # This function follows the "PyToolz" streaming data model to
    # obtain the illumination estimate. First, define each processing
    # step:
    read = io.imread
    normalize = (tlz.partial(stretchlim, bottom=stretch_quantile)
                 if stretch_quantile > 0
                 else skimage.img_as_float)
    rescale = rescale_to_11bits
    pad = fun.partial(skimage.util.pad, pad_width=radius, mode='reflect')
    rank_filter = fun.partial(rank.percentile, selem=skmorph.disk(radius),
                              p0=quantile)
    _unpad = fun.partial(unpad, pad_width=radius)
    unscale = rescale_from_11bits

    # Next, compose all the steps, apply to all images (streaming)
    bg = (tlz.pipe(fn, read, normalize, rescale, pad, rank_filter, _unpad,
                   unscale)
          for fn in fns)

    # Finally, reduce all the images and compute the estimate
    if method == 'mean' or method == 'average':
        illum, count = _reduce_with_count(np.add, bg)
        illum = skimage.img_as_float(illum) / count
    elif method == 'median':
        illum = np.median(list(bg), axis=0)
    elif method == 'histogram':
        raise NotImplementedError('histogram background illumination method '
                                  'not yet implemented.')
    else:
        raise ValueError('Method "%s" of background illumination finding '
                         'not recognised.' % method)

    return illum


def correct_image_illumination(im, illum, stretch_quantile=0, mask=None):
    """Divide input image pointwise by the illumination field.

    Parameters
    ----------
    im : np.ndarray of float
        The input image.
    illum : np.ndarray of float, same shape as `im`
        The illumination field.
    stretch_quantile : float, optional
        Stretch the image intensity to saturate the top and bottom
        quantiles given.
    mask : array of bool, same shape as im, optional
        Only stretch the image intensity where `mask` is ``True``.

    Returns
    -------
    imc : np.ndarray of float, same shape as `im`
        The corrected image.
    """
    if im.dtype != np.float:
        imc = skimage.img_as_float(im)
    else:
        imc = im.copy()
    imc /= illum
    lim = stretch_quantile
    imc = stretchlim(imc, lim, 1-lim, mask)
    return imc

