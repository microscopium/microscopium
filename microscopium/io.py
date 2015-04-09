from __future__ import absolute_import
from __future__ import print_function
import six
from skimage import io
try:
    import imageio as iio
except ImportError:
    iio = io


imread = io.imread


def imsave(fn, im, **kwargs):
    """Wrapper around various libraries that haven't got their act together.

    For TIFF, uses scikit-image's wrapper around tifffile.py. For other
    formats, uses imageio.

    Parameters
    ----------
    fn : string
        The filename to save to.
    im : array, shape (M, N[, 3])
        The image to save.
    kwargs : dict, optional
        Keyword arguments to the save function. Format dependent. For example,
        JPEG images take a ``quality`` (int) argument in [1, 95], while
        TIFF images take a ``compress`` (int) argument in [0, 9].

    Notes
    -----
    The ``fn`` and ``im`` arguments can be swapped -- the function will
    determine which to use by testing for string types.
    """
    if isinstance(im, six.string_types):
        fn, im = im, fn
    if fn.endswith('.tif'):
        io.imsave(fn, im, plugin='tifffile', **kwargs)
    else:
        iio.imsave(fn, im, **kwargs)

imwrite = imsave


