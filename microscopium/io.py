from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import six
from six.moves import range
from skimage import io
try:
    import imageio as iio
except ImportError:
    iio = io


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


def cat_channels(ims, order=[2, 0, 1]):
    """From a sequence of single-channel images, produce multichannels.

    Suppose the input is a list:

    ```
    ims = [green1, blue1, red1, green2, blue2, red2]
    ```

    Then the output will be:

    ```
    [rgb1, rgb2]
    ```

    (The order of channels in the list is arbitrary; the default is
    based on the data for which this software was created.)

    Parameters
    ----------
    ims : list of arrays
        A list of images in which consecutive images represent single
        channels of the same image. (See example.)
    order : list of int, optional
        The order in which the channels appear.

    Returns
    -------
    multi : iterator of arrays
        A list of the images composed into multi-channel images.
    """
    nchannels = len(order)
    while True:
        channels = [next(ims) for i in range(nchannels)]
        print(len(channels))
        print([(c.min(), c.max()) for c in channels])
        try:
            channels_sorted = [channels[j] for j in order]
        except IndexError:
            raise StopIteration()
        channel_im = np.concatenate([c[..., np.newaxis] for
                                     c in channels_sorted], axis=-1)
        yield channel_im

