from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
from PIL import Image
from six.moves import range


def imwrite(ar, fn, bitdepth=None):
    """Write a np.ndarray 2D volume to a .png or .tif image

    Parameters
    ----------
    ar : numpy ndarray, shape (M, N)
        The volume to be written to disk.
    fn : string
        The file name to which to write the volume.
    bitdepth : {8, 16, 32}, optional
        The bits per pixel.

    Returns
    -------
    None : None
        No value is returned.

    Notes
    -----
    The syntax `imwrite(fn, ar)` is also supported.
    """
    if type(fn) == np.ndarray and type(ar) == str:
        ar, fn = fn, ar
    fn = os.path.expanduser(fn)
    if 0 <= ar.max() <= 1 and ar.dtype == np.double:
        bitdepth = 16 if bitdepth is None else bitdepth
        imdtype = np.uint16 if bitdepth == 16 else np.uint8
        ar = ((2**bitdepth-1)*ar).astype(imdtype)
    if 1 < ar.max() < 256 and (bitdepth == None or bitdepth == 8):
        mode = 'L'
        mode_base = 'L'
        ar = ar.astype(np.uint8)
    elif 256 <= np.max(ar) < 2**16 and (bitdepth == None or \
                                                bitdepth == 16):
        mode = 'I;16'
        mode_base = 'I'
        ar = ar.astype(np.uint16)
    else:
        mode = 'RGBA'
        mode_base = 'RGBA'
        ar = ar.astype(np.uint32)
    im = Image.new(mode_base, ar.T.shape)
    im.fromstring(ar.tostring(), 'raw', mode)
    im.save(fn)


imsave = imwrite


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

