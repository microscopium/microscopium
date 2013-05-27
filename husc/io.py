import os
import numpy as np
import Image


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
    """
    fn = os.path.expanduser(fn)
    if 0 <= ar.max() <= 1 and ar.dtype == np.double:
        bitdepth = 16 if None else bitdepth
        imdtype = np.uint16 if bitdepth == 16 else np.uint8
        ar = ((2**bitdepth-1)*ar).astype(imdtype)
    if 1 < ar.max() < 256 and bitdepth == None or bitdepth == 8:
        mode = 'L'
        mode_base = 'L'
        ar = ar.astype(np.uint8)
    elif 256 <= np.max(ar) < 2**16 and bitdepth == None or \
                                                bitdepth == 16:
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

