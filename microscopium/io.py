from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import h5py
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
import six
from skimage import io
import imageio as iio


### Image IO

imread = io.imread

def imread(fn, discard_alpha=True, **kwargs):
    """Read an image, but discard alpha if it is present.

    Parameters
    ----------
    fn : string
        The image filename.
    discard_alpha : bool, optional
        Only keep channels 0, 1, and 2 if image looks RGBA.

    Other Parameters
    ----------------
    **kwargs : keyword arguments
        Arguments to pass through to imread.

    Returns
    -------
    im : numpy array, shape (M, N, 3)
        The loaded image.
    """
    im = io.imread(fn, **kwargs)
    if im.ndim == 3 and im.shape[-1] == 4:
        im = im[..., :3]
    return im


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


### Temporary files

@contextmanager
def temporary_file(suffix='', directory=None):
    """Yield a writeable temporary filename that is deleted on context exit.

    Parameters
    ----------
    suffix : string, optional
        Ensure the filename ends with this suffix. Useful to specify
        file extensions. (You must include the '.' yourself.)
    directory : string, optional
        Location in the filesystem in which to write the temporary file.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import io
    >>> with temporary_file('.tif') as tempfile:
    ...     im = np.zeros((5, 5), np.uint8)
    ...     io.imsave(tempfile, im)
    ...     assert np.all(io.imread(tempfile) == im)
    ...     name = tempfile
    >>> import os
    >>> assert not os.path.isfile(name)
    """
    tempfile_stream = NamedTemporaryFile(suffix=suffix, dir=directory,
                                         delete=False)
    tempfile = tempfile_stream.name
    tempfile_stream.close()
    yield tempfile
    os.remove(tempfile)


@contextmanager
def temporary_hdf5_dataset(shape, dtype, chunks=None, directory=None):
    """Yield a temporary HDF5 dataset for on-disk array storage.

    Parameters
    ----------
    shape : tuple of int
        The shape of the dataset.
    dtype : string
        Specification of data type. Must correspond to a numpy data
        type.
    chunks : tuple of int or True, optional
        The chunk size for storing the dataset.
    directory : string, optional
        The directory for the file containing the dataset.

    Examples
    --------
    >>> shape = (4, 5)
    >>> ar = np.random.rand(*shape)
    >>> with temporary_hdf5_dataset(shape, 'float32') as dset:
    ...     dset[:] = ar
    ...     np.testing.assert_allclose(dset, ar)
    """
    with temporary_file('.hdf5', directory) as hdf5_filename:
        f = h5py.File(hdf5_filename)
        dset = f.create_dataset('temp', shape, dtype, chunks=chunks)
        yield dset
        f.close()  # no need to delete the dataset, file will be deleted.


@contextmanager
def temporary_memmap(shape, dtype, directory=None):
    """Yield a temporary numpy memory-mapped array.

    Parameters
    ----------
    shape : tuple of int
        The shape of the memory map.
    dtype : numpy dtype
        Specification of the array data type.
    directory : string, optional
        Location in which to create the temporary file.

    Yields
    ------
    ar : numpy.memmapped array
        The numpy memmap.
    """
    with temporary_file('.memmap', directory) as fname:
        mmap = np.memmap(fname, dtype=dtype, mode='w', shape=shape)
        yield mmap


def emitter_function(kind='json', out_stream=sys.stdout):
    """Return a function that 'emits' data to an output stream.

    Parameters
    ----------
    kind : {'json', 'null'}, optional
        How to output the given data.
    out_stream : buffer, optional
        Where to output the data. Must support the Python buffer interface.

    Returns
    -------
    emit : function, takes dict as input
        An emitter function to write out features.
    """
    if kind == 'null':
        def null(*args, **kwargs): pass
        return null
    if kind == 'json':
        try:
            import ujson as json
        except ImportError:
            import json

        def emit(d):
            out = json.dumps(d) + '\n'
            out_stream.write(out)
        return emit
    raise ValueError('Unknown emitter type: %s. '
                     'Valid types are "null" and "json".' % kind)


@contextmanager
def feature_container(shape, dtype=np.float, in_memory=True,
                      out_of_core='HDF5', directory=None):
    """Yield a numpy-array compatible container to store feature maps.

    Parameters
    ----------
    shape : tuple of int
        The desired shape of the container.
    dtype : numpy dtype specification, optional
        The data type of the container.
    in_memory : bool, optional
        Whether the container should be in memory (ie, a numpy array) or
        on-disk (e.g. numpy.memmap, HDF5, or bcolz)
    out_of_core : {'memmap', 'HDF5'}, optional
        When ``in_memory`` is False, use this format.
    directory : string, optional
        Location for temporary files if ``in_memory`` is ``False``.

    Yields
    ------
    ar : array-like
        A container with the desired characteristics.

    Notes
    -----
    It is a stupid limitation of numpy memmaps that they cannot be any
    bigger than 2GB. This severely limits their utility but I'm leaving
    the implementation in for posterity.
    """
    if in_memory:
        ar = np.empty(shape, dtype)
        yield ar
        del ar
    elif out_of_core == 'memmap':
        with temporary_memmap(shape, dtype) as mmap:
            yield mmap
    elif out_of_core.lower() == 'hdf5':
        with temporary_hdf5_dataset(shape, dtype) as hdf5:
            yield hdf5
