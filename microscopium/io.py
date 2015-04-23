from __future__ import absolute_import
from __future__ import print_function
import os
import h5py
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
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
    >>> import numpy as np
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
