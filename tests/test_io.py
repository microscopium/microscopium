import numpy as np
from tempfile import NamedTemporaryFile
from microscopium import io as mio

def test_imsave_tif_compress():
    im = np.random.randint(0, 256, size=(1024, 1024, 3))
    with NamedTemporaryFile(suffix='.tif') as fout:
        fname = fout.name
        fout.close()
        mio.imsave(im, fname, compress=2)
        imin = mio.imread(fname)
        np.testing.assert_array_equal(im, imin)
