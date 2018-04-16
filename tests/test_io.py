import os
import numpy as np
from tempfile import NamedTemporaryFile
from microscopium import io as mio
from microscopium import pathutils as pth


def test_recursive_glob():
    abspath = os.path.dirname(__file__)
    tiffs0 = pth.all_matching_files(abspath, '*.tif')
    assert len(tiffs0) == 8
    assert tiffs0[0].startswith('/')
    tiffs1 = pth.all_matching_files(abspath, '*.TIF')
    assert len(tiffs1) == 0
    tiffs2 = pth.all_matching_files(abspath, '*.TIF', case_sensitive=False,
                                    full=False)
    assert len(tiffs2) == 8
    assert tiffs2[0].startswith('MYORES')


def test_imsave_tif_compress():
    im = np.random.randint(0, 256, size=(1024, 1024, 3)).astype(np.uint8)
    with NamedTemporaryFile(suffix='.tif') as fout:
        fname = fout.name
        fout.close()
        mio.imsave(im, fname, compress=2)
        imin = mio.imread(fname)
        np.testing.assert_array_equal(im, imin)
