from __future__ import absolute_import
import os
import tempfile
import numpy as np
from skimage import io
from microscopium import preprocess as pre
import pytest


@pytest.fixture
def image_files(request):
    # for clarity we define images as integer arrays in [0, 11) and
    # divide by 10 later
    i = np.array([[ 7,  4,  1,  1,  0],
                  [ 2,  5,  9,  6,  7],
                  [ 2,  3,  3,  8,  5],
                  [ 3,  0,  1,  7,  5],
                  [ 6,  0, 10,  1,  6]], np.uint8)
    j = np.array([[ 1, 10,  0,  9,  0],
                  [ 3, 10,  4,  1,  1],
                  [ 4, 10,  0,  7,  4],
                  [ 9,  3,  2,  0,  7],
                  [ 1,  3,  3,  9,  3]], np.uint8)
    k = np.array([[ 9,  1,  7,  7,  3],
                  [ 9,  1,  6,  2,  2],
                  [ 2,  8,  2,  0,  3],
                  [ 4,  3,  8,  9, 10],
                  [ 6,  0,  2,  3, 10]], np.uint8)
    files = []
    for im in [i, j, k]:
        f, fn = tempfile.mkstemp(suffix='.png')
        files.append(fn)
        io.imsave(fn, im)

    def cleanup():
        for fn in files:
            os.remove(fn)
    request.addfinalizer(cleanup)

    return files


def test_illumination_mean(image_files):
    illum = pre.find_background_illumination(image_files, radius=1,
                                             quantile=0.5,
                                             stretch_quantile=1e-7,
                                             method='mean')
    illum_true = np.array([[5.33, 5.33, 4.67, 1.67, 1.67],
                           [3.67, 6.67, 2.67, 4.33, 3.00],
                           [6.67, 3.00, 4.33, 3.00, 5.33],
                           [2.67, 2.67, 2.67, 6.67, 6.00],
                           [3.33, 2.00, 2.33, 6.33, 7.33]]) / 10
    np.testing.assert_array_almost_equal(illum, illum_true, decimal=1)


def test_illumination_median(image_files):
    illum = pre.find_background_illumination(image_files, radius=1,
                                             quantile=0.5,
                                             stretch_quantile=1e-7,
                                             method='median')
    illum_true = np.array([[ 4.,  5.,  4.,  1.,  1.],
                           [ 4.,  6.,  2.,  4.,  2.],
                           [ 8.,  3.,  4.,  2.,  7.],
                           [ 3.,  3.,  3.,  7.,  6.],
                           [ 3.,  3.,  3.,  7.,  7.]]) / 10
    np.testing.assert_array_almost_equal(illum, illum_true, decimal=1)


if __name__ == '__main__':
    np.testing.run_module_suite()
