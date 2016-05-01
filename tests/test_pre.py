import os
import tempfile
import numpy as np
from microscopium import preprocess as pre
from microscopium import io as mio
import pytest
import warnings


@pytest.fixture
def image_files(request):
    # for clarity we define images as integer arrays in [0, 11) and
    # divide by 10 later
    i = np.array([[7, 4, 1, 1, 0],
                  [2, 5, 9, 6, 7],
                  [2, 3, 3, 8, 5],
                  [3, 0, 1, 7, 5],
                  [6, 0, 10, 1, 6]], np.uint8)
    j = np.array([[1, 10, 0, 9, 0],
                  [3, 10, 4, 1, 1],
                  [4, 10, 0, 7, 4],
                  [9, 3, 2, 0, 7],
                  [1, 3, 3, 9, 3]], np.uint8)
    k = np.array([[9, 1, 7, 7, 3],
                  [9, 1, 6, 2, 2],
                  [2, 8, 2, 0, 3],
                  [4, 3, 8, 9, 10],
                  [6, 0, 2, 3, 10]], np.uint8)
    files = []
    for im in [i, j, k]:
        f, fn = tempfile.mkstemp(suffix='.png')
        files.append(fn)
        mio.imsave(fn, im)

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
    illum_true = np.array([[5., 5.33, 5.67, 3., 3.33],
                           [5.33, 5.33, 3., 5., 3.],
                           [5.33, 2.67, 5., 4., 5.],
                           [2.67, 3.67, 3.67, 5., 5.33],
                           [4.33, 2., 3.67, 5.33, 6.33]]) / 10
    np.testing.assert_array_almost_equal(illum, illum_true, decimal=1)


def test_illumination_median(image_files):
    illum = pre.find_background_illumination(image_files, radius=1,
                                             quantile=0.5,
                                             stretch_quantile=1e-7,
                                             method='median')
    illum_true = np.array([[4., 5., 6., 2., 2.],
                           [5., 5., 2., 6., 2.],
                           [4., 3., 6., 4., 7.],
                           [3., 3., 3., 7., 7.],
                           [4., 3., 2., 6., 6.]]) / 10
    np.testing.assert_array_almost_equal(illum, illum_true, decimal=1)


def conv(im):
    return np.round(np.clip(im, 0, np.inf) * 255).astype(np.uint8)


@pytest.fixture
def image_files_noise(request):
    """Three sham images; one has no signal, one has an intensity artifact."""
    r = np.random.RandomState(0)
    shape = (5, 5)
    # no signal
    i = conv(0.01 * np.ones(shape, dtype=float) + 0.005 * r.randn(*shape))
    # normal image
    j = conv(0.5 * r.rand(*shape))
    # blown-out corner
    k = 0.5 * r.rand(*shape)
    k[3:, 3:] = 1.0
    k = conv(k)
    files = []
    for im in [i, j, k]:
        f, fn = tempfile.mkstemp(suffix='.png')
        files.append(fn)
        mio.imsave(fn, im)

    def cleanup():
        for fn in files:
            os.remove(fn)
    request.addfinalizer(cleanup)

    illum = 0.01 * np.ones(shape, dtype=float)
    return files, illum


def test_correct_multiimage_illum(image_files_noise):
    files, illum = image_files_noise
    with mio.temporary_file('.tif') as out_fn:
        ims = pre.correct_multiimage_illumination(files, illum, (2 / 25), 0)
        i, j, k = list(ims)
        # 1. check noise is not blown out in i
        assert not np.any(i > 10)
        # 2. check blown out corner in k has not suppressed all other values
        assert np.median(k) > 100


cellomics_pattern = "MFGTMP_150406100001_A01f{0:02d}d0.TIF"

missing_test_fns = [
    ([cellomics_pattern.format(i) for i in range(25)], []),
    ([cellomics_pattern.format(i) for i in range(25)], [1, 13])
]

# delete "images" with fields 1 and 13 from second set of
# image filesnames
missing_test_fns[1][0].remove(cellomics_pattern.format(1))
missing_test_fns[1][0].remove(cellomics_pattern.format(13))


@pytest.mark.parametrize("fns, expected", missing_test_fns)
def test_find_missing_fields(fns, expected):
    actual = pre.find_missing_fields(fns)
    np.testing.assert_array_equal(actual, expected)


# create a list of parameters for testing the create missing mask files
# each entry in the tuple represents the fields: missing, order, rows, cols
# and expected (the expected output from the function)
missing_mask_test = [
    ([], [[0, 1, 2]], 10, 5, np.ones((10, 15), dtype=np.bool)),
    ([0, 5], [[0, 1, 2], [4, 5, 6]], 5, 10, np.ones((10, 30), dtype=np.bool)),
    ([3, 4], [[0, 1], [2, 3], [4, 5]], 10, 5, np.ones((30, 10), dtype=np.bool))
]

# insert False to missing areas of expected output
missing_mask_test[1][4][0:5, 0:10] = False
missing_mask_test[1][4][5:10, 10:20] = False

missing_mask_test[2][4][10:20, 5:10] = False
missing_mask_test[2][4][20:30, 0:5] = False


# pass the set of list parameters to the test_create_missing_mask
# function. the test wil run against every of parameters in the
# missing_mask_test list
@pytest.mark.parametrize("missing, order, rows, cols, expected",
                         missing_mask_test)
def test_create_missing_mask(missing, order, rows, cols, expected):
    actual = pre.create_missing_mask(missing, order, rows, cols)
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def test_image_files_montage(request):
    def make_test_montage_files(missing_fields):
        shape = (2, 2)

        fields = list(range(0, 25))
        for missing_field in missing_fields:
            fields.remove(missing_field)

        ims = [np.ones(shape, np.uint8) * i for i in fields]
        files = []

        for field, im in zip(fields, ims):
            prefix = "MFGTMP_140206180002_A01f{0:02d}d0".format(field)
            f, fn = tempfile.mkstemp(prefix=prefix, suffix=".tif")
            files.append(fn)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mio.imsave(fn, im)

        def cleanup():
            for file in files:
                os.remove(file)
        request.addfinalizer(cleanup)

        return files
    return make_test_montage_files


def test_montage_with_missing(test_image_files_montage):
    files = test_image_files_montage(missing_fields=[20])
    montage, mask, number_missing = pre.montage_with_missing(files)

    expect_montage = np.array([[0, 0, 21, 21, 22, 22, 23, 23, 24, 24],
                               [0, 0, 21, 21, 22, 22, 23, 23, 24, 24],
                               [19, 19, 6, 6, 7, 7, 8, 8, 9, 9],
                               [19, 19, 6, 6, 7, 7, 8, 8, 9, 9],
                               [18, 18, 5, 5, 0, 0, 1, 1, 10, 10],
                               [18, 18, 5, 5, 0, 0, 1, 1, 10, 10],
                               [17, 17, 4, 4, 3, 3, 2, 2, 11, 11],
                               [17, 17, 4, 4, 3, 3, 2, 2, 11, 11],
                               [16, 16, 15, 15, 14, 14, 13, 13, 12, 12],
                               [16, 16, 15, 15, 14, 14, 13, 13, 12, 12]],
                              np.uint8)

    np.testing.assert_array_equal(expect_montage, montage)


def test_montage_with_missing_mask(test_image_files_montage):
    files = test_image_files_montage(missing_fields=[3, 8])
    montage, mask, number_missing = pre.montage_with_missing(files)

    expected_mask = np.ones((10, 10), np.bool)
    expected_mask[6:8, 4:6] = False
    expected_mask[2:4, 6:8] = False

    np.testing.assert_array_equal(expected_mask, mask)


def test_montage_with_missing_number_missing(test_image_files_montage):
    files = test_image_files_montage(missing_fields=[10, 11, 12])
    montage, mask, number_missing = pre.montage_with_missing(files)
    assert number_missing == 3

if __name__ == '__main__':
    pytest.main()
