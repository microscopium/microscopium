import os
import numpy as np
from skimage import io
from collections import OrderedDict
from microscopium.screens import cellomics
import pytest
import tempfile


@pytest.fixture
def image_files_25(request):
    files = []
    for i in range(0, 25):
        image = np.ones((2, 2), np.uint8) * i
        file_prefix = '%02d' % (i,) + '_'  # filenames must be sortable
        f, fn = tempfile.mkstemp(prefix=file_prefix, suffix='.png')
        files.append(fn)
        io.imsave(fn, image)

    def cleanup():
        for fn in files:
            os.remove(fn)
    request.addfinalizer(cleanup)

    return files

@pytest.fixture
def image_files_6(request):
    files = []
    for i in range(0, 6):
        image = np.ones((2, 2), np.uint8) * i
        file_prefix = '%02d' % (i,) + '_'  # filenames must be sortable
        f, fn = tempfile.mkstemp(prefix=file_prefix, suffix='.png')
        files.append(fn)
        io.imsave(fn, image)

    def cleanup():
        for fn in files:
            os.remove(fn)
    request.addfinalizer(cleanup)

    return files


def test_cellomics_semantic_filename():
    test_fn = 'MFGTMP_140206180002_A01f00d0.TIF'
    expected = OrderedDict([('directory', ''),
                            ('prefix', 'MFGTMP'),
                            ('plate', 140206180002),
                            ('well', 'A01'),
                            ('field', 0),
                            ('channel', 0),
                            ('suffix', 'TIF')])
    assert cellomics.cellomics_semantic_filename(test_fn) == expected


def test_cellomics_semantic_filename2():
    test_fn = 'AS_09125_050116110001_A04f01d0.tif'
    expected = OrderedDict([('directory', ''),
                            ('prefix', 'AS_09125'),
                            ('plate', 50116110001),
                            ('well', 'A04'),
                            ('field', 1),
                            ('channel', 0),
                            ('suffix', 'tif')])
    assert cellomics.cellomics_semantic_filename(test_fn) == expected


def test_cellomics_semantic_filename3():
    test_fn = '/file_directory/AS_09125_050116110001_A03f04d1_stitch.tif'
    expected = OrderedDict([('directory', '/file_directory'),
                            ('prefix', 'AS_09125'),
                            ('plate', 50116110001),
                            ('well', 'A03'),
                            ('field', 4),
                            ('channel', 1),
                            ('suffix', 'tif')])
    assert cellomics.cellomics_semantic_filename(test_fn) == expected


def test_make_key2file():
    test_fns = ['AS_09125_050116110001_A03f00d0.tif',
                'AS_09125_050116110001_A03f01d0.tif',
                'AS_09125_050116110001_A03f02d0.tif',
                'AS_09125_050116110001_A04f00d0.tif',
                'AS_09125_050116110001_A04f01d0.tif',
                'AS_09125_050116110001_A04f02d0.tif']
    expected = {(50116110001, 'A03'): ['AS_09125_050116110001_A03f00d0.tif',
                                       'AS_09125_050116110001_A03f01d0.tif',
                                       'AS_09125_050116110001_A03f02d0.tif'],
                (50116110001, 'A04'): ['AS_09125_050116110001_A04f00d0.tif',
                                       'AS_09125_050116110001_A04f01d0.tif',
                                       'AS_09125_050116110001_A04f02d0.tif']}
    assert cellomics.make_key2file(test_fns) == expected

# test 5x5 'clockwise-right' sitching
def test_snail_stitch(image_files_25):

    order = [[20, 21, 22, 23, 24],
             [19, 6, 7, 8, 9],
             [18, 5, 0, 1, 10],
             [17, 4, 3, 2, 11],
             [16, 15, 14, 13, 12]]

    stitched = cellomics.snail_stitch(image_files_25, order)
    expected = np.array([[20, 20, 21, 21, 22, 22, 23, 23, 24, 24],
                         [20, 20, 21, 21, 22, 22, 23, 23, 24, 24],
                         [19, 19, 6, 6, 7, 7, 8, 8, 9, 9],
                         [19, 19, 6, 6, 7, 7, 8, 8, 9, 9],
                         [18, 18, 5, 5, 0, 0, 1, 1, 10, 10],
                         [18, 18, 5, 5, 0, 0, 1, 1, 10, 10],
                         [17, 17, 4, 4, 3, 3, 2, 2, 11, 11],
                         [17, 17, 4, 4, 3, 3, 2, 2, 11, 11],
                         [16, 16, 15, 15, 14, 14, 13, 13, 12, 12],
                         [16, 16, 15, 15, 14, 14, 13, 13, 12, 12]])

    np.testing.assert_array_equal(stitched, expected)

# test 3x2 'half-clockwise-left' stitching
def test_snail_stitch2(image_files_6):

    order = [[2, 3, 4],
             [1, 0, 5]]

    stitched = cellomics.snail_stitch(image_files_6, order)
    expected = np.array([[2, 2, 3, 3, 4, 4],
                         [2, 2, 3, 3, 4, 4],
                         [1, 1, 0, 0, 5, 5],
                         [1, 1, 0, 0, 5, 5]])

    np.testing.assert_array_equal(stitched, expected)