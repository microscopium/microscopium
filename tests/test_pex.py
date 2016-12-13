# !/user/bin/python

import numpy as np
import math
from skimage.transform import downscale_local_mean
from sklearn.utils.extmath import cartesian as skcartesian
import skynet.patch_extraction as pex

"""
Written by Don Teng. Last update: Dec 2016
Unit tests for patch_extraction.py libary
"""

def test_count_grids():
    """Returns the number of grids available from given 2D image dimensions.
    Note: if patchlen > img height or width, this currently returns 0, without
    a break. Maybe it should break during runtime to indicate an illogical error.
    """
    # a list of test inputs. Each record is in the format
    # [patch_len, im_nrows, im_ncols]
    test_inputs = [[1, 10, 10], [2, 10, 10],
    [3, 10, 10], [11, 10, 10]]
    expected = [100, 25, 9, 0]
    assert pex.count_grids(test_inputs[0][0],
    test_inputs[0][1],
    test_inputs[0][2]) == expected[0]
    assert pex.count_grids(test_inputs[1][0],
    test_inputs[1][1],
    test_inputs[1][2]) == expected[1]
    assert pex.count_grids(test_inputs[2][0],
    test_inputs[2][1],
    test_inputs[2][2]) == expected[2]
    assert pex.count_grids(test_inputs[3][0],
    test_inputs[3][1],
    test_inputs[3][2]) == expected[3]


def test_generate_patch_coords(n_patches, patch_len,
im_ncols, im_nrows, verbose=True,
method='non-overlapping'):
    """Original: generate a set of top-left patch coordinates.
    Should get coords: array of shape(n_patches, 2), where each entry in
    coords := top-left coordinates of a single patch.
    """


def test_extract_patches_3d(image, coords, patch_len, ds_factor):
    """Extracts 3d patches from an image, given an array of top-left coordinates
    Returns: patches: array, shape(n_patches, patch_len, patch_len, ch)
    """
    im_len = 20
    patch_len = 4
    coords = np.array([[9, 15], [12, 6], [0, 3]])
    n_patches = len(coords)

    test_img = np.arange(im_len*im_len*3).reshape(im_len, im_len, 3)
    ch = image.shape[2] # no. of colour channels
    expected = np.array([[[[ 585.,  586.,  587.],
         [ 588.,  589.,  590.],
         [ 591.,  592.,  593.],
         [ 594.,  595.,  596.]],

        [[ 645.,  646.,  647.],
         [ 648.,  649.,  650.],
         [ 651.,  652.,  653.],
         [ 654.,  655.,  656.]],

        [[ 705.,  706.,  707.],
         [ 708.,  709.,  710.],
         [ 711.,  712.,  713.],
         [ 714.,  715.,  716.]],

        [[ 765.,  766.,  767.],
         [ 768.,  769.,  770.],
         [ 771.,  772.,  773.],
         [ 774.,  775.,  776.]]],


       [[[ 738.,  739.,  740.],
         [ 741.,  742.,  743.],
         [ 744.,  745.,  746.],
         [ 747.,  748.,  749.]],

        [[ 798.,  799.,  800.],
         [ 801.,  802.,  803.],
         [ 804.,  805.,  806.],
         [ 807.,  808.,  809.]],

        [[ 858.,  859.,  860.],
         [ 861.,  862.,  863.],
         [ 864.,  865.,  866.],
         [ 867.,  868.,  869.]],

        [[ 918.,  919.,  920.],
         [ 921.,  922.,  923.],
         [ 924.,  925.,  926.],
         [ 927.,  928.,  929.]]]])

    assert pex.extract_patches_3d(test_img, coords, patch_len, (1,1,1)) == expected

# ad-hoc functions not part of the usual workflow are not tested.
# def generate_specific_pseudo_image(image, coords, patch_len):
# def generate_rand_pseudo_image(image, n_patches, verbose=False):
# def linearize_pseudo_images(pseudo_images):
