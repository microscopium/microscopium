# !/user/bin/python

import numpy as np
import math
from skimage.transform import downscale_local_mean
from sklearn.utils.extmath import cartesian as skcartesian

eps = np.finfo(float).eps

"""Written by: Don Teng, over 2016. 
Library containing functions directly related to extracting patches 
from an image. Import convention: 'import patch_extraction as pex'

Patch extraction is cut into 2 distinct steps:
1. Generate the top-left coordinates of patches to be extracted. 
These can be set to allow the patches to overlap, or otherwise.
2. Extract patches from an image, based on those top-left coordinates.
"""


def count_grids(patch_len, im_nrows, im_ncols):
    """Returns the number of grids available from given dimensions.

    Params
    ------
    patch_len: int; desired patch length. Patches must be square.
    im_nrows: int; no. of rows in the image, i.e. image height
    im_ncols: int; no. of cols in the image, i.e. image width

    Returns
    -------
    n_grids: int. No. of grids
    """
    n_grids = math.floor(im_ncols/patch_len)*math.floor(im_nrows/patch_len)
    return n_grids


def generate_patch_coords(n_patches, 
                          patch_len,
                          im_ncols, 
                          im_nrows, 
                          verbose=True,
                          method='non-overlapping'):
    """Generates a set of top-left patch coordinates.

    Params
    ------
    n_patches: int; desired number of patches
    patch_len: int; desired side length of each patch.
    im_ncols: int; no. of cols of some image, i.e. width
    im_nrows: int; no. of rows of some image, i.e. height
    method: 'random' or 'non-overlapping'.
        'random': Generates patches entirely at random, allowing them 
        to potentially overlap.
        'Non-overlapping': Cuts an image into a grid, then randomly 
        selects n_patches grids by their top-left coordinates.

    Returns
    -------
    coords: array of shape(n_patches, 2)
        each entry in coords := top-left coordinates of a single patch
    """
    if method=='random':
        coords = np.zeros((n_patches, 2))
        for row in coords:
            row[0] = np.random.randint(0,im_ncols - patch_len)
            row[1] = np.random.randint(0,im_nrows - patch_len)
        chosen = coords

    elif method == 'non-overlapping':
        n_grids = math.floor(im_ncols/patch_len)*math.floor(im_nrows/patch_len)
        if n_patches > n_grids:
            print("Too many patches requested, not enough grids")
            raise ValueError

        rlist = np.arange(0,im_nrows-patch_len,patch_len)
        clist = np.arange(0,im_ncols-patch_len,patch_len)
        coords = skcartesian(np.array((rlist, clist)))

        idx = np.random.choice(len(coords),n_patches, replace=False)
        chosen = coords[idx]

    return chosen


def extract_patches_3d(image, coords, patch_len, ds_factor):
    """Extracts 3d patches from an image, given an array of top-left coordinates
    Each set of coords is used on all colour channels, to extract a "block".

    Params
    ------
    image: array, shape (im_width, im_height, n_channels).
    Coords: array of coordinates, shape (n_patches, 2)
    patch_len: int.

    Returns
    -------
    patches: array, shape(n_patches, patch_len, patch_len, ch)
    """

    n_patches = coords.shape[0]
    ch = image.shape[2] # no. of colour channels

    patches = np.zeros((n_patches, patch_len, patch_len, ch))
    for i in range(n_patches):
        x, y = coords[i]
        patches[i] = image[x:x+patch_len, y:y+patch_len,:]

    patches_s = []
    for i in range(n_patches):
        patch = patches[i]
        patch_s = downscale_local_mean(patch, ds_factor)
        patches_s.append(patch_s)

    return np.array(patches_s)


def generate_specific_pseudo_image(image, coords, patch_len):
    """Extracts square patches from an image, given a set of coordinates

    Params
    ------
    image: array of float.
    coords: array of coordinates, shape (n_patches, 2).
    patch_len: int.

    Returns
    -------
    pseudo_img: array, shape (n_patches, patch_sidelen, patch_sidelen)
    """
    n_patches = coords.shape[0]

    if len(img.shape) > 2:
        pseudo_image = np.zeros((n_patches, patch_len, patch_len,img.shape[2]))
        for i in range(n_patches):
            Ux = coords[i][0]
            Uy = coords[i][1]
            pseudo_image[i] = image[Ux:Ux+patch_len, Uy:Uy+patch_len, :]

    elif len(img.shape) == 2:
        pseudo_image = np.zeros((n_patches, patch_len, patch_len))
        for i in range(n_patches):
            Ux = coords[i][0]
            Uy = coords[i][1]
            pseudo_image[i] = image[Ux:Ux+patch_len, Uy:Uy+patch_len]

    return pseudo_image


def generate_rand_pseudo_image(image, n_patches, verbose=False):
    """
    Extracts n_patches patches from an image
    Each pseudo_image = array, shape (n_patches, patch_len**2)
    Each patch is ravelled, but the pseudo_image isn't (yet).

    The patch will be square, shape patch_len**2
    But the image need not be square
    window_len must be < img_width and img_height, obviously.

    Parameters
    ----------
    image: array, shape (img_width, img_height, ch)
    patch_len: int; desired side length of each patch.
    n_patches: int; No. of patches to be extracted per image.
    verbose: Boolean, verbosity

    Returns
    -------
    pseudo_images: array, shape(n_images, n_patches, patch_size)
        Where patch_size = patch_len**2
    """

    n_images, img_height, img_width = image.shape
    patch_size = patch_len*patch_len
    pseudo_images = np.zeros((n_images, n_patches, patch_size))

    #Generate 2 uniform r.v.s,
    #which will be coordinates of the top-leftmost corner of the patch
    for i in range(n_images):
        #Generate n_patches from each image
        patches = np.zeros((n_patches, patch_size))
        for j in range(n_patches):
            Ux = np.random.randint(0,img_width - patch_len)
            Uy = np.random.randint(0,img_height - patch_len)
            patch = images[i][Uy:Uy+patch_len, Ux:Ux+patch_len]
            patch = patch.ravel(order = 'C')
            patches[j] = patch
        pseudo_images[i] = patches

    return pseudo_images


def linearize_pseudo_images(pseudo_images):
    """Ravels all patches in an image into a single vector.
    Coded as a separate step just to keep things modular;
    Assimilate into generate_pseudo_images if it might as well be a
    private function.

    Parameters
    ----------
    pseudo_images: array, shape(n_images, n_patches, patch_size)

    Returns
    -------
    pseudo_images_linarized: array, shape(n_images, (n_patches*patch_size))
    """
    n_images, n_patches, patch_size = pseudo_images.shape

    pseudo_images_linearized = np.zeros((n_images, (n_patches*patch_size)))
    for i in range(n_images):
        p_image = pseudo_images[i].ravel(order='C')
        pseudo_images_linearized[i] = p_image

    return pseudo_images_linearized
