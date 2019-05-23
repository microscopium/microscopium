import numpy as np
import pytest
from skimage.util import img_as_int, img_as_float

from microscopium import features


@pytest.fixture(scope="module", params=[img_as_int, img_as_float])
def haralick_image(request):
    haralick_image = np.array([[0, 0, 1, 1],
                               [0, 0, 1, 1],
                               [0, 2, 2, 2],
                               [2, 2, 3, 3]])
    haralick_image = request.param(haralick_image)
    return haralick_image


def test_haralick_features_8bit(haralick_image):
    fs, names = features.haralick_features(haralick_image,
                                           distances=[5],
                                           angles=[0])
    expected_names = [
        'haralick-contrast-distance5-angle0',
        'haralick-dissimilarity-distance5-angle0',
        'haralick-homogeneity-distance5-angle0',
        'haralick-ASM-distance5-angle0',
        'haralick-energy-distance5-angle0',
        'haralick-correlation-distance5-angle0']
    expected_features = np.array([0., 0., 0., 0., 0., 1.])
    assert np.allclose(fs, expected_features)
    assert names == expected_names
