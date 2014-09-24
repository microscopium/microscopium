import numpy as np
from husc import preprocessing as pre
import pytest


@pytest.fixture
def ims(request):
    # for clarity we define images as integer arrays in [0, 11) and
    # divide by 10 later
    i = np.array([[ 7,  4,  1,  1,  0],
                  [ 2,  5,  9,  6,  7],
                  [ 2,  3,  3,  8,  5],
                  [ 3,  0,  1,  7,  5],
                  [ 6,  0, 10,  1,  6]])
    j = np.array([[ 1, 10,  0,  9,  0],
                  [ 3, 10,  4,  1,  1],
                  [ 4, 10,  0,  7,  4],
                  [ 9,  3,  2,  0,  7],
                  [ 1,  3,  3,  9,  3]])
    k = np.array([[ 9,  1,  7,  7,  3],
                  [ 9,  1,  6,  2,  2],
                  [ 2,  8,  2,  0,  3],
                  [ 4,  3,  8,  9, 10],
                  [ 6,  0,  2,  3, 10]])
    ims = [i, j, k]
    ims = [(x.astype(float) / 10) for x in ims]

def test_illumination_mean(ims):
    pass
