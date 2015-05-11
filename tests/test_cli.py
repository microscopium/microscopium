import os
import pytest
import glob

import sh

@pytest.fixture
def env():
    """Return dictionary with useful directories to run tests

    Returns
    -------
    dirs : dict
        A dictionary with directories pointing to 'bin' (where to find
        the mic "binary"), 'testdata' (the location of test data other
        than images), and 'images' (the location of test images).
        Additionally, the dictionary contains 'env', an environment to
        run ``sh``
    """
    dirs = {}
    curdir = os.path.dirname(__file__)
    dirs['bin'] = os.path.abspath(os.path.join(curdir, '..', 'bin'))
    env_copy = os.environ.copy()
    env_copy['PATH'] = ':'.join(dirs['bin'], os.environ['PATH'])
    dirs['env'] = env_copy
    dirs['testdata'] = os.path.join(curdir, 'testdata')
    dirs['images'] = os.path.join(dirs['testdata'], 'images')
    return dirs

def test_features(env):
    images = glob.glob(os.path.join(env['images'], '*.tif'))
    sh.mic.features(S=20, n=2, s='myores', b=8, _env=env['env'])
