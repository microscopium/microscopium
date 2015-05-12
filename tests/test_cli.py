import os
import sys
import pytest
import glob
import json

import numpy as np
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
    dirs['root'] = os.path.abspath(os.path.join(curdir, '..'))
    dirs['bindir'] = os.path.abspath(os.path.join(dirs['root'], 'bin'))
    dirs['bin'] = os.path.join(dirs['bindir'], 'mic')
    env_copy = os.environ.copy()
    env_copy['PATH'] = ':'.join([dirs['bin'], os.environ['PATH']])
    env_copy['PYTHONPATH'] = ':'.join([dirs['root']] + sys.path)
    dirs['env'] = env_copy
    dirs['testdata'] = os.path.join(curdir, 'testdata')
    dirs['images'] = os.path.join(dirs['testdata'], 'images')
    return dirs


def test_features(env):
    def assert_close(current, expected):
        np.testing.assert_allclose(current, expected, atol=1e-3, rtol=1e-3)
    images = glob.glob(os.path.join(env['images'], '*.tif'))
    mic = sh.Command(env['bin'])
    out = mic.features(*images, S=20, n=2, s='myofusion', b=8,
                       random_seed=0, _env=env['env'])
    fin = open(os.path.join(env['testdata'], 'emitted-features.json'))
    for line, reference in zip(out.split('\n'), fin):
        if not line and reference == '\n':
            continue  # ignore blank lines
        d = json.loads(line)
        dref = json.loads(reference)
        if 'feature_vector' in d:
            assert_close(d['feature_vector'], dref['feature_vector'])
        elif 'pca_vector' in d:
            assert_close(d['feature_vector_std'], dref['feature_vector_std'])
            assert_close(d['pca_vector'], dref['pca_vector'])
        elif 'neighbours' in d:
            assert set(d['neighbours']) == set(dref['neighbours'])
