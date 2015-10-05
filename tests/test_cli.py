import os

import sys
import pytest
import glob
import json

import numpy as np
import sh


def assert_close(current, expected):
    np.testing.assert_allclose(current, expected, atol=1e-3, rtol=1e-3)


def check_reference_feature_json(output, reference_file):
    """Compare JSON to a reference using knowledge about its contents.

    Parameters
    ----------
    output : iterable of string
        The output being tested. Each string must contain valid JSON.
    reference_file : iterable of string
        The reference against which the output is being compared.
    """
    for line, reference in zip(output, reference_file):
        if not line and reference == '\n':
            continue  # ignore blank lines
        d = json.loads(line)
        dref = json.loads(reference)
        if 'feature_vector' in d:
            assert_close(d['feature_vector'], dref['feature_vector'])
        elif 'pca_vector' in d:
            # 'pca_vector' and 'feature_vector_std' are emitted in the same
            # line of JSON, so we only check for one in `d` and assume the
            # other is there.
            assert_close(d['feature_vector_std'], dref['feature_vector_std'])
            assert_close(d['pca_vector'], dref['pca_vector'])
        elif 'neighbours' in d:
            assert set(d['neighbours']) == set(dref['neighbours'])


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
    images = sorted(glob.glob(os.path.join(env['images'], '*.tif')))
    mic = sh.Command(env['bin'])
    out = mic.features(*images, S=20, n=2, s='myores', b=8,
                       random_seed=0, _env=env['env'])
    ref = open(os.path.join(env['testdata'], 'emitted-features.json'))
    check_reference_feature_json(out.split('\n'), ref)


def test_features_single_threshold(env):
    images = sorted(glob.glob(os.path.join(env['images'], '*.tif')))
    mic = sh.Command(env['bin'])
    out = mic.features(*images, S=20, n=2, s='myores', b=8, G=True,
                       random_seed=0, _env=env['env'])
    ref = open(os.path.join(env['testdata'], 'emitted-features-global-t.json'))
    check_reference_feature_json(out.split('\n'), ref)
