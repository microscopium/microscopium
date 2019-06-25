import json
from microscopium import metrics
import numpy as np
import os
import pandas as pd

abspath = os.path.dirname(__file__)

def string2tuple(string_tuple):
    # TODO add docstring
    string_values = string_tuple.split('-')
    coords = (int(string_values[0][1:]), string_values[1][1:-2])
    return coords

with open(os.path.join(abspath, 'testdata/wells_test.json')) as fin:
    collection = json.load(fin)

test_data = pd.read_csv(os.path.join(abspath, 'testdata/data_test.csv'),
                        index_col=0)

test_distance = pd.read_csv(os.path.join(abspath, 'testdata/distance_test.csv'),
                            header=None)

def test_gene_distance_score():
    expected_intra = []
    for i in range(0, 4):
        expected_intra.append(test_distance[2*i][2*i+1])
    intra, inter = metrics.gene_distance_score(test_data, collection)
    np.testing.assert_array_almost_equal(expected_intra, intra, decimal=4)

def test_gene_distance_score2():
    intra, inter = metrics.gene_distance_score(test_data, collection)
    assert np.mean(intra) < np.mean(inter)
