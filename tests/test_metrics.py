from husc import metrics
import numpy as np
import os
import pandas as pd
from pymongo import MongoClient
import subprocess as sp

abspath = os.path.dirname(__file__)

def string2tuple(string_tuple):
    # TODO add docstring
    string_values = string_tuple.split(', ')
    coords = (int(string_values[0][1:]), string_values[1][1:-2])
    return coords

client = MongoClient('localhost', 27017)
db = client['myofusion_test']
collection = db.wells_test

if db.wells_test.find({}).count() == 0:
    sp.Popen(['mongoimport', '-host', 'localhost:27017', '-d',
              'myofusion_test', '-c', 'wells_test',
              os.path.join(abspath, 'testdata/wells_test.json')])
    time.sleep(2)

test_data = pd.read_csv(os.path.join(abspath, 'testdata/data_test.csv'),
                        index_col=0, converters={0: string2tuple})

def test_mongo_group_by():
    expected = set({'Mbnl1': [(2490700, 'L13'), (2490702, 'L13')],
                'Nudt3': [(2490702, 'L04'), (2490701, 'L04')],
                'Lmbr1l': [(2490702, 'G03'), (2490701, 'G03')],
                'Pknox1': [(2490702, 'H05'), (2490700, 'H05')]})
    query = set(metrics.mongo_group_by(collection, 'gene_name'))
    assert expected == query

def test_gene_distance_score():
    expected_intra = []
    for i in range(0, 4):
        gene_pair = test_data.ix[2*i:2*i+2].values
        expected_intra.append(np.linalg.norm(gene_pair[0] - gene_pair[1]))
    intra, inter = metrics.gene_distance_score(test_data, collection)
    np.testing.assert_array_almost_equal(expected_intra, intra, decimal=4)

def test_gene_distance_score2():
    intra, inter = metrics.gene_distance_score(test_data, collection)
    assert np.mean(intra) < np.mean(inter)
