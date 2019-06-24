"""Generate pre-computed distance matrix for Euclidean distance.
"""
import os
from numpy import sqrt
import pandas as pd
import numpy as np

abspath = os.path.dirname(__file__)

def string2tuple(string_tuple):
    # TODO add docstring
    string_values = string_tuple.split(', ')
    coords = (int(string_values[0][1:]), string_values[1][1:-2])
    return coords

def euclidean_distance(a, b):
    return sqrt(sum((a-b)**2 for a, b in zip(a, b)))

test_data = pd.read_csv(os.path.join(abspath, 'testdata/data_test.csv'),
                        index_col=0)

distance_matrix = np.zeros((8, 8))

for i in range(0, 8):
    for j in range(0, 8):
        dist = euclidean_distance(test_data.values[i], test_data.values[j])
        distance_matrix[i, j] = dist

distance_pd = pd.DataFrame(distance_matrix)

distance_pd.to_csv(os.path.join(abspath, 'testdata/distance_test.csv'), index=False, header=False)
