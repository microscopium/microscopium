from __future__ import absolute_import
from . import myores, cellomics

__all__ = ['myores', 'cellomics']

d = {'myores': {'index': myores.filename2id,
                   'fmap':  myores.feature_map},
     'cellomics': {'index': cellomics.filename2id,
                   'fmap':  cellomics.feature_map}
    }

