from __future__ import absolute_import
from . import myores, cellomics, image_xpress

__all__ = ['myores', 'cellomics', 'image_xpress']

d = {'myores': {'index': myores.filename2id,
                   'fmap':  myores.feature_map},
     'cellomics': {'index': cellomics.filename2id,
                   'fmap':  cellomics.feature_map},
     'image-xpress': {'index': image_xpress.filename2id,
                      'fmap': image_xpress.feature_map}
    }

