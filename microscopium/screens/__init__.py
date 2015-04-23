from __future__ import absolute_import
from . import myofusion, cellomics

__all__ = ['myofusion', 'cellomics']

d = {'myofusion': {'index': myofusion.filename2id,
                   'fmap':  myofusion.feature_map},
     'cellomics': {'index': cellomics.filename2id,
                   'fmap':  cellomics.feature_map}
    }

