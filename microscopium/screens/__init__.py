from __future__ import absolute_import
from . import myofusion

__all__ = ['myofusion']

d = {'myofusion': {'index': myofusion.filename2coord,
                   'fmap':  myofusion.feature_map}}

