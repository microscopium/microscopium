from __future__ import absolute_import
#from distutils.core import setup
from setuptools import setup

descr = """
microscopium: unsupervised sample clustering and dataset exploration
for high content screens.
"""

DISTNAME            = 'microscopium'
DESCRIPTION         = 'Clustering of High Content Screen Images'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Juan Nunez-Iglesias'
MAINTAINER_EMAIL    = 'jni@fastmail.com'
URL                 = 'https://github.com/microscopium/microscopium'
LICENSE             = 'BSD 3-clause'
DOWNLOAD_URL        = 'https://github.com/microscopium/microscopium'
VERSION             = '0.1-dev'
PYTHON_VERSION      = (3, 6)
INST_DEPENDENCIES   = []


if __name__ == '__main__':

    setup(name=DISTNAME,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license=LICENSE,
        packages=['microscopium', 'microscopium.screens'],
        install_requires=INST_DEPENDENCIES,
        scripts=["bin/mic"]
    )
