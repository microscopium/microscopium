#from distutils.core import setup
from setuptools import setup

descr = """husc: HCS unsupervised sample clustering."""

DISTNAME            = 'husc'
DESCRIPTION         = 'Clustering of High Content Screen Images'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Juan Nunez-Iglesias'
MAINTAINER_EMAIL    = 'juan.n@unimelb.edu.au'
URL                 = 'https://github.com/jni/husc'
LICENSE             = 'BSD 3-clause'
DOWNLOAD_URL        = 'https://github.com/jni/husc'
VERSION             = '0.1-dev'
PYTHON_VERSION      = (2, 7)
INST_DEPENDENCIES   = {}


if __name__ == '__main__':

    setup(name=DISTNAME,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license=LICENSE,
        packages=['husc', 'husc.screens'],
        install_requires=INST_DEPENDENCIES,
        scripts=["bin/husc"]
    )
