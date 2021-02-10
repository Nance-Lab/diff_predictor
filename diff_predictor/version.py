from __future__ import absolute_import, division, print_function
from os.path import join as pjoin
from os.path import abspath, dirname

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
DESCRIPTION = "diff_predictor: a prediciton package for multiple particle tracking data"
# Long description will go up on the pypi page
LONG_DESCRIPTION = """
diff_predictor
========
Diff_predictor is a prediction package for multiple particle tracking data and
is intended for use alongside diff_classifier (https://github.com/Nance-Lab/diff_classifier).
It contains methods intendd to transform and redict MPT data, as well as methods for
analyzing prediction results including feature importancce using xgboost.
To get started using these components in your own software, please go to the
repository README_.
.. _README: https://github.com/uwescience/shablona/blob/master/README.md
License
=======
``diff_predictor`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
"""

NAME = "diff_predictor"
MAINTAINER = "David Shackelford"
MAINTAINER_EMAIL = "dash2927@uw.edu"
DESCRIPTION = DESCRIPTION
LONG_DESCRIPTION = LONG_DESCRIPTION
URL = "https://github.com/dash2927/diff_predicto"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "David Shackelford"
AUTHOR_EMAIL = "dash2927@uw.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'diff_predictor': [pjoin('data', '*')]}
PYTHON_REQUIRES = ">= 3.7.6"

src_dir = dirname(abspath(__file__))
requires_path = abspath(pjoin(src_dir, "requirements.txt"))
with open(requires_path) as f:
    REQUIRES = [line.strip('\n') for line in f.readlines()]
