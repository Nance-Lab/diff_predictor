import sys
import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = os.path.join('diff_predictor', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
# SETUP_REQUIRES = ['setuptools >= 24.2.0']
# This enables setuptools to install wheel on-the-fly
# SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []
PACKAGES = find_packages()

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            python_requires=PYTHON_REQUIRES,
            #setup_requires=SETUP_REQUIRES,
            install_requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)
