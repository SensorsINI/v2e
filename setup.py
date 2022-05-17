"""Setup script for v2e."""

import setuptools
from setuptools import setup, find_packages

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

version="1.5.1"
package_name="v2e"

setup(
    name=package_name,
    version=version,
    description='Generates synthetic DVS events from conventional video',

    author="Tobi Delbruck, Yuhuang Hu, Zhe He",
    author_email="yuhuang.hu@ini.uzh.ch, tobi@ini.uzh.ch",

    python_requires=">={}".format("3.8"),

    #  packages=find_packages(include=['v2ecore', 'v2e.*']),
    packages=find_packages(),
    url='https://github.com/SensorsINI/v2e',
    scripts=["v2e.py"],
    install_requires=[
        'numpy==1.20',
        'argcomplete',
        'engineering-notation', # not available on conda
        'tqdm',
        'opencv-python', # just opencv for conda
        'h5py',
        'torch', # pytorch for conda
        'torchvision',
        'numba',
        #  'Gooey',
        'matplotlib',
        'plyer',
        'screeninfo' # to get monitor sizes for cv2 window placement
    ],

    entry_points={
        'console_scripts': ['v2e=v2e:main']
    },

    classifiers=list(filter(None, classifiers.split('\n'))),
)
