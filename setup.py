"""Setup script for v2e."""

import setuptools
from setuptools import setup, find_packages

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3.7
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

    python_requires=">={}".format("3.7"),

    #  packages=find_packages(include=['v2ecore', 'v2e.*']),
    packages=find_packages(),
    url='https://github.com/SensorsINI/v2e',
    install_requires=[
        'numpy==1.20; python_version<"3.10"',
        'numpy>=1.24; python_version>="3.10"',
        'argcomplete',
        'engineering-notation', # not available on conda
        'tqdm',
        'opencv-python', # just opencv for conda
        'h5py', # ddd20 recordings
        'torch', # pytorch for conda
        'torchvision',
        'numba',
        #  'Gooey',
        'matplotlib', # used for some optional statistics plotting
        'plyer',
        'screeninfo', # to get monitor sizes for cv2 window placement
        'easygui', # eacy open a file from no arg invocation
        'scikit-image' # for some synthetic_input scripts
    ],

    scripts=['v2e.py', 'dataset_scripts/ddd/ddd_extract_data.py'],

    entry_points={
        'console_scripts': ['v2e=v2e:main']
    },

    classifiers=list(filter(None, classifiers.split('\n'))),
)
