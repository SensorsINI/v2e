"""Setup script for v2e."""

from setuptools import setup

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

setup(
    name='v2e',
    version="1.4.1",

    author="Tobi Delbruck, Yuhuang Hu, Zhe He",
    author_email="tobi@ini.uzh.ch",

    packages=["v2ecore"],

    scripts=["v2e.py"],

    classifiers=list(filter(None, classifiers.split('\n'))),
)
