import glob
import setuptools

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import build_ext

from setuptools import find_packages
import numpy as np
import os

with open("README.md", 'r') as readme:
    long_description = readme.read()

# to include data in the package, use MANIFEST.in

ext_modules = [
   Extension(name="wholecell.utils._build_sequences",
             sources=[os.path.join("wholecell", "utils", "_build_sequences.pyx")],
	     include_dirs = [np.get_include()],
             ),
   Extension(name="wholecell.utils.complexation",
             sources=[os.path.join("wholecell", "utils", "mc_complexation.pyx")],
	     include_dirs = [np.get_include()]
             ),
   Extension(name="wholecell.utils._fastsums",
             sources=[os.path.join("wholecell", "utils", "_fastsums.pyx")],
	     include_dirs = [np.get_include()]),
]

packages = find_packages(where=".")

setup(
    name='vivarium-ecoli',
    version='0.0.1',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    packages=find_packages(where="."),
    author='Eran Agmon, Ryan Spangler',
    author_email='eagmon@stanford.edu, ryan.spangler@gmail.com',
    url='https://github.com/CovertLab/vivarium-ecoli',
    license='MIT',
    entry_points={
        'console_scripts': []},
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'vivarium-core>=0.3.13',
        'vivarium-multibody>=0.0.16',
        'vivarium-convenience>=0.0.3',
        'biopython==1.77',
        'Unum==4.1.4',
        'numba==0.50.1',
        'ipython==7.16.1',
        'aesara==2.0.8',
        'decorator',
        'iteround',
        'stochastic-arrow',
        'cobra',
        'matplotlib',
        'pytest',
    ])
