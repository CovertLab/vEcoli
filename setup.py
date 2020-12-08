from __future__ import absolute_import, division, print_function

import glob
import setuptools

from distutils.core import setup
from Cython.Build import cythonize

import numpy as np
import os

with open("README.md", 'r') as readme:
    long_description = readme.read()

# to include data in the package, use MANIFEST.in

build_sequences_module = cythonize(
	os.path.join("wholecell", "utils", "_build_sequences.pyx"),
	# annotate=True,
	)

setup(
	name = "Build sequences",
	ext_modules = build_sequences_module,
	include_dirs = [np.get_include()]
	)

complexation_module = cythonize(
	os.path.join("wholecell", "utils", "mc_complexation.pyx"),
	# annotate=True,
	)

setup(
	name = "Monte-carlo complexation",
	ext_modules = complexation_module,
	include_dirs = [np.get_include()]
	)

fast_polymerize_sums_module = cythonize(
	os.path.join("wholecell", "utils", "_fastsums.pyx"),
	#compiler_directives = {'linetrace': True},
	# annotate=True, # emit an html file with annotated C code
	)

setup(
	name = "Fast polymerize sums",
	ext_modules = fast_polymerize_sums_module,
	include_dirs = [np.get_include()]
	)

setup(
    name='vivarium-ecoli',
    version='0.0.1',
    packages=[
        'ecoli',
        'ecoli.composites',
        'ecoli.experiments',
        'ecoli.processes',
        'wholecell',
        'wholecell.utils',
        'reconstruction'],
    author='Eran Agmon, Ryan Spangler',
    author_email='eagmon@stanford.edu, ryan.spangler@gmail.com',
    url='https://github.com/CovertLab/vivarium-ecoli',
    license='MIT',
    entry_points={
        'console_scripts': []},
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'vivarium-cell>=0.0.17',
        'biopython==1.77',
        'Unum==4.1.4',
        'numba==0.50.1',
        'Theano==1.0.5',
        'ipython==7.16.1',
        ])
