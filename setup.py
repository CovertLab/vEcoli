import os

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

build_sequences_module = cythonize(
	os.path.join("wholecell", "utils", "_build_sequences.pyx"),
	)

setup(
	name = "Build sequences",
	ext_modules = build_sequences_module,
	include_dirs = [np.get_include()],
	)

complexation_module = cythonize(
	os.path.join("wholecell", "utils", "mc_complexation.pyx"),
	)

setup(
	name = "Monte-carlo complexation",
	ext_modules = complexation_module,
	include_dirs = [np.get_include()],
	)

fast_polymerize_sums_module = cythonize(
	os.path.join("wholecell", "utils", "_fastsums.pyx"),
	)

setup(
	name = "Fast polymerize sums",
	ext_modules = fast_polymerize_sums_module,
	include_dirs = [np.get_include()],
	)
