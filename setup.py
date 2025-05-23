import os
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# List of your Cython extension modules
extensions = [
    Extension(
        "wholecell.utils._build_sequences",
        [os.path.join("wholecell", "utils", "_build_sequences.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "wholecell.utils.mc_complexation",
        [os.path.join("wholecell", "utils", "mc_complexation.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "wholecell.utils._fastsums",
        [os.path.join("wholecell", "utils", "_fastsums.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "wholecell.utils._trna_charging",
        [os.path.join("wholecell", "utils", "_trna_charging.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

# Use cythonize on the extensions list
setup(
    name="Vivarium Ecoli Extensions",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)
