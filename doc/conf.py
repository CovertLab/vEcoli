# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from datetime import datetime
import os
import shutil
import sys
sys.path.insert(0, os.path.abspath('..'))

from docutils.nodes import Text
from sphinx.addnodes import pending_xref
from sphinx.ext import apidoc
from sphinx.ext.intersphinx import missing_reference


# -- Project information -----------------------------------------------------

project = 'Vivarium E. coli'
author = 'The Vivarium E. coli Authors'
copyright = '2021-{}, {}'.format(datetime.now().year, author)

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'venv']

# Causes warnings to be thrown for all unresolvable references. This
# will help avoid broken links.
nitpicky = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for extensions --------------------------------------------------

# -- sphinx.ext.intersphinx options --
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'vivarium': (
        'https://vivarium-core.readthedocs.io/en/latest/',
        None,
    ),
}


# -- sphinx.ext.autodoc options --
autodoc_inherit_docstrings = False
# The Python dependencies aren't really required for building the docs
autodoc_mock_imports = [
    'cobra', 'arrow', 'IPython', 'six', 'numba', 'line-profiler',
    'matplotlib', 'sympy', 'iteround', 'vivarium_multibody', 'pytest',
    # Runs code on import and fails due to missing solvers.
    'wholecell.utils.modular_fba',
]
# Concatenate class and __init__ docstrings
autoclass_content = 'both'

def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    if name.startswith('test_'):
        return True
    return None


# -- Custom Extensions -------------------------------------------------

def run_apidoc(_):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    module_path = os.path.join(cur_dir, '..', 'ecoli')
    apidoc_dir = os.path.join(cur_dir, 'reference', 'api')
    if os.path.exists(apidoc_dir):
        shutil.rmtree(apidoc_dir)
    os.makedirs(apidoc_dir, exist_ok=True)
    apidoc.main(['-f', '-e', '-o', apidoc_dir, module_path])


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)
    app.connect('builder-inited', run_apidoc)
