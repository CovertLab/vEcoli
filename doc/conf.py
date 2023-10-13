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
from pprint import pformat
import shutil
import sys
sys.path.insert(0, os.path.abspath('..'))

import docutils
from docutils.nodes import Text
from docutils.parsers.rst import Parser
from sphinx.addnodes import pending_xref
from sphinx.ext import apidoc
from sphinx.ext.intersphinx import missing_reference


# -- Project information -----------------------------------------------------

project = 'Vivarium E. coli'
author = 'The Vivarium E. coli Authors'
copyright = '2021-{}, {}'.format(datetime.now().year, author)

# The full version, including alpha/beta/rc tags
release = '1.0.0'


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
    'nbsphinx',
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

# -- nbsphinx options --

# Never execute Jupyter notebooks.
nbsphinx_execute = 'never'

# -- sphinx.ext.intersphinx options --
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'vivarium': (
        'https://vivarium-core.readthedocs.io/en/latest/',
        None,
    ),
    'numpy': ('https://numpy.org/doc/stable', None),
    'unum': ('https://unum.readthedocs.io/en/latest/', None)
}


# -- sphinx.ext.autodoc options --
autodoc_inherit_docstrings = False
# The Python dependencies aren't really required for building the docs
autodoc_mock_imports = [
    'stochastic_arrow', 'numba', 'line-profiler', 'iteround', 'aesara', 'pandas',
    # Runs code on import and fails due to missing solvers.
    'wholecell.utils.modular_fba',
    # Runs code on import and fails due to missing packages
    'ecoli.library.parameters',
    'sympy', 'cv2', 'Bio', 'tqdm', 'cvxpy', 'pymunk', 'skimage', 'dill',
]
# Move typehints from signature into description
autodoc_typehints = "description"
# Concatenate class and __init__ docstrings
autoclass_content = 'both'
# Remove domain objects (e.g. functions, classes, attributes) from 
# table of contents
toc_object_entries = False

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

    notebooks_dst = os.path.join(cur_dir, 'notebooks')
    notebooks_src = os.path.join(cur_dir, '..', 'notebooks')
    if os.path.exists(notebooks_dst):
        shutil.rmtree(notebooks_dst)
    shutil.copytree(notebooks_src, notebooks_dst)

    exclude = (
        os.path.join(cur_dir, path) for path in (
            '../ecoli/analysis',
            '../ecoli/plots',
            '../ecoli/experiments/ecoli_master_sim_tests.py',
        )
    )

    # Custom templates to only put top-level document titles in 
    # table of contents
    template_dir = 'apidoc_templates/'
    apidoc.main(['-t', template_dir,
        '-f', '-e', '-E', '-M', '-o', apidoc_dir, module_path, *exclude])


objects_to_pprint = {}


def autodoc_process_signature_handler(
        _app, what, name, obj, _options, _signature, _return_annotation):
    '''Save class attributes before their signatures are processed.

    In ``object_description_handler``, we will use these saved objects
    to generate pretty representations of their default values.
    '''
    if what != 'attribute':
        return
    assert name not in objects_to_pprint
    objects_to_pprint[name] = obj


def object_description_handler(_, domain, objtype, contentnode):
    '''Make representations of attribute default values pretty.

    We transform the representations of the following attributes:

    * defaults
    * topology

    Using the objects saved by ``autodoc_process_signature_handler``,
    generate pretty representations of the objects and insert them into
    the documentation as a code block.

    WARNING: The algorithm for identifying attributes is clumsy. We just
    check that the paragraph starts with the attribute name and type
    hint (if applicable). This can cause problems if lines in the
    docstring look start with an attribute name. We attempt to mitigate
    this problem by also checking that the object name was saved by
    ``autodoc_process_signature_handler``.
    '''
    if objtype != 'class' or domain != 'py':
        return
    for child in contentnode.children:
        if child.astext().startswith('defaults: Dict['):
            name = contentnode.source.split()[-1] + '.defaults'
        elif child.astext().startswith('topology'):
            name = contentnode.source.split()[-1] + '.topology'
        else:
            continue

        obj = objects_to_pprint.get(name)
        if not obj:
            continue
        obj_pprint = pformat(obj, indent=4)
        value_node = child.children[0].children[-1]

        parser = Parser()
        settings = docutils.frontend.OptionParser(
            components=(docutils.parsers.rst.Parser,)).get_default_values()
        document = docutils.utils.new_document('<rst-doc>', settings=settings)
        lines = ['.. code-block:: python', '']
        for line in obj_pprint.split('\n'):
            lines.append('   ' + line)
        text = '\n'.join(lines)
        parser.parse(text, document)
        new_node = document.children[0]

        value_node.replace_self(new_node)


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)
    app.connect('autodoc-process-signature', autodoc_process_signature_handler)
    app.connect(
        'object-description-transform', object_description_handler)
    app.connect('builder-inited', run_apidoc)
