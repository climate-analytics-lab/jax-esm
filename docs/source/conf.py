# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# The version shown in the docs is read from the package, which is the single
# source of truth for it (pyproject.toml's `tool.setuptools.dynamic` reads the
# same attribute), so the docs cannot advertise a version the code does not
# have. The import has to follow the sys.path entry above, hence the noqa.
import jem  # noqa: E402

project = 'JAX-ESM'
copyright = '2026, Climate Analytics Lab'
author = 'Tien-Yiao Hsu and Climate Analytics Lab'
release = jem.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
    # MyST lets sphinx parse docs/source/design/*.md alongside the .rst
    # pages (same setup as jax-gcm).
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
myst_heading_anchors = 3

# Never execute notebooks as part of the docs build. The example notebooks are
# checked in with their outputs cleared, so 'auto' would execute every one of
# them on every docs build (~16 minutes) -- and the docs environment does not
# install the Veros fork the experimental examples need, so those would fail
# outright. The examples CI job is what executes the notebooks and proves they
# run; the docs show their code only. How (and whether) to publish executed
# outputs is a Phase 3 decision.
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
autosummary_generate_overwrite = True  # Regenerate on each buil

napoleon_google_docstring = True
napoleon_numpy_docstring = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']


#nbsphinx_prolog = """
#{% if env.doc2path(env.docname, base=None).endswith('.ipynb') %}
#Download `current notebook <{{ env.doc2path(env.docname, base=None) }}>`_
#{% endif %}
#----
#"""


