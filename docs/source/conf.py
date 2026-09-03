# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


project = 'JAX-ESM'
copyright = '2026, Climate Analytics Lab'
author = 'Tien-Yiao Hsu and Climate Analytics Lab'

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

# Only execute notebooks that don't already have saved output cells.
nbsphinx_execute = 'auto'

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


