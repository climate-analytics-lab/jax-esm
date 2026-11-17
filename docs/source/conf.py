# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import importlib.util
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
# The same string as `authors` in pyproject.toml.
author = 'JAX-ESM Developers'
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
    'sphinx.ext.intersphinx',
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

# The example notebooks are checked in with their outputs cleared, so a page
# built without executing them shows code only. Executing all six takes
# several minutes on a CPU (a few minutes each), which is too slow for an
# everyday docs build, so execution is an explicit switch: set
# JEM_DOCS_EXECUTE_NOTEBOOKS=1 (or pass `-e` to docs/build_docs.sh) to run
# every notebook while the docs are built and render its figures; otherwise
# the notebooks are rendered as they are checked in, without outputs. The
# examples CI job (tests/examples) is what proves the notebooks run on every
# pull request; this switch is for a build whose pages should show the plots.
# The timeout matches that test's per-notebook budget of 1800 s.
_execute_notebooks = os.environ.get('JEM_DOCS_EXECUTE_NOTEBOOKS', '') not in (
    '', '0', 'false', 'False')
nbsphinx_execute = 'always' if _execute_notebooks else 'never'
nbsphinx_timeout = 1800

templates_path = ['_templates']
# `docs/Makefile`'s `copy_examples` rsyncs `examples/` into
# `docs/source/examples/`, and MyST parses every `.md` under there; a
# `README.md` anywhere under `examples/` (the top-level index and one per
# example directory) is a GitHub-facing index (rendered on github.com, never
# linked from a toctree), not a docs page -- the actual Sphinx pages for the
# examples are `examples.rst`/`experimental.rst` -- so without this it is
# orphaned and warns "document isn't included in any toctree". Both patterns
# are needed: Sphinx's `**` glob requires at least one directory between
# `examples/` and the filename, so it alone misses the top-level
# `examples/README.md`.
exclude_patterns = ['examples/README.md', 'examples/**/README.md']

autosummary_generate = True
autosummary_generate_overwrite = True  # Regenerate on each build

# Helpers for the autosummary templates (jem_autosummary.py documents them).
# `jem_exports` is what `jem` re-exports, read by
# _templates/autosummary/exports.rst; `module_aliases` names the typing aliases
# a module assigns that autodoc would otherwise leave out, read by
# _templates/autosummary/module.rst.
sys.path.insert(0, os.path.abspath('_ext'))
import jem_autosummary  # noqa: E402

autosummary_context = {
    'jem_exports': jem_autosummary.exports(jem),
    'module_aliases': jem_autosummary.ModuleAliases(),
}

# The module pages (docs/source/_templates/autosummary/module.rst) are a bare
# `automodule`, so what each page documents is decided here: every public
# member (including those without a docstring, so a function or class is never
# silently absent from the reference), in source order, which is the order the
# modules are written to be read in. Inherited members are left out: the
# State/Forcing/Derived structs and the slab models inherit a large surface
# from tree_math and SlabModelBase that is documented where it is defined.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'member-order': 'bysource',
}

# `jem.components.veros_component` and `jem.components.veros.setups.*` import
# `veros` at module scope to subclass its `VerosSetup`. The jittable Veros
# fork is not on PyPI, so a build without it (Read the Docs installs only
# docs/requirements.txt and the project) mocks the package so those pages
# still import and render. When `veros` is importable (the GitHub `docs` job
# puts the fork on PYTHONPATH) nothing is mocked and autodoc documents the
# real classes; mocking an importable package would replace them.
autodoc_mock_imports = [] if importlib.util.find_spec('veros') else ['veros']

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# A struct documents its fields in an `Attributes` section while autodoc also
# documents each annotated field it finds (`undoc-members` above); rendered as
# `.. attribute::` directives the two collide as duplicate object
# descriptions. `:ivar:` fields render the section as a field list instead,
# leaving autodoc's entry the only anchor for the attribute.
napoleon_use_ivar = True


# -- Cross-references into other projects -------------------------------------
# The reference builds with -n (nitpicky): a type in a signature or docstring
# that resolves to nothing is a warning. Projects that publish an inventory
# are resolved through intersphinx; the rest are listed in nitpick_ignore_regex
# below, each for a stated reason, so that the check stays strict for every
# `jem.*` object.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'jax': ('https://docs.jax.dev/en/latest', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

nitpick_ignore_regex = [
    # Packages that publish no Sphinx inventory (jcm/jax-gcm, jax_datetime,
    # omegaconf/hydra, the Veros fork, dinosaur, coordax). Their objects appear
    # in jem's signatures and docstrings; there is nothing to link them to.
    ('py:.*', r'(jcm|jax_datetime|omegaconf|hydra|veros|dinosaur|coordax)(\..*)?'),
    # Import aliases written as such in docstring types (`jnp.ndarray`,
    # `xr.Dataset`, ...); the full names are resolved through intersphinx.
    ('py:.*', r'(jnp|jdt|xr|np|cx)\..*'),
    # Private helpers a docstring mentions by name. Private members are not
    # part of the reference, so there is no entry to link to.
    ('py:.*', r'_[A-Za-z0-9_]+'),
    # Names that appear unqualified in a signature because the annotation is a
    # string (`from __future__ import annotations`) and autodoc cannot qualify
    # it: the typing aliases documented at jem.base.component.{Carry,
    # Diagnostics,Exchanger} and pathlib.Path.
    ('py:class', r'Carry|Diagnostics|Exchanger|Path'),
    # `ESMFRegridders` subclasses `Mapping[str, ESMFRegridder]`; autodoc prints
    # the base unqualified.
    ('py:class', r'ESMFRegridder'),
    # Words napoleon turns into a type from a numpy-style "name : type" line
    # that is prose rather than a Python type ("optional", "path-like", ...).
    ('py:class', r'optional|callables?|iterables?|mappings?|sequences?|pairs?|path-like|pytree|Boolean|default .+'),
]


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


