{{ fullname | escape | underline }}

.. A package page for a package that re-exports names from its submodules. Each
   name is documented once, on the page of the module that defines it, so that
   an unqualified cross-reference to it is never ambiguous; the table below
   links there. `jem_exports` is built from `jem.__all__` in conf.py, so the
   table cannot drift from the package.

.. automodule:: {{ fullname }}
   :no-members:

.. The table lists full paths, so it must not be read relative to the module
   the automodule above made current.

.. currentmodule:: None

.. autosummary::
{% for item in jem_exports %}
   ~{{ item }}
{%- endfor %}
