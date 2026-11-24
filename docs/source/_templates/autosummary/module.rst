{{ fullname | escape | underline }}

.. Every module page documents the module's own public API in full (see
   ``autodoc_default_options`` in conf.py), so each function, class and method
   has an anchor a cross-reference can land on. A module with an ``__all__``
   (a package re-exporting its parts) documents the names in ``__all__`` under
   the package path, which is the path the rest of the documentation and the
   docstrings use for them; a module without one documents what it defines.

.. automodule:: {{ fullname }}
{% for item in module_aliases(fullname) %}
.. autodata:: {{ fullname }}.{{ item }}
{% endfor %}
