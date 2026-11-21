{{ fullname | escape | underline }}

.. A package page documents the package docstring only. What the package
   re-exports is documented once, on the page of the module that defines it
   (an unqualified cross-reference to such a name would be ambiguous if the
   re-export were documented as well), and its submodules have pages of their
   own.

.. automodule:: {{ fullname }}
   :no-members:
