"""Helpers the autosummary templates in ``_templates/autosummary`` call.

They live in an importable module, not in ``conf.py``, and ``ModuleAliases`` is
a class with ``__call__`` rather than a function, because they are handed to
the templates through ``autosummary_context`` and Sphinx caches a config value
only if it is serialisable, which it never judges a function (or a class
defined in ``conf.py``) to be.
"""

import ast
import importlib
import inspect

from sphinx.pycode import ModuleAnalyzer


def exports(package):
    """Dotted paths of the objects ``package`` re-exports in ``__all__``.

    Each path is the object's *defining* module, which is where its page is.
    A re-export that no jem module defines (a typing alias such as
    ``Exchanger``) is left out: ``ModuleAliases`` documents it on the page of
    the module that assigns it.
    """
    paths = []
    for name in package.__all__:
        obj = getattr(package, name)
        module = getattr(obj, '__module__', '')
        if module.startswith(f'{package.__name__}.') and hasattr(obj, '__qualname__'):
            paths.append(f'{module}.{obj.__qualname__}')
    return sorted(paths)


class ModuleAliases:
    """Callable returning ``_module_aliases(modname)``; see that function."""

    def __call__(self, modname):
        """Return the aliases ``modname`` assigns that autodoc would drop."""
        return _module_aliases(modname)


def _module_aliases(modname):
    """Public typing aliases ``modname`` assigns that autodoc would drop.

    A module-level alias (``Exchanger = Callable[...]``, ``Carry = Any``)
    reports the module of the typing object it names, not the module that
    assigns it, so autodoc's "defined in this module" test omits it from a
    module without ``__all__`` unless it carries a ``#:`` doc comment. It is
    still a public name that other pages cross-reference, so the module
    template documents it explicitly. Names with a ``#:`` comment are already
    documented and are excluded, so nothing is documented twice.
    """
    mod = importlib.import_module(modname)
    if hasattr(mod, '__all__') or not hasattr(mod, '__file__'):
        return []
    try:
        tree = ast.parse(inspect.getsource(mod))
        documented = {
            name for scope, name in ModuleAnalyzer.for_module(modname).find_attr_docs()
            if not scope
        }
    except (OSError, TypeError):
        return []
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            name = getattr(target, 'id', None)
            if not name or name.startswith('_') or name in documented:
                continue
            value = getattr(mod, name, None)
            if (hasattr(value, '__module__') and value.__module__ != modname
                    and not inspect.isclass(value)
                    and not inspect.isroutine(value)
                    and not inspect.ismodule(value)):
                names.append(name)
    return names
