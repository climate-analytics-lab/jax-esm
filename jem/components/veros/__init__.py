"""Importable Veros case setups, for :data:`jem.config.ocean.veros`'s ``setup``.

The Veros *wrapper* -- the :class:`~jem.base.component.Component` adapter a
coupled run actually steps -- is :mod:`jem.components.veros_component`, one
level up. This package holds the *cases*: the factories under
:mod:`jem.components.veros.setups` that each return a ``veros.VerosSetup``
subclass for one idealised or realistic ocean geometry, which
``VerosComponent.from_setup`` imports by dotted path and builds.

This module imports nothing, deliberately: Veros is an optional dependency,
and importing it here would make ``import jem.components`` (and so
``import jem``) fail wherever Veros is not installed. Only a configuration
that actually names a setup under this package pays for importing Veros.
"""
