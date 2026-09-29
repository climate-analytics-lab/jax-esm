"""Veros case setups: factories returning a ``veros.VerosSetup`` subclass.

Each module here is one ocean geometry -- :mod:`.double_drake` an idealised
two-continent basin on a uniform lat-lon grid, :mod:`.earth` a realistic
rotated-pole global ocean read from a SCRIP grid -- and is named from a
configuration's ``ocean.setup`` as an importable dotted path (never a file
path), e.g. ``jem.components.veros.setups.double_drake.double_drake_setup``.

This module imports nothing, for the same reason
:mod:`jem.components.veros` does not: Veros is an optional dependency, and
only a configuration that actually names one of these setups should have to
have it installed.
"""
