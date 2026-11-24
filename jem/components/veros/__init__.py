"""Importable Veros case setups, for ``jem/config/ocean/veros.yaml``'s ``setup``.

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
For the same reason it holds the one Veros-side constant JAX-ESM records,
``VEROS_SUPPORTED_REV``, as plain data.
"""

#: The jittable-Veros revision JAX-ESM is supported against.
#:
#: JAX-ESM runs a Veros fork (https://github.com/meteorologytoday/veros-jittable)
#: that is a source checkout, not a PyPI release, and whose development branch
#: moves. A branch name would mean the same JAX-ESM commit passes today and
#: fails tomorrow for a reason that is not in this repository, and ``main`` of
#: the fork is a different commit from the one the gates run against, so the
#: revision is recorded as a full commit sha, the one thing that names exactly
#: one tree. It is the commit ``dev/naive-fix-AD-needs-physical-justification``
#: pointed at when it was last validated against this repository's gates.
#:
#: ``.github/workflows/tests.yml`` (``env.VEROS_REV``), the ``Dockerfile``
#: (``ARG VEROS_REV``) and the install steps in ``README.md`` and
#: ``docs/source/getting_started.rst`` each name it, and
#: ``tests/unit/test_veros_contract.py`` fails if any of them drifts from this
#: constant.
#:
#: To bump it: check the new commit out, run the Veros gates (the two Veros
#: test files in their own pytest processes, as the workflow does, and a
#: ``+configuration=veros-earth coupled_run=short_run`` run), then change this
#: constant and every place the test names in the same commit.
VEROS_SUPPORTED_REV = "180366b73b0fc1938b3db8fbdb5426bc546905c6"
