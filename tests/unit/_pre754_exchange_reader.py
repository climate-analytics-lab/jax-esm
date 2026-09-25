"""Frozen copy of ``jem/components/jcm/exchange_fields.py`` as of ``756cc2c``.

``756cc2c`` is the last commit before the jax-gcm#754 migration (jax-gcm PR
877) collapsed that module onto jax-gcm's single ``SurfaceExchange``
contract. This file exists **only** so
``test_speedy_new_reader_agrees_with_the_pre_754_reader`` in
``test_jcm_component.py`` can pin the migration's sign and unit mapping --
comparing the new single reader against the pre-migration one on a real
model step -- without depending on git history being present in the
checkout.

CI's ``actions/checkout`` makes a shallow clone, so ``git show
756cc2c:...`` (what this module replaces) fails there with exit status 128:
the commit simply is not in the runner's object store. Loading the old
reader's source from history therefore made the PR's most important
verification silently not run in CI at all, while appearing to pass
locally (a full-history development checkout hides the problem). This
module is the hermetic fix: the old code, vendored once, executed exactly
as-is.

Only what the equivalence test exercises is kept -- the SPEEDY reader,
``GRAMS_PER_KILOGRAM``, the diagnostics keys it reads and the
``SurfaceExchange`` NamedTuple it returns. The old ``echam()`` reader (which
only ever raised ``NotImplementedError``) and ``detect()`` are dropped as
dead weight; they are not part of the equivalence this test checks.

**Do not import this from production code, and do not "fix" or "update" it
to track the current ``jem.components.jcm.exchange_fields``.** Its entire
value is that it is frozen: it must keep computing the pre-#754 mapping
byte for byte, forever, as the fixed point the new reader is checked
against. If the mapping under test ever needs to change, that is a change
to the *new* reader and to what this test asserts, not to this file.

Source: ``git show 756cc2c:jem/components/jcm/exchange_fields.py``
(verbatim below, trimmed as described above).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

# JCM's SPEEDY physics reports the surface water fluxes as mass density
# fluxes in g m-2 s-1 (``jcm/physics/speedy/units_table.csv``: ``evap``,
# ``precnv``, ``precls``); JEM works in SI, so every one of them is divided
# by this.
GRAMS_PER_KILOGRAM = 1000.0

# The diagnostics key that identifies SPEEDY physics output: SPEEDY's
# surface-flux term writes ``_surface_flux`` (leading underscore: a struct,
# not a plain output array; ``jcm/physics/surface/speedy_surface_flux.py``).
SPEEDY_SURFACE_KEY = "_surface_flux"


class SurfaceExchange(NamedTuple):
    """Surface fluxes and near-surface wind, in JEM's conventions.

    Every field is a ``(ix, il)`` horizontal map on the atmosphere's nodal
    grid (the grid-cell mean over land and sea; JCM no longer publishes the
    per-tile breakdown).

    Attributes
    ----------
    total_heat_flux : jax.Array
        Net heat flux out of the surface, ``W m-2``, **positive upward**.
    evaporation : jax.Array
        Evaporation, ``kg m-2 s-1``, positive upward.
    precipitation : jax.Array
        Total precipitation (convective plus large-scale) reaching the
        surface, ``kg m-2 s-1``, positive downward.
    u0 : jax.Array
        Near-surface zonal wind, ``m s-1``.
    v0 : jax.Array
        Near-surface meridional wind, ``m s-1``.

    """

    total_heat_flux: jnp.ndarray
    evaporation: jnp.ndarray
    precipitation: jnp.ndarray
    u0: jnp.ndarray
    v0: jnp.ndarray


def speedy(diagnostics: dict[str, Any]) -> SurfaceExchange:
    """Read the surface exchange out of SPEEDY physics diagnostics.

    Parameters
    ----------
    diagnostics : dict
        One coupling step's physics diagnostics dict, with the length-1
        save axis already stripped.

    Returns
    -------
    SurfaceExchange
        The fluxes converted to JEM's sign and unit conventions.

    Notes
    -----
    Source fields and their JCM conventions (``jcm/physics/surface/
    speedy_surface_flux.py`` and ``jcm/physics/speedy/units_table.csv``):

    ``_surface_flux.hfluxn``
        ``W m-2``, net heat flux **downward** into the surface, weighted by
        land fraction over the land/sea tiles. JEM takes heat flux positive
        upward, so it is negated exactly here, at the component boundary.
    ``_surface_flux.evap``
        ``g m-2 s-1``, evaporation, already upward-positive.
    ``_convection.precnv``, ``_condensation.precls``
        ``g m-2 s-1``, convective and large-scale precipitation, downward.
    ``_surface_flux.u0``, ``_surface_flux.v0``
        ``m s-1``, wind extrapolated to the surface layer (sigma = 0.99).

    """
    surface_flux = diagnostics[SPEEDY_SURFACE_KEY]
    precipitation = (
        diagnostics["_convection"].precnv + diagnostics["_condensation"].precls
    )
    return SurfaceExchange(
        total_heat_flux=-surface_flux.hfluxn,
        evaporation=surface_flux.evap / GRAMS_PER_KILOGRAM,
        precipitation=precipitation / GRAMS_PER_KILOGRAM,
        u0=surface_flux.u0,
        v0=surface_flux.v0,
    )
