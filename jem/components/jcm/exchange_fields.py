"""Surface exchange fields, read off jax-gcm's package-independent contract.

:class:`jcm.physics.surface.surface_exchange.SurfaceExchange` is published
identically under ``diagnostics["surface_exchange"]`` by every physics
package that resolves a surface (SPEEDY and ECHAM; Held-Suarez opts out
because it has no surface fluxes at all), so this module has exactly one
reader, :func:`from_diagnostics`, and no per-package branch.

Sign and unit reconciliation
-----------------------------
The two conventions differ, and getting this wrong silently inverts the
coupling rather than failing loudly, so the mapping is spelled out field by
field, against ``docs/source/design/surface_exchange.md`` (jax-gcm) and this
repository's own ``CLAUDE.md`` ("Sign and mask conventions"):

============================  =================================  ========================================  ========================
Contract field (jax-gcm)      jax-gcm sign / units                JEM field                                  JEM sign / units
============================  =================================  ========================================  ========================
``net_heat_flux``             W m-2, **positive down** (into the  ``total_heat_flux``                        W m-2, **positive up**
                               surface medium)
``evaporation``                kg m-2 s-1, positive up            ``evaporation``                            kg m-2 s-1, positive up
``precipitation``              kg m-2 s-1, positive down, >= 0     ``precipitation``                          kg m-2 s-1, positive down
``wind_u``/``wind_v``          m s-1, at the package's own         ``u0``/``v0``                              m s-1, unchanged
                               reference (``wind_reference``)
``sensible_heat_flux``        (not exchanged -- folded into        --                                         --
``latent_heat_flux``           jem's own ``total_heat_flux``)
``stress_u``/``stress_v``     (not exchanged -- JEM's own          --                                         --
                               Veros coupling computes its own
                               stress independently, see below)
``wind_speed``                m s-1, scalar                        (not exchanged -- ``wind_u``/``wind_v``
                                                                     already carry the direction)
============================  =================================  ========================================  ========================

- **``total_heat_flux = -net_heat_flux``.** The only sign flip needed: jax-gcm
  publishes the net downward flux into the surface medium (SPEEDY's
  ``hfluxn`` convention, SW_net + LW_net - SHF - LHF); JEM's convention is the
  net flux leaving the surface into the atmosphere, i.e. exactly the
  negative.
- **``evaporation``, ``precipitation``, ``wind_u``/``wind_v`` need no unit
  conversion**, only the same reshape ``total_heat_flux`` needs -- see below.
- **``sensible_heat_flux``/``latent_heat_flux``/``stress_u``/``stress_v``/
  ``wind_speed`` are on the contract but JEM does not exchange them
  separately**: the slab ocean/land/sea-ice models and ``jem.exchangers``
  couple on the *net* ``total_heat_flux``, and momentum is not exchanged
  through this struct at all -- :class:`jem.fluxes.VerosExchange` derives its
  own stress from ``u0``/``v0`` with an independent bulk drag law rather than
  reusing ``stress_u``/``stress_v`` (see that module's docstring for why).
  ``wind_speed`` is redundant once the vector is available
  (``hypot(wind_u, wind_v) == wind_speed`` is a contract invariant).

Column-vectorized packages publish flat, not gridded
------------------------------------------------------
Every guaranteed field's horizontal layout is whatever the *publishing*
physics package's ``ComposablePhysics`` was built with -- ``(ix, il)`` for a
grid-hosted package (SPEEDY's default), or a flat ``(ncols,)`` for one built
with ``vectorize_columns=True`` (ECHAM's default:
:func:`jcm.physics.echam.echam_terms.echam_physics` passes it). Only the
accumulated *tendencies* ``ComposablePhysics.compute_tendencies`` returns are
un-flattened back to the grid before the caller ever sees them; the
diagnostics dict -- everything under ``diagnostics["surface_exchange"]``
included -- is returned exactly as the terms built it, flat columns and all
(see ``jcm.physics.composable_physics.ComposablePhysics.compute_tendencies``,
whose ``_reshape_tendencies_to_3d`` call is scoped to the tendency
accumulator, not ``diagnostics``). :func:`from_diagnostics` therefore takes
the atmosphere's nodal ``(ix, il)`` shape and reshapes every guaranteed field
onto it, unconditionally: a field already gridded reshapes to itself
(no-op), and a flat one is recovered exactly, because jax-gcm's own flatten
is a plain C-order ``reshape(ncols)`` of an ``(ix, il)`` array with longitude
as the major axis (``jcm.physics.composable_physics._flattened_column_sharding``'s
docstring) -- the exact inverse of ``reshape(ix, il)``, so no transpose or
column-order bookkeeping is needed on this side either. This one reshape is
what makes an ECHAM-composed (column-vectorized) atmosphere's surface
exchange usable at all: before jax-gcm#911/#914 unblocked the wind, no
coupled ECHAM step ran far enough to reach this, so the shape mismatch
between the flat diagnostics and JEM's gridded ``JCMDerived`` carry -- a
``lax.scan`` carry-structure error identical in kind to (but distinct from)
jax-esm#129's wind-vector block -- was never exercised.

The near-surface wind's reference height is package-specific
-----------------------------------------------------------------
:attr:`~jem.components.jcm.exchange_fields.SurfaceExchange.u0`/``v0`` are the
near-surface wind's true-east/true-north *components*, needed by
:func:`jem.fluxes.bulk_wind_stress` (the independent bulk-drag law
:class:`jem.fluxes.VerosExchange` applies for a Veros ocean). They are read
verbatim from the contract's ``wind_u``/``wind_v``, which sit at whichever
reference the *publishing* package's own surface closure defines -- ECHAM's
stability-corrected 10 m wind, SPEEDY's ``fwind0``-scaled lowest-level wind
-- recorded in the struct's static ``wind_reference`` field
(``jcm.physics.surface.surface_exchange.WIND_REFERENCES``). JAX-ESM does not
read ``wind_reference`` today: :func:`jem.fluxes.bulk_wind_stress` applies
the same bulk law regardless of which reference the wind it is handed sits
at (it was already an independent computation from SPEEDY's own bulk
formula, not a lookup of it -- see that function's docstring), so there is
no reference-dependent branch to drive with it. A consumer that DOES need to
know the height reads ``diagnostics["surface_exchange"].wind_reference``
directly off the opaque ``JCMDerived.physics`` passthrough.

What this replaces
-------------------
Before jax-gcm#754, this module read SPEEDY's private ``_surface_flux``/
``_convection``/``_condensation`` diagnostics keys by hand and could not
build a grid-mean struct for ECHAM at all. #754 (jax-gcm PR 877) collapsed
the heat and water fluxes onto one published struct but left the near-surface
wind a private, SPEEDY-only read (``_surface_flux.u0``/``.v0``), because the
contract published only the scalar ``wind_speed`` -- a direction cannot be
recovered from a magnitude, and ECHAM's boundary-layer scheme diagnosed only
a wind *speed*, nothing else. jax-gcm#911/#914 closes that gap by adding
``wind_u``/``wind_v`` to the published contract for every package, so this
module's read of SPEEDY's private diagnostics key is gone along with the
per-package branch it required: ``from_diagnostics`` needs no physics-package
knowledge at all any more, for the wind or anything else. That is what makes
an ECHAM-composed coupled model reach its first real step (closing
jax-esm#129) -- reaching it is also what surfaced the column-vectorization
reshape above, a second, independent gap the wind block had been hiding.

The old ``speedy()`` reader's source (commit ``756cc2c``, the
last commit before the #754 collapse) is vendored, frozen, as
``tests/unit/_pre754_exchange_reader.py`` and used directly, as the
historical baseline, by the numeric old-vs-new equivalence test in
``tests/unit/test_jcm_component.py``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jcm.physics.surface.surface_exchange import surface_exchange_from


class SurfaceExchange(NamedTuple):
    """Surface fluxes and near-surface wind, in JEM's conventions.

    Every field is a ``(ix, il)`` horizontal map on the atmosphere's nodal
    grid (the grid-cell mean over land and sea -- jax-gcm's contract
    guarantees only the grid mean; see
    ``docs/source/design/surface_exchange.md``), reshaped onto it by
    :func:`from_diagnostics` if the publishing package built it flat (see the
    module docstring's "Column-vectorized packages publish flat, not
    gridded" section).

    Attributes
    ----------
    total_heat_flux : jax.Array
        Net heat flux out of the surface, ``W m-2``, **positive upward**.
    evaporation : jax.Array
        Evaporation, ``kg m-2 s-1``, positive upward.
    precipitation : jax.Array
        Total precipitation (convective plus large-scale/stratiform)
        reaching the surface, ``kg m-2 s-1``, positive downward.
    u0 : jax.Array
        Near-surface zonal wind, ``m s-1``, at the publishing package's own
        reference (see the module docstring's "near-surface wind" section).
    v0 : jax.Array
        Near-surface meridional wind, ``m s-1``. Same reference as ``u0``.

    """

    total_heat_flux: jnp.ndarray
    evaporation: jnp.ndarray
    precipitation: jnp.ndarray
    u0: jnp.ndarray
    v0: jnp.ndarray


def from_diagnostics(
    diagnostics: dict[str, Any], *, nodal_shape: tuple[int, int],
) -> SurfaceExchange:
    """Read the surface exchange out of one step's physics diagnostics.

    Every guaranteed field of :class:`jcm.physics.surface.surface_exchange.
    SurfaceExchange` is filled identically by every physics package that
    resolves a surface, so this function needs no knowledge of which package
    produced ``diagnostics`` beyond the grid it is on. See the module
    docstring for the field-by-field sign/unit mapping and for why every
    field is reshaped onto ``nodal_shape`` unconditionally.

    Parameters
    ----------
    diagnostics : dict
        One coupling step's physics diagnostics dict, with the length-1
        save axis already stripped.
    nodal_shape : tuple of int
        The atmosphere's horizontal nodal shape, ``(ix, il)``
        (``jcm.model.Model.coords.nodal_shape[1:]``, the same value
        :class:`~jem.components.jcm.component.JCMComponent` caches as
        ``self.nodal_shape``). Required, not inferred: a package built
        ``vectorize_columns=True`` (ECHAM's default) publishes every field
        flat, ``(ix * il,)``, and there is no reliable way to recover the
        grid shape from a 1-D array alone.

    Returns
    -------
    SurfaceExchange
        The fluxes and wind translated to JEM's sign and unit conventions,
        each field shaped ``nodal_shape``.

    Raises
    ------
    KeyError
        If the composed physics package publishes no ``surface_exchange``
        struct at all (Held-Suarez opts out -- it resolves no surface
        fluxes). Raised by
        :func:`jcm.physics.surface.surface_exchange.surface_exchange_from`,
        which names the packages that do publish and points at the design
        doc. Prefer
        :meth:`~jcm.physics.composable_physics.ComposablePhysics.require_surface_exchange`
        at composition time (see ``jem.runners.build_atmosphere``) so this
        is caught before the first coupled step rather than during it.

    """
    exchange = surface_exchange_from(diagnostics)

    def _gridded(field: jnp.ndarray) -> jnp.ndarray:
        return field.reshape(nodal_shape)

    return SurfaceExchange(
        total_heat_flux=_gridded(-exchange.net_heat_flux),
        evaporation=_gridded(exchange.evaporation),
        precipitation=_gridded(exchange.precipitation),
        u0=_gridded(exchange.wind_u),
        v0=_gridded(exchange.wind_v),
    )
