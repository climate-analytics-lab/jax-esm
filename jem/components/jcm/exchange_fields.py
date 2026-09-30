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
``stress_u``/``stress_v``     N m-2, **positive down**: the       ``eastward_wind_stress``/                  N m-2, the stress ON the
                               eastward/northward momentum flux    ``northward_wind_stress``                  surface, unchanged
                               into the surface
``wind_u``/``wind_v``          m s-1, at the package's own         ``u0``/``v0``                              m s-1, unchanged
                               reference (``wind_reference``)
``sensible_heat_flux``        (not exchanged -- folded into        --                                         --
``latent_heat_flux``           jem's own ``total_heat_flux``)
``wind_speed``                m s-1, scalar                        (not exchanged -- ``wind_u``/``wind_v``
                                                                     already carry the direction)
============================  =================================  ========================================  ========================

- **``total_heat_flux = -net_heat_flux``.** The only sign flip needed: jax-gcm
  publishes the net downward flux into the surface medium (SPEEDY's
  ``hfluxn`` convention, SW_net + LW_net - SHF - LHF); JEM's convention is the
  net flux leaving the surface into the atmosphere, i.e. exactly the
  negative.
- **``evaporation``, ``precipitation``, ``stress_u``/``stress_v`` and
  ``wind_u``/``wind_v`` need no unit conversion**, only the same reshape
  ``total_heat_flux`` needs -- see below.
- **The wind stress keeps the contract's sign.** jax-gcm's "positive down"
  for a momentum flux means the stress the atmosphere exerts *on* the
  surface (minus the stress on the atmosphere; surface westerlies give
  ``stress_u > 0``), which is already the convention an ocean integrates --
  Veros' ``surface_taux`` pushes the water eastward when positive -- and is
  JEM's own convention for a stress (``CLAUDE.md``, "Sign and mask
  conventions"), so there is nothing to negate. The JEM names say
  *eastward*/*northward* because the atmosphere's grid is unrotated, and a
  consumer on a rotated grid must rotate the pair as a vector (see
  :func:`jem.fluxes.rotate_vector`).
- **The wind stress is the one the atmosphere received.** jax-gcm fills
  every guaranteed field from the fluxes the atmosphere column was actually
  given that step ("published == delivered"), stability-corrected by the
  package's own surface closure, so a component that takes this stress
  receives exactly the momentum the atmosphere lost: momentum is conserved
  across the interface up to the regrid, which the ocean coupling does
  conservatively. That is why :class:`jem.fluxes.VerosExchange` uses it
  instead of deriving a stress of its own from ``u0``/``v0`` (jax-esm#132).
- **``sensible_heat_flux``/``latent_heat_flux``/``wind_speed`` are on the
  contract but JEM does not exchange them separately**: the slab
  ocean/land/sea-ice models and ``jem.exchangers`` couple on the *net*
  ``total_heat_flux``, and ``wind_speed`` is redundant once the vector is
  available (``hypot(wind_u, wind_v) == wind_speed`` is a contract
  invariant).
- **Every field is a grid-box mean over land, sea and sea ice** -- the only
  thing the contract guarantees. An ocean coupled through a coastal cell
  therefore receives that cell's *mean* heat, freshwater and momentum flux,
  land fraction included, rather than an open-water one; the contract's
  per-tile fields are where an open-water flux would come from, and neither
  package fills them for the fluxes yet (jax-esm#147).

Column-vectorized packages publish flat, not gridded
------------------------------------------------------
Every guaranteed field's horizontal layout is whatever the *publishing*
physics package's ``ComposablePhysics`` was built with -- ``(ix, il)`` for a
grid-hosted package (SPEEDY's default), or a flat ``(ncols,)`` for one built
with ``vectorize_columns=True`` (ECHAM's default). Only the accumulated
*tendencies* ``ComposablePhysics.compute_tendencies`` returns are
un-flattened back to the grid; the diagnostics dict, including
``diagnostics["surface_exchange"]``, is returned exactly as the terms built
it. :func:`from_diagnostics` therefore reshapes every guaranteed field onto
the atmosphere's nodal ``(ix, il)`` shape unconditionally -- a no-op for an
already-gridded field, and the exact inverse of jax-gcm's own flatten
(a plain C-order ``reshape(ncols)``, longitude-major) for a flat one, so no
transpose or column-order bookkeeping is needed here.

The near-surface wind's reference height is package-specific
-----------------------------------------------------------------
:attr:`~jem.components.jcm.exchange_fields.SurfaceExchange.u0`/``v0`` are read
verbatim from the contract's ``wind_u``/``wind_v``, which sit at whichever
reference the publishing package's own surface closure diagnoses -- the
struct's static ``wind_reference`` field names which: SPEEDY's
``fwind0``-scaled lowest model level (``"lowest_level"``), ECHAM's
stability-corrected 10 m wind (``"10m"``). JEM publishes the wind for a
consumer that needs the wind itself (a sea-ice free-drift term, a wave
model) and says nothing about its height beyond that; nothing in JEM turns it
into a stress, because the stress is published directly and a drag law
applied to winds at two different heights would give each package a
different ocean forcing for the same flow.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jcm.physics.surface.surface_exchange import surface_exchange_from


class SurfaceExchange(NamedTuple):
    """Surface fluxes, wind stress and near-surface wind, in JEM's conventions.

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
    eastward_wind_stress : jax.Array
        Eastward stress the atmosphere exerts on the surface, ``N m-2``
        (jax-gcm's ``stress_u``, sign unchanged: surface westerlies give a
        positive value). See the module docstring for why it is the stress
        a surface component should take.
    northward_wind_stress : jax.Array
        Northward stress the atmosphere exerts on the surface, ``N m-2``
        (jax-gcm's ``stress_v``). Same convention.
    u0 : jax.Array
        Near-surface zonal wind, ``m s-1``, at the publishing package's own
        reference (see the module docstring's "near-surface wind" section).
    v0 : jax.Array
        Near-surface meridional wind, ``m s-1``. Same reference as ``u0``.

    """

    total_heat_flux: jnp.ndarray
    evaporation: jnp.ndarray
    precipitation: jnp.ndarray
    eastward_wind_stress: jnp.ndarray
    northward_wind_stress: jnp.ndarray
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
        The fluxes, stress and wind translated to JEM's sign and unit
        conventions, each field shaped ``nodal_shape``.

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
    return SurfaceExchange(
        total_heat_flux=(-exchange.net_heat_flux).reshape(nodal_shape),
        evaporation=exchange.evaporation.reshape(nodal_shape),
        precipitation=exchange.precipitation.reshape(nodal_shape),
        eastward_wind_stress=exchange.stress_u.reshape(nodal_shape),
        northward_wind_stress=exchange.stress_v.reshape(nodal_shape),
        u0=exchange.wind_u.reshape(nodal_shape),
        v0=exchange.wind_v.reshape(nodal_shape),
    )
