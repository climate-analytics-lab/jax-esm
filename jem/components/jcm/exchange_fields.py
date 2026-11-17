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

.. list-table::
   :header-rows: 1

   * - Contract field (jax-gcm)
     - jax-gcm sign / units
     - JEM field
     - JEM sign / units
   * - ``net_heat_flux``
     - W m-2, **positive down** (into the surface medium)
     - ``total_heat_flux``
     - W m-2, **positive up**
   * - ``evaporation``
     - kg m-2 s-1, positive up
     - ``evaporation``
     - kg m-2 s-1, positive up
   * - ``precipitation``
     - kg m-2 s-1, positive down, >= 0
     - ``precipitation``
     - kg m-2 s-1, positive down
   * - ``stress_u``/``stress_v``
     - N m-2, **positive down**: the eastward/northward momentum flux into
       the surface
     - ``eastward_wind_stress``/``northward_wind_stress``
     - N m-2, the stress ON the surface, unchanged
   * - ``wind_u``/``wind_v``
     - m s-1, at the package's own reference (``wind_reference``)
     - ``u0``/``v0``
     - m s-1, unchanged
   * - the same fields' ``sea`` tile (or ``water`` + ``sea_ice``)
     - as the grid-mean field, over the sea part of the cell
     - ``sea_total_heat_flux``, ``sea_evaporation``,
       ``sea_eastward_wind_stress``/``sea_northward_wind_stress``
     - as the unprefixed field
   * - ``sensible_heat_flux``, ``latent_heat_flux``
     - (not exchanged -- folded into JEM's own ``total_heat_flux``)
     - --
     - --
   * - ``wind_speed``
     - m s-1, scalar, at ``wind_reference``
     - ``wind_speed``
     - m s-1, unchanged
   * - ``downward_shortwave_flux``, ``downward_longwave_flux``
     - W m-2, positive down; the longwave incident
     - same names
     - unchanged
   * - ``surface_pressure``
     - Pa
     - ``normalized_surface_pressure``
     - over jax-gcm's ``p0``, dimensionless
   * - ``air_temperature``/``air_specific_humidity`` (``sea`` tile where
       published)
     - K, kg kg-1, at ``air_reference``
     - ``sea_air_temperature``/``sea_air_specific_humidity``
     - unchanged
   * - ``precip_rain``/``precip_snow``, or ``precipitation`` split by
       ``air_potential_temperature`` and ``surface_pressure``
     - kg m-2 s-1, positive down
     - ``rainfall``/``snowfall``
     - unchanged (:func:`split_precipitation`)

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
- **``sensible_heat_flux``/``latent_heat_flux`` are on the contract but JEM
  does not exchange them separately**: the slab ocean/sea-ice models and
  ``jem.exchangers`` couple on the *net* ``total_heat_flux``.
- **What a thermodynamic sea ice reads.** A sea ice that solves its own
  surface energy balance (the Winton ice) needs the atmosphere's state rather
  than its fluxes, and takes it unchanged from the contract: the downward
  shortwave and the incident downward longwave (``downward_shortwave_flux``,
  ``downward_longwave_flux``), the scalar ``wind_speed`` at the package's
  reference, the surface pressure divided by jax-gcm's ``p0``
  (``normalized_surface_pressure``) and the air the sea's fluxes were
  computed against (``sea_air_temperature``, ``sea_air_specific_humidity``:
  the ``sea`` tile of ``air_temperature_tile``/``air_specific_humidity_tile``,
  else the grid value). It also needs precipitation by phase:
  ``rainfall``/``snowfall`` are the package's own split where it publishes
  one, else MITgcm AIM's rule (:func:`split_precipitation`).
- **The unprefixed fields are grid-box means over land, sea and sea ice**;
  the ``sea_*`` fields are the same fluxes over the sea part of the cell
  alone (open water and sea ice together), which is what an ocean takes. They
  come from the contract's per-tile fields, read by tile name: the ``sea``
  tile where the package publishes one (SPEEDY, whose grid mean is exactly
  ``(1 - fmask) * sea + fmask * land``), or the fraction-weighted mean of
  its ``water`` and ``sea_ice`` tiles. A package that publishes no flux tile
  (ECHAM: its implicit vertical-diffusion solve delivers only the grid mean)
  leaves the ``sea_*`` fields equal to the grid mean, so an ocean on a coast
  cell then also receives the land part's flux; :class:`JCMComponent` logs
  that once when it builds its carry. ``land_evaporation`` is the evaporation
  over the land part, the ``land`` tile, which a land component's soil
  supplies; it falls back to the grid mean the same way, and
  :func:`jem.exchangers.default_exchanges` refuses to couple a land to a
  package that has no land tile. See ``docs/source/design/exchange.md`` and
  jax-gcm's ``docs/source/design/surface_exchange.md``.

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
import jcm.constants as jcm_constants
from jcm.physics.surface.surface_exchange import surface_exchange_from


#: The temperature below which MITgcm's AIM (SPEEDY's physics in MITgcm) makes
#: a cell's precipitation snow for its thermodynamic sea ice: the freezing point
#: of pure water, ``tFreeze = celsius2K`` (``pkg/aim_v23/phy_inphys.F``).
SNOW_TEMPERATURE_THRESHOLD = 273.15


def split_precipitation(exchange: Any) -> tuple[Any, Any]:
    """Return ``(rainfall, snowfall)``, ``kg m-2 s-1``, in the publisher's layout.

    A package that splits its precipitation by phase (``precip_rain``,
    ``precip_snow``; ECHAM's microphysics) is taken at its word. SPEEDY
    condenses liquid only, so its precipitation is split as MITgcm's AIM
    splits SPEEDY's for its thermodynamic sea ice
    (``pkg/aim_v23/phy_snow_precip.F``, ``aim_aim2sioce.F``): all of a cell's
    precipitation is snow where the lowest level's potential temperature
    brought to the surface pressure, ``theta (ps / p0)**kappa``, is below
    :data:`SNOW_TEMPERATURE_THRESHOLD`, and rain otherwise. That is the
    lowest level's temperature extrapolated dry-adiabatically to the
    surface, AIM's ``ThA(kGrd) * PSA**kappa``.

    AIM also corrects the atmosphere's condensation heating for the latent
    heat of fusion of the snow it makes. SPEEDY in jax-gcm does not, so the
    snow arrives as water that never released that heat, and the ice that
    stores it has to account for it
    (``WintonSeaiceModel(frozen_precipitation_latent_heat_release_mode=True)``,
    its default).
    """
    precipitation = exchange.precipitation
    if exchange.precip_rain is not None and exchange.precip_snow is not None:
        return exchange.precip_rain, exchange.precip_snow
    surface_air = (exchange.air_potential_temperature
                   * (exchange.surface_pressure / jcm_constants.p0) ** jcm_constants.akap)
    snow = surface_air < SNOW_TEMPERATURE_THRESHOLD
    return (jnp.where(snow, 0.0, precipitation),
            jnp.where(snow, precipitation, 0.0))


class SurfaceExchange(NamedTuple):
    """Surface fluxes, wind stress and near-surface wind, in JEM's conventions.

    Every field is a ``(ix, il)`` horizontal map on the atmosphere's nodal
    grid -- the grid-cell mean over land and sea, or for the ``sea_*``
    fields the mean over the sea part of the cell (see the module
    docstring) -- reshaped onto it by
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
    sea_total_heat_flux, sea_evaporation : jax.Array
        ``total_heat_flux`` and ``evaporation`` over the sea part of the
        cell (see :func:`sea_tile`), in the same conventions.
    sea_eastward_wind_stress, sea_northward_wind_stress : jax.Array
        The stress on the sea part of the cell, in the same conventions.
    land_evaporation : jax.Array
        ``evaporation`` over the land part of the cell (the ``land`` tile).
    downward_shortwave_flux : jax.Array
        Downward shortwave at the surface, ``W m-2``, positive down.
    downward_longwave_flux : jax.Array
        Incident downward longwave at the surface, ``W m-2``, positive down:
        the flux in the air, before a surface's emissivity takes its share.
    wind_speed : jax.Array
        Near-surface wind speed, ``m s-1``, at the same reference as
        ``u0``/``v0``.
    normalized_surface_pressure : jax.Array
        Surface pressure over jax-gcm's reference ``p0``, dimensionless.
    sea_air_temperature, sea_air_specific_humidity : jax.Array
        The air, ``K`` and ``kg kg-1``, the sea's fluxes were computed
        against, at the package's ``air_reference``.
    rainfall, snowfall : jax.Array
        Precipitation by phase, ``kg m-2 s-1``, positive down; they sum to
        ``precipitation`` (:func:`split_precipitation`).

    """

    total_heat_flux: jnp.ndarray
    evaporation: jnp.ndarray
    precipitation: jnp.ndarray
    eastward_wind_stress: jnp.ndarray
    northward_wind_stress: jnp.ndarray
    u0: jnp.ndarray
    v0: jnp.ndarray
    sea_total_heat_flux: jnp.ndarray
    sea_evaporation: jnp.ndarray
    sea_eastward_wind_stress: jnp.ndarray
    sea_northward_wind_stress: jnp.ndarray
    land_evaporation: jnp.ndarray
    downward_shortwave_flux: jnp.ndarray
    downward_longwave_flux: jnp.ndarray
    wind_speed: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray
    sea_air_temperature: jnp.ndarray
    sea_air_specific_humidity: jnp.ndarray
    rainfall: jnp.ndarray
    snowfall: jnp.ndarray


def sea_tile(exchange: Any, field: str) -> Any:
    """``field`` over the sea part of the cell, or ``None`` if not published.

    The sea part is open water and sea ice together: the ``sea`` tile where a
    package publishes one, else the fraction-weighted mean of its ``water``
    and ``sea_ice`` tiles (zero where the cell has no sea, which no ocean
    cell is). ``exchange`` is jax-gcm's
    :class:`~jcm.physics.surface.surface_exchange.SurfaceExchange`; tiles are
    selected by name with its ``tile`` method, never by an assumed index.
    """
    sea = exchange.tile(field, "sea")
    if sea is not None:
        return sea
    water, ice = exchange.tile(field, "water"), exchange.tile(field, "sea_ice")
    if water is None or ice is None:
        return None
    water_fraction = exchange.tile("tile_fraction", "water")
    ice_fraction = exchange.tile("tile_fraction", "sea_ice")
    sea_fraction = water_fraction + ice_fraction
    return jnp.where(
        sea_fraction > 0.0,
        (water_fraction * water + ice_fraction * ice)
        / jnp.where(sea_fraction > 0.0, sea_fraction, 1.0),
        0.0,
    )


def publishes_land_evaporation(exchange: Any) -> bool:
    """Whether the package publishes its evaporation over the land part.

    Read off the struct's static tile names and which tile fields are filled,
    like :func:`publishes_sea_tiles`.
    """
    return exchange.tile("evaporation", "land") is not None


def publishes_sea_tiles(exchange: Any) -> bool:
    """Whether every flux :func:`from_diagnostics` reads has a sea tile.

    Read off the struct's static tile names and which tile fields are filled,
    so a structural template (``Physics.get_empty_data``) answers it without
    a step being integrated.
    """
    return all(
        sea_tile(exchange, field) is not None
        for field in ("net_heat_flux", "evaporation", "stress_u", "stress_v")
    )


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

    def over_land(field: str) -> jnp.ndarray:
        # The grid mean stands in where the package publishes no land tile;
        # default_exchanges refuses to force a land component with it.
        land = exchange.tile(field, "land")
        values: jnp.ndarray = getattr(exchange, field) if land is None else land
        return values.reshape(nodal_shape)

    def over_sea(field: str) -> jnp.ndarray:
        # The grid mean stands in where the package publishes no sea tile
        # (see the module docstring); which case applies is static.
        sea = sea_tile(exchange, field)
        values: jnp.ndarray = getattr(exchange, field) if sea is None else sea
        return values.reshape(nodal_shape)

    def sea_air(field: str) -> jnp.ndarray:
        # The air the sea's fluxes were computed against: the sea tile where
        # the package publishes per-tile air (SPEEDY), else its one reference
        # air (ECHAM).
        sea = sea_tile(exchange, field)
        values: jnp.ndarray = getattr(exchange, field) if sea is None else sea
        return values.reshape(nodal_shape)

    rainfall, snowfall = split_precipitation(exchange)
    return SurfaceExchange(
        total_heat_flux=(-exchange.net_heat_flux).reshape(nodal_shape),
        evaporation=exchange.evaporation.reshape(nodal_shape),
        precipitation=exchange.precipitation.reshape(nodal_shape),
        eastward_wind_stress=exchange.stress_u.reshape(nodal_shape),
        northward_wind_stress=exchange.stress_v.reshape(nodal_shape),
        u0=exchange.wind_u.reshape(nodal_shape),
        v0=exchange.wind_v.reshape(nodal_shape),
        sea_total_heat_flux=-over_sea("net_heat_flux"),
        sea_evaporation=over_sea("evaporation"),
        sea_eastward_wind_stress=over_sea("stress_u"),
        sea_northward_wind_stress=over_sea("stress_v"),
        land_evaporation=over_land("evaporation"),
        downward_shortwave_flux=exchange.downward_shortwave_flux.reshape(nodal_shape),
        downward_longwave_flux=exchange.downward_longwave_flux.reshape(nodal_shape),
        wind_speed=exchange.wind_speed.reshape(nodal_shape),
        normalized_surface_pressure=(
            exchange.surface_pressure / jcm_constants.p0).reshape(nodal_shape),
        sea_air_temperature=sea_air("air_temperature"),
        sea_air_specific_humidity=sea_air("air_specific_humidity"),
        rainfall=rainfall.reshape(nodal_shape),
        snowfall=snowfall.reshape(nodal_shape),
    )
