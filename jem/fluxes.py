"""The surface exchange a declarative table cannot carry.

:data:`jem.exchangers.VEROS_OCEAN_EXCHANGES` is the copy-only half of the
coupling between a JCM atmosphere and a Veros ocean: every row it holds moves
a *scalar* field from one carry to another unchanged (bar a regrid). Two
things a Veros ocean needs cannot be table rows at all:

- **the surface wind stress** Veros integrates (``forcing.surface_taux`` /
  ``surface_tauy``) is a *vector*. The atmosphere publishes it
  (``derived.sea_eastward_wind_stress`` / ``sea_northward_wind_stress``,
  jax-gcm's own delivered surface stress over the sea part of the cell), but
  in true east/north components, while
  an ocean on a rotated-pole grid integrates it in that grid's own local
  x/y frame. Getting from one to the other mixes the two components
  (:func:`rotate_vector`), which a row moving one field at a time cannot
  express;
- **the "swamp" sea-ice mask** is not a field either -- it is a condition
  (has the surface reached the freezing point?) applied to two fields the
  table already knows how to move.

This module is where that computed half of the coupling lives, so that an
exchanger built from it is wiring -- which regridder, which grid file --
rather than physics: :func:`mask_fluxes_under_ice` and :func:`rotate_vector`
are pure functions with no notion of a carry, and :class:`VerosExchange` is
the thin :class:`~jem.base.component.Exchanger` that reads the atmosphere and
ocean carries, calls them, and writes the result back with ``.replace(...)``.

Why the ocean takes the atmosphere's stress rather than computing its own
--------------------------------------------------------------------------
The stress jax-gcm publishes is the one the atmosphere's lowest layer was
actually given that step ("published == delivered"; see
:mod:`jem.components.jcm.exchange_fields`), computed by the physics
package's own stability-dependent surface closure and averaged over the
coupling interval. Handing the ocean that stress conserves momentum across
the interface: what the atmosphere loses the ocean gains, up to the
conservative regrid. A drag law applied here to the published near-surface
wind would not: it would be a second, independent stress law, blind to
stability, and -- because SPEEDY publishes its lowest-level wind and ECHAM
a 10 m one -- a law whose answer changes with the composed physics package
for a reason that has nothing to do with the ocean. It would also be applied
to the *interval-mean* wind, and a quadratic drag law of a mean wind is
smaller than the mean of the instantaneous stress, which is what the
published field is. The two differed by 30-40 % in magnitude on a T21
SPEEDY aquaplanet (jax-esm#132), and over the open ocean of
``+configuration=veros-double-drake`` after 30 days the published stress is
about 1.6 times the old ``Cd = 1e-3`` law's (RMS), with the same sign
everywhere and a pattern correlation of 0.92. JAX-ESM therefore has one
stress law for every package: the atmosphere's.

Two properties of that stress are inherited rather than chosen here. It is
the stress over the **sea tile** of each atmosphere cell (open water and sea
ice together; ``derived.sea_eastward_wind_stress`` and
``sea_northward_wind_stress``), not the grid-box mean, so a coastal ocean cell
takes none of the land's drag; it is regridded weighted by the sea fraction.
Where a physics package publishes no sea tile the adapter falls back to the
grid-box mean (:mod:`jem.components.jcm.exchange_fields`). And it is the
stress against a surface **at rest**:
jax-gcm reserves ``ForcingData.ocean_u``/``ocean_v`` for a relative-wind
stress but does not yet apply them (jax-gcm#915), so JEM does not send the
ocean's surface current back to the atmosphere.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from jem import constants
from jem.base.component import Carry, CouplingTime
from jem.exchangers import regrid_weighted

logger = logging.getLogger(__name__)


def _identity(value: jax.Array) -> jax.Array:
    """Return ``value`` unchanged; the regridder for a single shared grid."""
    return value


def rotate_vector(
    u: jax.Array, v: jax.Array, cos_angle: jax.Array, sin_angle: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Rotate an (east, north) vector field into a grid's own local frame.

    A grid whose pole has been rigidly displaced from Earth's true pole (a
    "rotated" grid, such as the packaged ``RotatedGaussianLatLon``) has its
    own local x/y axes, which do not point true-east/true-north except along
    its equator. ``cos_angle``/``sin_angle`` are that grid's per-cell
    rotation angle -- the angle between true north and the grid's local
    y-axis -- and rotating an (east, north) vector by it gives the vector's
    components in the grid's own frame:

    ``x = cos(angle) * u + sin(angle) * v``
    ``y = -sin(angle) * u + cos(angle) * v``

    Parameters
    ----------
    u, v : jax.Array
        The vector's true-east and true-north components.
    cos_angle, sin_angle : jax.Array
        The grid's per-cell rotation angle, broadcastable against ``u``/``v``
        (see :func:`read_rotation_angles`).

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The vector's components in the grid's local (x, y) frame.

    """
    x = cos_angle * u + sin_angle * v
    y = -sin_angle * u + cos_angle * v
    return x, y


def mask_fluxes_under_ice(
    sea_surface_temperature: jax.Array,
    heat_flux: jax.Array,
    freshwater_flux: jax.Array,
    *,
    freezing_point: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Zero the fluxes that would cool an ocean already at the freezing point.

    A simple "swamp" sea-ice insulation, ported from
    ``veros/setups/global_1deg``: once the surface reaches the freezing
    point, sea ice forms and insulates it from further heat and freshwater
    exchange with the atmosphere, so both fluxes are masked to zero there.
    ``heat_flux`` is **upward-positive** in JAX-ESM's convention (positive
    means the ocean is losing heat), so a flux that would *warm* the ocean
    (melt the ice) is negative and is always let through regardless of
    temperature -- only a *cooling* flux at or below the freezing point is
    masked. ``freezing_point`` defaults to
    ``jem.constants.seawater_freezing_point_K``, the freezing point every slab
    component uses, read at call time so a ``set_constants`` override
    reaches it.

    This is a condition on two fields, not a component's state: it reads
    ``sea_surface_temperature`` and returns the masked ``(heat_flux,
    freshwater_flux)`` pair, and touches nothing else.

    Parameters
    ----------
    sea_surface_temperature : jax.Array
        The ocean's surface temperature, K.
    heat_flux, freshwater_flux : jax.Array
        The fluxes to mask, upward-positive, broadcastable against
        ``sea_surface_temperature``.
    freezing_point : float, optional
        The freezing point below which a cooling flux is masked, K. Default
        ``jem.constants.seawater_freezing_point_K``.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The masked ``(heat_flux, freshwater_flux)``.

    """
    if freezing_point is None:
        freezing_point = constants.seawater_freezing_point_K
    not_frozen = sea_surface_temperature > freezing_point
    would_warm = heat_flux < 0
    ice_free = jnp.logical_or(not_frozen, would_warm)
    return heat_flux * ice_free, freshwater_flux * ice_free


def read_rotation_angles(scrip_grid_file: str) -> tuple[jax.Array, jax.Array]:
    """Read a SCRIP grid's per-cell rotation angles, on the (n_lon, n_lat) layout.

    ``grid_cos_angle``/``grid_sin_angle`` are stored flat, ``(grid_size,)``,
    in SCRIP's lon-fastest order; this reshapes them to ``(n_lon, n_lat)``
    with ``order="F"`` using the file's own ``grid_dims`` -- the same layout
    :class:`jem.utils.esmf_regrid.ESMFRegridder` and
    :class:`jem.components.slab.grid.SlabGrid` both use, so the angles this
    returns line up index-for-index with a regridded wind-stress field with
    no transpose needed.

    Parameters
    ----------
    scrip_grid_file : str
        Path to a SCRIP grid file carrying ``grid_cos_angle``/
        ``grid_sin_angle`` (a rotated grid, such as the packaged
        ``RotatedGaussianLatLon.SCRIP.nc``). An unrotated grid's SCRIP file
        has no such variables, since its axes are already true east/north --
        pass ``rotation_grid_file=None`` to :class:`VerosExchange` for that
        case rather than pointing this at one.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(cos_angle, sin_angle)``, each ``(n_lon, n_lat)``.

    Raises
    ------
    KeyError
        If the file has no ``grid_cos_angle``/``grid_sin_angle`` variable.

    """
    grid = xr.open_dataset(scrip_grid_file)
    for name in ("grid_cos_angle", "grid_sin_angle"):
        if name not in grid:
            raise KeyError(
                f"{scrip_grid_file!r} has no {name!r} variable, so it cannot "
                "be used as VerosExchange's rotation_grid_file -- an "
                "unrotated grid's SCRIP file carries none, because its axes "
                "are already true east/north. Pass rotation_grid_file=None "
                "for that case."
            )
    n_lon, n_lat = (int(n) for n in grid["grid_dims"].to_numpy())
    cos_angle = jnp.asarray(
        grid["grid_cos_angle"].to_numpy().reshape((n_lon, n_lat), order="F")
    )
    sin_angle = jnp.asarray(
        grid["grid_sin_angle"].to_numpy().reshape((n_lon, n_lat), order="F")
    )
    return cos_angle, sin_angle


class VerosExchange:
    """The exchanger a JCM atmosphere and a Veros ocean are coupled through.

    Reproduces the ``atm``/``ocn`` rows of
    :data:`jem.exchangers.VEROS_OCEAN_EXCHANGES` -- the surface heat and
    freshwater fluxes over the sea part of each cell onto the ocean, the sea
    surface temperature back onto the atmosphere -- and adds the two things
    that table cannot express: the atmosphere's published surface wind stress
    over the sea part of the cell, regridded onto the ocean grid
    and rotated into its local frame when ``rotation_grid_file`` is given
    (see the module docstring for why the ocean takes that stress rather than
    one of its own), and :func:`mask_fluxes_under_ice` applied to the heat
    and freshwater fluxes. It is what makes a Veros configuration mechanically,
    as well as thermodynamically, forced. On another grid the fluxes and
    stress are weighted by the sea part of each atmosphere cell and the SST
    by Veros' own wet mask (:func:`jem.exchangers.regrid_weighted`), so
    neither side takes the other's land. With a land component present it
    also makes the table's land rows: the precipitation and the land tile's
    evaporation to the land, its soil water availability back to the
    atmosphere (list ``soilw_am`` in ``coupling.exchanged_forcing``, since
    this exchanger cannot be read off the way a table can), and its runoff to
    the ocean. It does not touch a sea-ice
    component: :data:`VEROS_OCEAN_EXCHANGES`'s ``seaice`` rows have no
    counterpart here, because the configurations this exchanger serves
    compose ``seaice=none``; a hand-written exchanger replaces the
    declarative table entirely, so a sea ice composed into one of them is
    stepped unforced (jax-esm#160).

    Parameters
    ----------
    regrid : Mapping[str, Callable] or None
        Whatever :func:`jem.runners.build_regridders` produced -- named
        regridders, plus the role aliases :data:`jem.runners.REGRID_ROLES`
        gives them. Two roles are used: ``"a2o_flux"`` for the two stress
        components and the two fluxes going onto the ocean grid (mapped
        **conservatively**, so a flux's budget -- momentum included --
        survives the interface; each is over the sea part of the
        atmosphere's cells, so it is weighted by that part,
        ``atm.derived.sea_fraction``, with
        :func:`jem.exchangers.regrid_weighted`), and ``"o2a_state"`` for the
        sea surface temperature coming back (mapped **bilinearly**, so it is
        not left with a conservative map's staircase -- see
        :mod:`jem.regrid` for the same reasoning applied to the declarative
        table), weighted by Veros' wet mask. ``"o2a_flux"`` (the
        conservative ocean-to-atmosphere map) backs it up: a bilinear map
        reads only the four nearest columns, so an atmosphere cell with a
        sliver of wet ocean can have none of them wet, and the SST is then
        taken from the conservative map, which covers every cell that
        overlaps a wet column (see :meth:`check_masks`). Either of
        ``"a2o_flux"``/``"o2a_state"`` missing from ``regrid`` -- and ``regrid=None``
        entirely -- falls back to the identity, which is exactly the
        single-grid double-drake configuration (``regrid=same_grid``
        composes to ``None``, which :func:`jem.runners.build_regridders`
        turns into an empty mapping).
    rotation_grid_file : str or None
        A SCRIP grid file to read :func:`read_rotation_angles` from, when the
        ocean grid's axes are not true east/north (a rotated-pole grid, such
        as the packaged ``RotatedGaussianLatLon``). ``None`` (the default)
        means no rotation -- the identity -- which is right for a grid whose
        axes already are true east/north (the double-drake configuration's
        uniform lat-lon ocean grid).
    freezing_point : float, optional
        Passed to :func:`mask_fluxes_under_ice`; ``None`` (the default) is
        ``jem.constants.seawater_freezing_point_K``, the freezing point the
        rest of the coupled model uses.

    Notes
    -----
    **The two masks must agree.** The fluxes are weighted by the atmosphere's
    sea part and the SST by Veros' wet mask, so each side takes only water
    from the other; but a flux the atmosphere computed over sea that has no
    wet column under it reaches no ocean, and an atmosphere cell the ocean
    cannot reach would be given a land fill value as its SST. The two
    coastlines therefore have to be one: the atmosphere's land fraction is
    derived from the ocean's wet mask (``jem.tools.veros_earth_terrain``,
    as ``+configuration=veros-earth`` does), and :meth:`check_masks` raises
    on the first coupled step if a cell of the atmosphere has sea but no wet
    ocean mapped under it.

    Every numeric tunable is a Python default on this constructor, never a
    YAML value -- a configuration names only ``_target_`` and, where needed,
    ``rotation_grid_file``; the CLI reaches the rest with, for example,
    ``+coupling.exchanger.freezing_point=271.2``. There is deliberately no
    drag coefficient: the stress is the atmosphere's, and tuning the ocean's
    momentum forcing means tuning the atmosphere's surface closure, so the
    two sides of the interface cannot disagree about it. The rotation-angle
    file is read and the regridder lookups are resolved once, here in
    ``__init__``; :meth:`__call__` runs inside the traced coupled step and
    does no I/O.

    The destination-dtype cast this exchanger applies (see the build-order
    comment in :meth:`__call__`) is not unique to this hand-written class:
    the declarative :class:`jem.exchangers.Exchange` table applies the same
    cast in its own ``__call__``, so an ``ocean=veros`` configuration that
    composes the default table instead of this exchanger gets the same fix.

    """

    def __init__(
        self,
        *,
        regrid: Mapping[str, Callable[[Any], Any]] | None = None,
        rotation_grid_file: str | None = None,
        freezing_point: float | None = None,
    ) -> None:
        """Build the exchanger; see the class docstring for the parameters."""
        regrid = regrid or {}
        self._a2o_flux: Callable[[Any], Any] = regrid.get("a2o_flux", _identity)
        self._o2a_state: Callable[[Any], Any] = regrid.get("o2a_state", _identity)
        # Conservative map used wherever the bilinear one reaches no wet
        # column; absent, the bilinear map is also the one coverage is
        # judged by.
        self._o2a_cover: Callable[[Any], Any] = regrid.get(
            "o2a_flux", self._o2a_state)
        self.rotation_grid_file = rotation_grid_file
        self._rotation_angles: tuple[jax.Array, jax.Array] | None = (
            None if rotation_grid_file is None
            else read_rotation_angles(rotation_grid_file)
        )
        self.freezing_point = freezing_point
        logger.debug("Built %r", self)

    def __call__(
        self, components: dict[str, Carry], time: CouplingTime
    ) -> dict[str, Carry]:
        """Apply the wind stress, the ice mask and the rest of the coupling.

        Parameters
        ----------
        components : dict[str, jem.base.component.Carry]
            Every component's carry, as the coupler hands it over. Only
            ``"atm"`` and ``"ocn"`` are read or written.
        time : jem.base.component.CouplingTime
            The coupler's clock. Only ``time.step`` is read (to check the
            masks once, on step 0); the coupling does not depend on the
            date. Part of the :data:`jem.base.component.Exchanger`
            signature every exchanger takes.

        Returns
        -------
        dict[str, jem.base.component.Carry]
            ``components`` with ``"atm"`` and ``"ocn"`` replaced; every other
            entry is passed through unchanged. Nothing is mutated in place,
            and every source is read from the incoming carries before
            anything is written, so the result does not depend on the order
            the fields happen to be touched in (the same rule
            :meth:`jem.exchangers.Exchange.__call__` states).

        """
        # The coupling does not depend on the date; only the step count is
        # read, to run the mask check once, on a run's first step.

        atm = components["atm"]
        ocn = components["ocn"]

        # Read every source first, before any carry is replaced. The target
        # dtypes are read here too, from the untouched forcing sections:
        # Veros runs double precision internally (importing its JAX backend
        # flips `jax.config.jax_enable_x64` to True process-wide as a side
        # effect the first time `veros.core` is imported -- see
        # `jem.components.veros_component.configure_veros_runtime`), and that
        # flip lands wherever `build_coupler` happens to be when it fires: the
        # ocean's own carry is built afterward and so is entirely float64,
        # while the atmosphere's carry is *mixed* -- whatever jax-gcm had
        # already allocated at `Model` construction (built first) stays
        # float32, and anything allocated after the flip (every per-step
        # diagnostic, including `derived.sea_eastward_wind_stress` and
        # `derived.sea_total_heat_flux`)
        # comes out float64 too. So which atmosphere fields are float32
        # depends on build order, not on any promise this module makes.
        # `jax.lax.scan` requires a step's output carry to match its input
        # dtype exactly regardless, so a value written across the atm/ocn
        # boundary without a matching cast breaks the *very first* coupled
        # step with an opaque dtype-mismatch error from deep inside
        # `Coupler.generate_trajectory_function` -- not a physics bug, but
        # one this exchanger is the right place to close, since it is the one
        # place a value is known to cross the boundary, and reading the
        # destination's dtype at trace time (rather than assuming one) is
        # what makes the coupling robust to that ordering.
        # The fluxes over the sea part of each cell, not the cell means: on a
        # partly-land cell the land part's flux and drag stay with the
        # atmosphere's own land (jem.components.jcm.exchange_fields).
        eastward_wind_stress = atm["derived"].sea_eastward_wind_stress
        northward_wind_stress = atm["derived"].sea_northward_wind_stress
        total_heat_flux = atm["derived"].sea_total_heat_flux
        total_freshwater_flux = atm["derived"].sea_total_freshwater_flux
        sea_fraction = atm["derived"].sea_fraction
        ocean_sea_surface_temperature = ocn["derived"].sea_surface_temperature
        ocean_forcing_dtypes = {
            "surface_taux": ocn["forcing"].surface_taux.dtype,
            "surface_tauy": ocn["forcing"].surface_tauy.dtype,
            "heat_flux": ocn["forcing"].heat_flux.dtype,
            "freshwater_flux": ocn["forcing"].freshwater_flux.dtype,
        }
        atmosphere_sea_surface_temperature_dtype = (
            atm["forcing"].sea_surface_temperature.dtype
        )

        # Wind stress: regrid the atmosphere's stress onto the ocean grid,
        # then rotate it into that grid's local frame if it has one. The
        # stress is a flux of momentum, so it goes through the conservative
        # a2o map like the heat and freshwater fluxes, which keeps the
        # area-integrated momentum the atmosphere lost over the sea equal to
        # what the ocean receives, to the extent the two grids agree on where
        # the sea is. Like them it is weighted by the sea part of each
        # atmosphere cell (`_a2o_sea`), so a coastal ocean cell is not handed
        # the drag of an atmosphere cell with no sea in it. Regridding *before* rotating (never the other way
        # around) is deliberate: JCM's own grid is unrotated, so the two
        # components are true east/north everywhere on it, which is what makes
        # a component-wise conservative regrid of each of them well defined
        # (there is no single frame change that could be "moved before" the
        # regrid to simplify this). The rotation angles, by contrast, are
        # defined per *ocean* cell (`read_rotation_angles` reads them off the
        # ocean's own SCRIP file), so they only make sense to apply once the
        # stress is already sitting on that grid. The rotation re-expresses
        # the same vector in the ocean's own frame, so it does not change the
        # momentum delivered either.
        surface_taux = self._a2o_sea(eastward_wind_stress, sea_fraction)
        surface_tauy = self._a2o_sea(northward_wind_stress, sea_fraction)
        if self._rotation_angles is not None:
            cos_angle, sin_angle = self._rotation_angles
            surface_taux, surface_tauy = rotate_vector(
                surface_taux, surface_tauy, cos_angle, sin_angle)

        # Heat and freshwater fluxes: regrid onto the ocean grid, then mask
        # wherever the ocean the flux is destined for has reached the
        # freezing point.
        heat_flux, freshwater_flux = mask_fluxes_under_ice(
            ocean_sea_surface_temperature,
            self._a2o_sea(total_heat_flux, sea_fraction),
            self._a2o_sea(total_freshwater_flux, sea_fraction),
            freezing_point=self.freezing_point,
        )

        # Sea surface temperature: regrid back onto the atmosphere's grid.
        # Weighted by Veros' wet mask, so an atmosphere cell on the coast
        # takes the SST of the water under it, not the land columns' fill.
        # An atmosphere cell that no wet column reaches takes the unweighted
        # map of the SST field, which over land is Veros' fill value
        # (273.15 K) (`jem.exchangers.regrid_weighted`); `_o2a_sea` makes
        # that impossible for a cell with sea in it, and `check_masks` raises
        # if the terrain's coastline is not the ocean's.
        wet = ocn["derived"].sea_fraction
        sea_surface_temperature_on_atm = self._o2a_sea(
            ocean_sea_surface_temperature, wet
        )
        self._guard_masks(time.step, sea_fraction, wet)

        ocn = dict(ocn, forcing=ocn["forcing"].replace(
            surface_taux=surface_taux.astype(ocean_forcing_dtypes["surface_taux"]),
            surface_tauy=surface_tauy.astype(ocean_forcing_dtypes["surface_tauy"]),
            heat_flux=heat_flux.astype(ocean_forcing_dtypes["heat_flux"]),
            freshwater_flux=freshwater_flux.astype(
                ocean_forcing_dtypes["freshwater_flux"]
            ),
        ))
        sea_surface_temperature_on_atm = sea_surface_temperature_on_atm.astype(
            atmosphere_sea_surface_temperature_dtype
        )
        atm = dict(atm, forcing=atm["forcing"].replace(
            sea_surface_temperature=sea_surface_temperature_on_atm,
        ))
        exchanged = dict(components, atm=atm, ocn=ocn)
        if "lnd" in components:
            exchanged = self._exchange_with_land(components, exchanged)
        return exchanged

    @staticmethod
    def _exchange_with_land(
        incoming: dict[str, Carry], exchanged: dict[str, Carry]
    ) -> dict[str, Carry]:
        """Apply the land rows of :data:`jem.exchangers.VEROS_OCEAN_EXCHANGES`.

        The same copies the declarative table makes, all on the atmosphere's
        grid: the precipitation and the land tile's evaporation to the land,
        its soil water availability back to the atmosphere, and its runoff,
        as a mass rate over its whole grid, to the ocean, which spreads it
        over its own wet surface. Sources are read from ``incoming``, so the
        result does not depend on what was written before.
        """
        atm, lnd = incoming["atm"], incoming["lnd"]

        def like(value, current):
            return jnp.asarray(value, dtype=current.dtype)

        land_forcing = lnd["forcing"]
        lnd = dict(lnd, forcing=land_forcing.replace(
            precipitation=like(atm["derived"].precipitation,
                               land_forcing.precipitation),
            evaporation=like(atm["derived"].land_evaporation,
                             land_forcing.evaporation),
        ))
        atm_out, ocn_out = exchanged["atm"], exchanged["ocn"]
        atm_out = dict(atm_out, forcing=atm_out["forcing"].replace(
            soilw_am=like(incoming["lnd"]["derived"].soil_water_availability,
                          atm_out["forcing"].soilw_am),
        ))
        ocn_out = dict(ocn_out, forcing=ocn_out["forcing"].replace(
            runoff_mass_rate=like(incoming["lnd"]["derived"].runoff_mass_rate,
                                  ocn_out["forcing"].runoff_mass_rate),
        ))
        return dict(exchanged, atm=atm_out, ocn=ocn_out, lnd=lnd)

    def _a2o_sea(self, field: jax.Array, sea_fraction: jax.Array) -> jax.Array:
        """Map a field over the sea part of the atmosphere's cells onto the ocean.

        Weighted by that part (:func:`jem.exchangers.regrid_weighted`), so a
        coastal ocean cell takes nothing from an atmosphere cell with no sea
        in it. On one grid every ocean cell is its own atmosphere cell, so
        the field passes through unchanged.
        """
        if self._a2o_flux is _identity:
            return field
        mapped: jax.Array = regrid_weighted(self._a2o_flux, field, sea_fraction)
        return mapped

    def _o2a_sea(self, field: jax.Array, wet: jax.Array) -> jax.Array:
        """Map a field over Veros' wet columns onto the atmosphere's grid.

        Weighted by Veros' wet mask (``derived.sea_fraction``); on one grid
        the field passes through unchanged. Where the state map (bilinear)
        reaches no wet column the conservative map stands in, weighted the
        same way: it covers every atmosphere cell that overlaps wet ocean, so
        a cell with sea in it never takes the unweighted field over land.
        """
        if self._o2a_state is _identity:
            return field
        mapped: jax.Array = regrid_weighted(self._o2a_state, field, wet)
        if self._o2a_cover is self._o2a_state:
            return mapped
        covered_by_state = self._o2a_state(wet) > 0.0
        backup: jax.Array = regrid_weighted(self._o2a_cover, field, wet)
        return jnp.where(covered_by_state, mapped, backup)

    def _guard_masks(
        self, step: jax.Array, sea_fraction: jax.Array, wet: jax.Array
    ) -> None:
        """Run :meth:`check_masks` on a run's first coupling step.

        The masks live in the carries, so inside a traced step they are not
        values the exchanger can read at construction; a host callback
        receives them concretely instead. It sits under a ``lax.cond`` on
        ``step == 0`` so that it costs one device-to-host copy per run, not
        one per step, and it only reads: nothing flows back into the step,
        so it changes neither the carry nor a gradient. A failing check
        surfaces as the error ``jax`` raises for a failed callback, whose
        message is :meth:`check_masks`'s. On one grid the maps are identity
        and the masks are one array: nothing to check.
        """
        if self._o2a_state is _identity:
            return
        reached = self._o2a_cover(wet)
        jax.lax.cond(
            jnp.asarray(step) == 0,
            lambda: jax.debug.callback(self.check_masks, sea_fraction, reached),
            lambda: None,
        )

    @staticmethod
    def check_masks(sea_fraction: Any, reached: Any) -> None:
        """Raise if an atmosphere cell has sea that no wet ocean column reaches.

        Parameters
        ----------
        sea_fraction : array-like
            The atmosphere's sea fraction per cell (``1 - fmask`` of its
            terrain), on the atmosphere's grid.
        reached : array-like
            Veros' wet mask mapped onto the atmosphere's grid: positive
            where some wet column overlaps (or, for a bilinear map, is
            sampled by) the cell.

        Raises
        ------
        ValueError
            If any cell has ``sea_fraction > 0`` and ``reached <= 0``. Such a
            cell is a disagreement between the atmosphere's coastline and the
            ocean's: its sea flux reaches no water and it would take a land
            fill value as its SST. The message gives the count and the
            remedy.

        Notes
        -----
        Only this direction is checked because it is the one a mismatch can
        leave unserved: a wet ocean column always overlaps some atmosphere
        cell with sea in it once the atmosphere's sea is the wet area under
        it, and a column whose overlapping cells carry little sea still takes
        their normalised flux (:func:`jem.exchangers.regrid_weighted`).

        """
        sea = np.asarray(sea_fraction)
        unreached = (sea > 0.0) & (np.asarray(reached) <= 0.0)
        count = int(unreached.sum())
        if count:
            raise ValueError(
                f"{count} atmosphere cell(s) have sea (sea fraction "
                f"{float(sea[unreached].min()):.2f} to "
                f"{float(sea[unreached].max()):.2f}) but no wet ocean column "
                "mapped under them, so their sea flux reaches no ocean and "
                "their sea surface temperature has no water to come from: "
                "the atmosphere's land-sea mask and the ocean's wet mask "
                "disagree. Run the atmosphere on a terrain file derived from "
                "the ocean's mask "
                "(jem.tools.veros_earth_terrain.veros_earth_terrain; "
                "+configuration=veros-earth does)."
            )

    def __repr__(self) -> str:
        """Name the regridders this exchange holds and whether it rotates."""
        def name(regridder: Callable[[Any], Any]) -> str:
            if regridder is _identity:
                return "identity"
            return getattr(regridder, "__name__", repr(regridder))

        a2o = name(self._a2o_flux)
        o2a = name(self._o2a_state)
        rotates = (
            "no" if self._rotation_angles is None
            else f"yes ({self.rotation_grid_file})"
        )
        return (
            f"{type(self).__name__}(a2o_flux={a2o}, o2a_state={o2a}, "
            f"rotates={rotates}, freezing_point={self.freezing_point})"
        )
