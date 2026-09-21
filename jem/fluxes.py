"""The surface exchange a declarative table cannot carry.

:data:`jem.exchangers.VEROS_OCEAN_EXCHANGES` is the copy-only half of the
coupling between a JCM atmosphere and a Veros ocean: every row it holds moves
a field from one carry to another unchanged (bar a regrid). Two things a
Veros ocean needs are not copies of anything the atmosphere publishes, and so
cannot be table rows at all:

- **the surface wind stress** Veros integrates (``forcing.surface_taux`` /
  ``surface_tauy``) is not a field the atmosphere has -- it publishes a
  near-surface *wind* (``derived.u0`` / ``v0``), and turning a wind into a
  stress is a bulk drag law, optionally followed by a rotation into the
  ocean grid's own frame;
- **the "swamp" sea-ice mask** is not a field either -- it is a condition
  (has the surface reached the freezing point?) applied to two fields the
  table already knows how to move.

This module is where that computed half of the coupling lives, so that an
exchanger built from it is wiring -- which regridder, which grid file --
rather than physics: :func:`bulk_wind_stress`, :func:`mask_fluxes_under_ice`
and :func:`rotate_vector` are pure functions with no notion of a carry, and
:class:`VerosExchange` is the thin :class:`~jem.base.component.Exchanger`
that reads the atmosphere and ocean carries, calls them, and writes the
result back with ``.replace(...)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import xarray as xr

from jem.base.component import Carry, CouplingTime

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


def bulk_wind_stress(
    u: jax.Array,
    v: jax.Array,
    *,
    drag_coefficient: float = 1e-3,
    air_density: float = 1.22,
    min_speed: float = 1e-3,
) -> tuple[jax.Array, jax.Array]:
    """Return the surface wind stress a bulk drag law gives for a near-surface wind.

    ``tau = drag_coefficient * air_density * |wind| * wind``, applied
    component-wise. ``|wind|`` is computed as
    ``sqrt(max(u**2 + v**2, min_speed**2))``: the floor is on the **squared**
    speed, before the square root, rather than on ``|wind|`` itself, because
    ``d sqrt(x)/dx`` is unbounded as ``x -> 0`` -- automatic differentiation
    through ``sqrt(u**2 + v**2)`` at zero wind gives ``NaN`` even though the
    primal value (zero) is perfectly finite. Flooring the argument of the
    square root keeps both the value and its derivative bounded, at the cost
    of a wind speed that never reads below ``min_speed``. This mirrors the
    identical fix :class:`jem.components.veros_component.VerosComponent`
    already applies to Veros' own ``forc_tke_surface``.

    Parameters
    ----------
    u, v : jax.Array
        The near-surface wind's components, in the frame the ocean's stress
        is to be given in (rotate first with :func:`rotate_vector` if the
        ocean grid is not true-east/true-north).
    drag_coefficient : float
        Dimensionless bulk drag coefficient. Default ``1e-3``.
    air_density : float
        Near-surface air density, kg/m^3. Default ``1.22``.
    min_speed : float
        The wind speed floor described above, m/s. Default ``1e-3``.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The wind stress's components, in the same frame as ``u``/``v``.

    Notes
    -----
    This bulk law is a deliberately independent computation, not a lookup of
    what SPEEDY itself already computed. JCM's own surface scheme separately
    derives a sea-surface stress (``jcm.physics.surface.speedy_surface_flux``,
    published as ``SurfaceTypeFluxes.ustr``/``vstr``, e.g.
    ``ustr = -sfp.cds * rho_wind * air.u_bottom`` with a stability-corrected
    ``rho_wind``), and this function's ``drag_coefficient=1e-3``,
    ``air_density=1.22`` law over the same near-surface wind will not agree
    with it in general. Momentum is therefore *not* conserved between the
    atmosphere and the ocean across this exchange -- this is faithful to the
    original (pre-package) double-drake/earth drivers, which computed the
    ocean's wind stress this same independent way rather than reusing
    SPEEDY's.

    """
    speed = jnp.sqrt(jnp.maximum(u**2 + v**2, min_speed**2))
    scale = drag_coefficient * air_density * speed
    return scale * u, scale * v


def mask_fluxes_under_ice(
    sea_surface_temperature: jax.Array,
    heat_flux: jax.Array,
    freshwater_flux: jax.Array,
    *,
    freezing_point: float = 271.35,
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
    masked. ``freezing_point`` defaults to ``271.35`` K, i.e. ``273.15 -
    1.8``, the freezing point of seawater at typical salinity.

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
    freezing_point : float
        The freezing point below which a cooling flux is masked, K. Default
        ``271.35`` (``273.15 - 1.8``).

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The masked ``(heat_flux, freshwater_flux)``.

    """
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
    returns line up index-for-index with a regridded ``u0``/``v0`` field with
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
    freshwater fluxes onto the ocean, the sea surface temperature back onto
    the atmosphere -- and adds the two things that table cannot express: a
    :func:`bulk_wind_stress` computed from the atmosphere's near-surface wind
    (rotated into the ocean grid's frame first, when ``rotation_grid_file``
    is given), and :func:`mask_fluxes_under_ice` applied to the heat and
    freshwater fluxes. It is what makes a Veros configuration mechanically,
    as well as thermodynamically, forced. It does not touch a land or
    sea-ice component -- :data:`VEROS_OCEAN_EXCHANGES`'s ``lnd``/``seaice``
    rows have no counterpart here, because the configurations this exchanger
    serves compose ``land=none``/``seaice=none``.

    Parameters
    ----------
    regrid : Mapping[str, Callable] or None
        Whatever :func:`jem.runners.build_regridders` produced -- named
        regridders, plus the role aliases :data:`jem.runners.REGRID_ROLES`
        gives them. Two roles are used: ``"a2o_flux"`` for the wind and the
        two fluxes going onto the ocean grid (mapped **conservatively**, so a
        flux's budget survives the interface), and ``"o2a_state"`` for the
        sea surface temperature coming back (mapped **bilinearly**, so it is
        not left with a conservative map's staircase -- see
        :mod:`jem.regrid` for the same reasoning applied to the declarative
        table). Either role missing from ``regrid`` -- and ``regrid=None``
        entirely -- falls back to the identity, which is exactly the
        single-grid double-drake configuration (``regrid=same_grid``
        composes to an empty mapping).
    rotation_grid_file : str or None
        A SCRIP grid file to read :func:`read_rotation_angles` from, when the
        ocean grid's axes are not true east/north (a rotated-pole grid, such
        as the packaged ``RotatedGaussianLatLon``). ``None`` (the default)
        means no rotation -- the identity -- which is right for a grid whose
        axes already are true east/north (the double-drake configuration's
        uniform lat-lon ocean grid).
    drag_coefficient, air_density, min_speed : float
        Passed to :func:`bulk_wind_stress`.
    freezing_point : float
        Passed to :func:`mask_fluxes_under_ice`.

    Notes
    -----
    Every numeric tunable is a Python default on this constructor, never a
    YAML value -- a configuration names only ``_target_`` and, where needed,
    ``rotation_grid_file``; the CLI reaches the rest with, for example,
    ``+coupling.exchanger.drag_coefficient=2e-3``. The rotation-angle file is
    read and the regridder lookups are resolved once, here in ``__init__``;
    :meth:`__call__` runs inside the traced coupled step and does no I/O.

    The destination-dtype cast this exchanger applies (see the build-order
    comment in :meth:`__call__`) is not unique to this hand-written class:
    the declarative :class:`jem.exchangers.Exchange` table is gaining the
    same cast (in the sibling branch that owns ``jem/exchangers.py``), so an
    ``ocean=veros`` configuration that composes the default table instead of
    this exchanger steps through the same fix.

    """

    def __init__(
        self,
        *,
        regrid: Mapping[str, Callable[[Any], Any]] | None = None,
        rotation_grid_file: str | None = None,
        drag_coefficient: float = 1e-3,
        air_density: float = 1.22,
        min_speed: float = 1e-3,
        freezing_point: float = 271.35,
    ) -> None:
        """Build the exchanger; see the class docstring for the parameters."""
        regrid = regrid or {}
        self._a2o_flux: Callable[[Any], Any] = regrid.get("a2o_flux", _identity)
        self._o2a_state: Callable[[Any], Any] = regrid.get("o2a_state", _identity)
        self.rotation_grid_file = rotation_grid_file
        self._rotation_angles: tuple[jax.Array, jax.Array] | None = (
            None if rotation_grid_file is None
            else read_rotation_angles(rotation_grid_file)
        )
        self.drag_coefficient = drag_coefficient
        self.air_density = air_density
        self.min_speed = min_speed
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
            The coupler's clock. Unused -- this coupling does not depend on
            the date -- but part of the :data:`jem.base.component.Exchanger`
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
        del time  # this coupling does not depend on the date

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
        # diagnostic, including `derived.u0` and `derived.total_heat_flux`)
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
        u0 = atm["derived"].u0
        v0 = atm["derived"].v0
        total_heat_flux = atm["derived"].total_heat_flux
        total_freshwater_flux = atm["derived"].total_freshwater_flux
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

        # Wind stress: regrid the wind onto the ocean grid, rotate into its
        # local frame if it has one, then apply the bulk drag law. Regridding
        # *before* rotating (never the other way around) is deliberate: JCM's
        # own grid is unrotated, so `u0`/`v0` are true east/north everywhere
        # on it, which is what makes a component-wise conservative regrid of
        # each of them well defined (there is no single frame change that
        # could be "moved before" the regrid to simplify this). The rotation
        # angles, by contrast, are defined per *ocean* cell
        # (`read_rotation_angles` reads them off the ocean's own SCRIP file),
        # so they only make sense to apply once the wind is already sitting
        # on that grid.
        wind_x = self._a2o_flux(u0)
        wind_y = self._a2o_flux(v0)
        if self._rotation_angles is not None:
            cos_angle, sin_angle = self._rotation_angles
            wind_x, wind_y = rotate_vector(wind_x, wind_y, cos_angle, sin_angle)
        surface_taux, surface_tauy = bulk_wind_stress(
            wind_x, wind_y,
            drag_coefficient=self.drag_coefficient,
            air_density=self.air_density,
            min_speed=self.min_speed,
        )

        # Heat and freshwater fluxes: regrid onto the ocean grid, then mask
        # wherever the ocean the flux is destined for has reached the
        # freezing point.
        heat_flux, freshwater_flux = mask_fluxes_under_ice(
            ocean_sea_surface_temperature,
            self._a2o_flux(total_heat_flux),
            self._a2o_flux(total_freshwater_flux),
            freezing_point=self.freezing_point,
        )

        # Sea surface temperature: regrid back onto the atmosphere's grid.
        sea_surface_temperature_on_atm = self._o2a_state(
            ocean_sea_surface_temperature
        )

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
        return dict(components, atm=atm, ocn=ocn)

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
            f"rotates={rotates}, drag_coefficient={self.drag_coefficient}, "
            f"air_density={self.air_density}, min_speed={self.min_speed}, "
            f"freezing_point={self.freezing_point})"
        )
