"""Bucket land: SPEEDY's soil water as a two-layer leaky bucket.

The land's *temperature* is not here. It is the atmosphere's own: jax-gcm
integrates SPEEDY's slab land (``land_model.f90``) inside its surface-flux
term, forced by the flux over the land part of each cell. What SPEEDY does
not evolve is its soil water: its soil water availability ``soilw_am`` is a
prescribed climatology. This component evolves the soil water instead, from
the precipitation and the evaporation over the land part of each cell, and
hands the atmosphere the availability it diagnoses, so the land's
evaporation responds to its own wetting and drying.

The model was first written by Tien-Yiao Hsu as the soil-water part of
JAX-ESM's former thermal slab land (jax-esm#154); the thermal part is gone,
since it duplicated SPEEDY's own.
"""

import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jcm.constants as jcm_constants
import numpy as np
import tree_math

from jem.base.component import Carry, CouplingTime, Diagnostics
from jem.components.slab.base import (
    SlabModelBase,
    forcing_variable,
    load_monthly_climatology,
    role_attrs,
)
from jem.components.slab.bucket_land_model.params import BucketLandParameters
from jem.components.slab.grid import SlabGrid
from jem.utils.cycles import evaluate_cyclic_linear

logger = logging.getLogger(__name__)

#: Density of liquid water (kg m-3): converts a water flux (kg m-2 s-1) into a
#: depth of water per second (m s-1).
WATER_DENSITY = 1000.0

#: The variable a soil-water-availability climatology is read from -- the
#: name jax-gcm's packaged boundary conditions give SPEEDY's ``soilw_am``.
SOIL_WATER_AVAILABILITY_VARIABLE = "soilw_am"


@tree_math.struct
class BucketLandState:
    """The water the soil holds.

    Attributes
    ----------
    soil_water_content : (n_lon, n_lat, 2) array
        Volumetric water content (m3 m-3) of the surface and the deep layer.
        Zero where the cell has no land.

    """

    soil_water_content: jnp.ndarray


@tree_math.struct
class BucketLandForcing:
    """What the atmosphere hands the land, filled by an exchanger.

    Attributes
    ----------
    precipitation : (n_lon, n_lat) array
        Precipitation (kg m-2 s-1, positive down). One value per cell: the
        atmosphere's precipitation falls on land and sea alike.
    evaporation : (n_lon, n_lat) array
        Evaporation over the land part of the cell (kg m-2 s-1, positive up;
        negative is dew) -- the atmosphere's land tile, not the grid-cell
        mean, which would charge the soil with the sea's evaporation.

    """

    precipitation: jnp.ndarray
    evaporation: jnp.ndarray

    @classmethod
    def zeros(cls, shape):
        return cls(jnp.zeros(shape), jnp.zeros(shape))


@tree_math.struct
class BucketLandDerived:
    """What the land diagnoses each step, for the atmosphere, the ocean and output.

    Every flux is per unit of land area (kg m-2 s-1) and zero where the cell
    has no land.

    Attributes
    ----------
    soil_water_availability : (n_lon, n_lat) array
        SPEEDY's ``soilw_am``: the water the two layers can give, as a
        fraction of what they hold between wilting point and field capacity,
        ``(d1 max(W1 - wp1, 0) + veg d2 max(W2 - wp2, 0)) / (d1 (fc1 - wp1) +
        d2 (fc2 - wp2))``, clipped to [0, 1]. The deep layer counts only
        through vegetation, whose roots reach it.
    deep_drainage : (n_lon, n_lat) array
        Water draining out of the bottom of the deep layer, leaving the model.
    soil_water_overflow : (n_lon, n_lat) array
        Water above field capacity, summed over the two layers, leaving the
        model.
    runoff : (n_lon, n_lat) array
        ``deep_drainage + soil_water_overflow``: all the water that left the
        land this step.
    unmet_evaporation : (n_lon, n_lat) array
        The part of ``forcing.evaporation`` the soil could not give without
        going below its wilting points (never negative). The atmosphere
        evaporated it anyway, so it is water the coupled system created; it
        is reported rather than clipped away, so the land's own budget still
        closes and the gap is visible.
    runoff_mass_rate : () array
        The runoff over the whole grid (kg s-1): ``sum(area * land_fraction *
        runoff)``. An ocean that takes it spreads it over its own wet area.

    """

    soil_water_availability: jnp.ndarray
    deep_drainage: jnp.ndarray
    soil_water_overflow: jnp.ndarray
    runoff: jnp.ndarray
    unmet_evaporation: jnp.ndarray
    runoff_mass_rate: jnp.ndarray

    @classmethod
    def zeros(cls, shape, **overrides):
        """Zero-filled derived fields, with ``overrides`` in place."""
        fields = {
            name: overrides.get(name, jnp.zeros(shape))
            for name in ("soil_water_availability", "deep_drainage",
                         "soil_water_overflow", "runoff", "unmet_evaporation")
        }
        return cls(**fields, runoff_mass_rate=overrides.get(
            "runoff_mass_rate", jnp.zeros(())))


def soil_water_availability(
    soil_water_content: jnp.ndarray, params: BucketLandParameters
) -> jnp.ndarray:
    """SPEEDY's soil water availability ``soilw_am`` of a two-layer soil.

    ``(d1 max(W1 - wp1, 0) + veg d2 max(W2 - wp2, 0)) / (d1 (fc1 - wp1) +
    d2 (fc2 - wp2))``, clipped to [0, 1]: with SPEEDY's depths
    (``d2 = idep2 d1``), wilting points and field capacity this is the
    availability its forcing preprocessor computes from the soil water
    climatology. ``soil_water_content`` is ``(..., 2)``.
    """
    available = _available_water(soil_water_content, params)
    return jnp.clip(jnp.sum(available, axis=-1) / _capacity(params), 0.0, 1.0)


def _available_water(soil_water_content, params):
    """Depth (m) of water each layer can give: ``(..., 2)``."""
    depth = params.soil_bucket_depth
    wilting = params.soil_water_wilting_point
    above_wilting = jnp.maximum(soil_water_content - wilting, 0.0)
    return jnp.stack(
        [depth[0] * above_wilting[..., 0],
         params.vegetation_fraction * depth[1] * above_wilting[..., 1]],
        axis=-1,
    )


def _capacity(params):
    """Depth (m) of water the two layers hold between wilting point and capacity."""
    depth = params.soil_bucket_depth
    return jnp.sum(
        depth * (params.soil_water_field_capacity - params.soil_water_wilting_point))


class BucketLandModel(SlabModelBase):
    """Two-layer leaky bucket of soil water under the atmosphere's own land.

    Only cells with some land (``grid.land_mask``) are integrated; the others
    hold no water and report zeros. Each layer ``k`` holds a volumetric water
    content ``W_k`` over a depth ``d_k``, between its wilting point ``wp_k``
    and field capacity ``fc_k``. A coupling step of length ``dt`` applies, in
    turn:

    1. **Precipitation** ``P`` into the surface layer.
    2. **Evaporation** ``E`` (the atmosphere's, over the land part of the
       cell), shared between the layers in proportion to the water each can
       give -- ``d1 max(W1 - wp1, 0)`` and ``veg d2 max(W2 - wp2, 0)``, the
       terms of the availability -- and all from the surface layer when
       neither has any. Dew (``E < 0``) all goes into the surface layer. No
       layer is taken below its wilting point; what it cannot give is
       ``derived.unmet_evaporation``.
    3. **Drainage** of each layer's water above its wilting point on its
       ``drainage_timescale``: the surface layer into the deep one, the deep
       layer out of the model (``derived.deep_drainage``). It is stepped
       implicitly in the drained water, ``e_new = e / (1 + dt / tau)``, so it
       is stable and never drains a layer below its wilting point, whatever
       the ratio of the coupling step to the time scale.
    4. **Overflow**: each layer is capped at its field capacity, and the
       excess leaves the model (``derived.soil_water_overflow``).

    Water is conserved to rounding: over a step, per unit of land area,
    ``rho_w * sum(d * dW) / dt = P - (E - unmet) - deep_drainage - overflow``.

    The availability diagnosed from the soil water at the end of the step is
    ``derived.soil_water_availability``, which the standard coupling hands
    the atmosphere as ``soilw_am``.

    Initial condition
    -----------------
    With a soil-water-availability climatology (``soil_water_clim_file``,
    variable ``soilw_am``) the soil starts from the availability observed in
    the month the run starts in: both layers are filled to the same fraction
    of their range, the one that gives that availability (or field capacity,
    where vegetation keeps the deep layer from reaching it). Without one,
    both layers start at ``params.initial_soil_water_fill`` of their range.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: BucketLandParameters | None = None,
        *,
        name: str = "lnd",
        soil_water_clim_file: str | None = None,
    ):
        """Initialize the bucket land.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid, the atmosphere's: the land takes the
            atmosphere's fluxes and hands back its availability without
            regridding. It has to carry cell areas
            (``SlabGrid.cell_solid_angle``) for the runoff's mass rate.
        params : BucketLandParameters, optional
            Tunable parameters; defaults to :meth:`BucketLandParameters.default`.
        name : str
            Component name in the coupler's workflow and carry. The default is
            the name the standard coupling wires a land under.
        soil_water_clim_file : str, optional
            netCDF file with a 12-month ``soilw_am`` climatology on the model
            grid, used for the initial condition only.

        Raises
        ------
        ValueError
            If a parameter is out of its range, the grid carries no cell
            areas, or the climatology has NaNs over this grid's land.
        FileNotFoundError
            If ``soil_water_clim_file`` does not exist.

        """
        super().__init__(name=name, grid=grid)
        self.params = BucketLandParameters.default() if params is None else params
        _validate_parameters(self.params)
        if grid.cell_solid_angle is None:
            raise ValueError(
                f"{name}: the grid carries no cell areas (SlabGrid.cell_solid_angle), "
                "which the runoff's mass rate needs. Build it with "
                "SlabGrid.from_coords or from a SCRIP file with grid_area."
            )

        self.soil_water_clim_file = soil_water_clim_file
        self.soil_water_climatology = None
        if soil_water_clim_file is not None:
            if not Path(soil_water_clim_file).exists():
                raise FileNotFoundError(
                    f"Climatology file \"{soil_water_clim_file!s:s}\" does not exist.")
            logger.info("%s: loading %r climatology from %s", name,
                        SOIL_WATER_AVAILABILITY_VARIABLE, soil_water_clim_file)
            climatology = load_monthly_climatology(
                soil_water_clim_file, SOIL_WATER_AVAILABILITY_VARIABLE, grid)
            if bool(jnp.any(jnp.isnan(climatology) & grid.land_mask[..., None])):
                raise ValueError(
                    f"Soil water climatology \"{soil_water_clim_file!s:s}\" has NaNs "
                    "over land points of this grid: the file's land mask and the "
                    "grid's disagree."
                )
            self.soil_water_climatology = climatology

    def initialize(self, params: BucketLandParameters | None = None) -> Carry:
        """Build the initial land carry; see the class docstring."""
        params = self._initial_params(params)
        land = self.grid.land_mask
        wilting = params.soil_water_wilting_point
        capacity = params.soil_water_field_capacity

        if self.soil_water_climatology is not None:
            target = jnp.clip(jnp.where(
                land[..., None], self.soil_water_climatology, 0.0), 0.0, 1.0)
            target = evaluate_cyclic_linear(self.start_year_fraction, target)
            # Both layers at the same fraction ``s`` of their range give an
            # availability of ``s * (d1 (fc1 - wp1) + veg d2 (fc2 - wp2)) /
            # capacity``, so ``s`` follows from the target directly.
            full = _available_water(
                jnp.broadcast_to(capacity, self.grid.shape + (2,)), params)
            reachable = jnp.sum(full, axis=-1) / _capacity(params)
            fill = jnp.clip(target / jnp.where(reachable > 0.0, reachable, 1.0), 0.0, 1.0)
        else:
            fill = jnp.broadcast_to(
                jnp.asarray(params.initial_soil_water_fill, dtype=float), self.grid.shape)

        soil_water = jnp.where(
            land[..., None], wilting + fill[..., None] * (capacity - wilting), 0.0)
        return {
            "params": params,
            "state": BucketLandState(soil_water),  # type: ignore[call-arg]
            "forcing": BucketLandForcing.zeros(self.grid.shape),
            "derived": BucketLandDerived.zeros(
                self.grid.shape,
                soil_water_availability=jnp.where(
                    land, soil_water_availability(soil_water, params), 0.0),
            ),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the soil water by one coupling step; see the class docstring."""
        params = carry["params"]
        forcing = carry["forcing"]
        water = carry["state"].soil_water_content
        land = self.grid.land_mask
        dt = time.dt

        depth = params.soil_bucket_depth
        wilting = params.soil_water_wilting_point
        capacity = params.soil_water_field_capacity
        tau = params.drainage_timescale

        # 1. Precipitation, into the surface layer.
        surface = water[..., 0] + forcing.precipitation / WATER_DENSITY * dt / depth[0]
        deep = water[..., 1]

        # 2. Evaporation, shared by the water each layer can give; dew all to
        # the surface layer. A layer is not taken below its wilting point.
        available = _available_water(jnp.stack([surface, deep], axis=-1), params)
        total = jnp.sum(available, axis=-1)
        surface_share = jnp.where(
            total > 0.0, available[..., 0] / jnp.where(total > 0.0, total, 1.0), 1.0)
        evaporated = forcing.evaporation / WATER_DENSITY * dt
        surface_share = jnp.where(evaporated < 0.0, 1.0, surface_share)
        surface_asked = surface - surface_share * evaporated / depth[0]
        deep_asked = deep - (1.0 - surface_share) * evaporated / depth[1]
        surface = jnp.maximum(surface_asked, wilting[0])
        deep = jnp.maximum(deep_asked, wilting[1])
        unmet = ((surface - surface_asked) * depth[0]
                 + (deep - deep_asked) * depth[1]) * WATER_DENSITY / dt

        # 3. Drainage of the water above the wilting point, implicit in it.
        surface_excess = surface - wilting[0]
        surface_kept = surface_excess / (1.0 + dt / tau[0])
        into_deep = (surface_excess - surface_kept) * depth[0]
        surface = wilting[0] + surface_kept
        deep_excess = deep + into_deep / depth[1] - wilting[1]
        deep_kept = deep_excess / (1.0 + dt / tau[1])
        drained = (deep_excess - deep_kept) * depth[1]
        deep = wilting[1] + deep_kept

        # 4. Overflow above field capacity leaves the model.
        surface_capped = jnp.minimum(surface, capacity[0])
        deep_capped = jnp.minimum(deep, capacity[1])
        overflow = ((surface - surface_capped) * depth[0]
                    + (deep - deep_capped) * depth[1]) * WATER_DENSITY / dt

        new_water = jnp.where(
            land[..., None], jnp.stack([surface_capped, deep_capped], axis=-1), 0.0)
        deep_drainage = jnp.where(land, drained * WATER_DENSITY / dt, 0.0)
        overflow = jnp.where(land, overflow, 0.0)
        runoff = deep_drainage + overflow
        land_area = (self.grid.cell_solid_angle * jcm_constants.rearth ** 2
                     * self.grid.fractional_mask)
        derived = BucketLandDerived.zeros(
            self.grid.shape,
            soil_water_availability=jnp.where(
                land, soil_water_availability(new_water, params), 0.0),
            deep_drainage=deep_drainage,
            soil_water_overflow=overflow,
            runoff=runoff,
            unmet_evaporation=jnp.where(land, unmet, 0.0),
            runoff_mass_rate=jnp.sum(land_area * runoff),
        )

        diagnostics = {
            "state": BucketLandState(new_water),  # type: ignore[call-arg]
            "forcing": forcing,
            "derived": derived,
        }
        return {"params": params, **diagnostics}, diagnostics

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for the land output."""
        state = diagnostics["state"]
        forcing = diagnostics["forcing"]
        derived = diagnostics["derived"]
        dims = ("time",) + self.grid.dims
        flux = {"units": "kg m-2 s-1"}

        def per_land_area(name, long_name, role="derived", **attrs):
            return (dims, getattr(derived, name), {
                "long_name": long_name, **flux,
                "comment": "per unit land area; zero where the cell has no land",
                **attrs, **role_attrs(role)})

        return {
            "soil_water_content": (
                dims + ("soil_layer",), state.soil_water_content, {
                    "long_name": "Volumetric soil water content of the surface "
                                 "and deep layer",
                    "units": "m3 m-3", **role_attrs("state")}),
            "soil_water_availability": (
                dims, derived.soil_water_availability, {
                    "long_name": "Soil water availability (SPEEDY's soilw_am)",
                    "units": "1", **role_attrs("derived")}),
            "deep_drainage": per_land_area(
                "deep_drainage", "Drainage out of the deep soil layer"),
            "soil_water_overflow": per_land_area(
                "soil_water_overflow", "Soil water above field capacity"),
            "runoff": per_land_area(
                "runoff", "Runoff: deep drainage plus overflow"),
            "unmet_evaporation": per_land_area(
                "unmet_evaporation",
                "Evaporation the soil could not supply", positive="up"),
            "runoff_mass_rate": (
                ("time",), derived.runoff_mass_rate, {
                    "long_name": "Runoff summed over the grid",
                    "units": "kg s-1", **role_attrs("derived")}),
            forcing_variable("precipitation"): (
                dims, forcing.precipitation, {
                    "long_name": "Precipitation the land was forced with",
                    **flux, "positive": "down", **role_attrs("forcing")}),
            forcing_variable("evaporation"): (
                dims, forcing.evaporation, {
                    "long_name": "Evaporation over the land part of the cell "
                                 "the land was forced with",
                    **flux, "positive": "up", **role_attrs("forcing")}),
        }


def _validate_parameters(params: BucketLandParameters) -> None:
    """Refuse parameters the bucket cannot integrate, naming the field."""
    def array(field):
        return np.asarray(getattr(params, field), dtype=float)

    for field in ("soil_bucket_depth", "drainage_timescale",
                  "soil_water_field_capacity", "soil_water_wilting_point"):
        value = array(field)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(
                f"params.{field} must be two finite values (surface, deep); "
                f"got {value!r}.")
    if np.any(array("soil_bucket_depth") <= 0.0):
        raise ValueError("params.soil_bucket_depth must be positive.")
    if np.any(array("drainage_timescale") <= 0.0):
        raise ValueError("params.drainage_timescale must be positive.")
    wilting, capacity = array("soil_water_wilting_point"), array("soil_water_field_capacity")
    if np.any(wilting < 0.0) or np.any(capacity <= wilting) or np.any(capacity > 1.0):
        raise ValueError(
            "params.soil_water_wilting_point and soil_water_field_capacity must "
            f"satisfy 0 <= wilting < capacity <= 1 per layer; got {wilting!r} and "
            f"{capacity!r}.")
    for field in ("vegetation_fraction", "initial_soil_water_fill"):
        value = array(field)
        if not np.all(np.isfinite(value)) or np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError(f"params.{field} must lie in [0, 1]; got {value!r}.")
