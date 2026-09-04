"""Land surface model component - translated from Speedy Fortran.

This module implements a slab land-surface model with:
- Temperature evolution with heat capacity and dissipation
- Snow depth climatology
- Soil water availability climatology
- Land/ice-sheet discrimination based on albedo

Physics based on SPEEDY (Simplified Parameterizations, primiTivE-Equation DYnamics):
Molteni, F. (2003). Atmospheric simulations using a GCM with simplified
physical parametrizations. I: model climatology and variability in
multi-decadal experiments. Climate Dynamics, 20(2-3), 175-191.

Programmer: Aya Lalou

Translation from: https://github.com/samhatfield/speedy.f90/blob/master/source/land_model.f90
"""
import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import tree_math

from jem.base.component import Carry, CouplingTime, Diagnostics
from jem.components.slab.base import (
    MASKED_SURFACE_TEMPERATURE,
    SlabModelBase,
    end_of_step,
    first_present_variable,
    load_monthly_climatology,
)
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_land_model.params import SlabLandParameters
from jem.utils.cycles import evaluate_cyclic_linear

logger = logging.getLogger(__name__)

_MONTHS_PER_YEAR = 12

# Names the same field goes by in the boundary files JEM is given: SPEEDY's
# own spelling first, then jax-gcm's. Resolving by name (rather than by
# position in the file, as this model used to) is what lets
# ``load_monthly_climatology`` verify that the file is on the model grid.
_SNOW_DEPTH_NAMES = ("snowd", "snowc")
_SOIL_WATER_NAMES = ("soilw", "soilw_am")
_SURFACE_TEMPERATURE_NAMES = ("stl",)


@tree_math.struct
class LandState:
    land_surface_temperature: jnp.ndarray
    snowc: jnp.ndarray
    soilw: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        land_surface_temperature=None,
        snowc=None,
        soilw=None,
    ):
        return cls(
            land_surface_temperature if land_surface_temperature is not None else jnp.zeros(shape),
            snowc if snowc is not None else jnp.zeros(shape),
            soilw if soilw is not None else jnp.zeros(shape),
        )


@tree_math.struct
class LandForcing:
    total_heat_flux: jnp.ndarray

    @classmethod
    def zeros(cls, shape, total_heat_flux=None):
        return cls(
            total_heat_flux if total_heat_flux is not None else jnp.zeros(shape),
        )


class SlabLandModel(SlabModelBase):
    """Slab land-surface model with:

    - Heat capacity-based temperature evolution
    - Snow depth and soil moisture from climatology
    - Separate treatment of soil and ice sheets

    Based on SPEEDY's land model with prescribed climatological boundary
    conditions. The prognostic variable is the *anomaly* of the surface
    temperature about its climatology: it is damped towards zero on
    ``params.tdland`` and forced by the surface heat flux, and the climatology
    is added back at the end of the step. Cells whose land fraction is below
    ``params.flandmin`` have no anomaly at all (SPEEDY's ``dmask``), so a cell
    that is mostly ocean simply follows the climatology.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: SlabLandParameters | None = None,
        *,
        name: str = "lnd",
        land_clim_file: str | None = None,
        surface_albedo: jnp.ndarray | None = None,
    ):
        """Initialize the land surface model.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid.
        params : SlabLandParameters, optional
            Tunable parameters; defaults to
            :meth:`SlabLandParameters.default`.
        name : str
            Component name in the coupler's workflow and carry.
        land_clim_file : str, optional
            netCDF file holding 12-month climatologies of surface temperature
            (``stl``), snow depth (``snowd`` or ``snowc``) and soil water
            availability (``soilw`` or ``soilw_am``) on the model grid. Fields
            the file does not carry fall back to idealized ones.
        surface_albedo : jnp.ndarray, optional
            Surface albedo on the model grid, used to tell an ice sheet from
            soil (SPEEDY reads it from its topography file). Defaults to a
            uniform ``params.surface_albedo``, which is below the ice
            threshold, so a model built without one is all soil everywhere.

        Raises
        ------
        ValueError
            If a parameter is out of range or ``surface_albedo`` is not on the
            model grid.
        FileNotFoundError
            If ``land_clim_file`` does not exist.

        """
        super().__init__(name=name, grid=grid)
        self.params = SlabLandParameters.default() if params is None else params
        self.land_clim_file = land_clim_file

        if not float(self.params.tdland) > 0.0:
            raise ValueError("tdland must be a positive number of seconds.")
        if not 0.0 <= float(self.params.land_threshold) <= 1.0:
            raise ValueError("land_threshold must be a land fraction in [0, 1].")
        if not 0.0 <= float(self.params.flandmin) <= 1.0:
            raise ValueError("flandmin must be a land fraction in [0, 1].")
        if float(self.params.snow_depth_to_cover_scale) <= 0.0:
            raise ValueError(
                "snow_depth_to_cover_scale must be a positive snow depth in mm."
            )

        if surface_albedo is None:
            self.surface_albedo = jnp.full(self.grid.shape, self.params.surface_albedo)
        else:
            surface_albedo = jnp.asarray(surface_albedo)
            if tuple(surface_albedo.shape) != self.grid.shape:
                raise ValueError(
                    f"surface_albedo has shape {tuple(surface_albedo.shape)}, but the "
                    f"grid is {self.grid.shape} (n_lon, n_lat)."
                )
            self.surface_albedo = surface_albedo

        # Boundary data is *configuration*, so it is read here rather than in
        # ``initialize()``: that keeps initialize() pure with respect to self
        # and surfaces a bad file at construction.
        (
            self.surface_temperature_climatology,
            self.snow_depth_climatology,
            self.soil_water_climatology,
        ) = self._load_climatologies()

    def _load_climatologies(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Load the three boundary climatologies, with idealized fallbacks.

        Every field is read BY NAME through
        :func:`~jem.components.slab.base.load_monthly_climatology`, which
        checks it against the model grid. The previous positional read
        (``jnp.array(ds["stl"].values)``) accepted any array whose shape
        happened to fit, so a file on a different grid, or written in a
        different axis order, loaded silently and wrongly.
        """
        shape = self.grid.shape
        monthly = shape + (_MONTHS_PER_YEAR,)

        if self.land_clim_file is None:
            logger.info(
                "%s: no land climatology file; using an idealized climatology.",
                self.name,
            )
            return (
                self._idealized_land_temperature(),
                jnp.zeros(monthly),
                jnp.full(monthly, 0.5),
            )

        if not Path(self.land_clim_file).exists():
            raise FileNotFoundError(
                f"Land climatology file \"{self.land_clim_file!s:s}\" does not exist."
            )

        surface_temperature = self._load_named(_SURFACE_TEMPERATURE_NAMES)
        if surface_temperature is None:
            logger.warning(
                "%s: land climatology file %s has none of %r; using an idealized "
                "surface-temperature climatology.",
                self.name,
                self.land_clim_file,
                _SURFACE_TEMPERATURE_NAMES,
            )
            surface_temperature = self._idealized_land_temperature()

        snow_depth = self._load_named(_SNOW_DEPTH_NAMES)
        if snow_depth is None:
            logger.warning(
                "%s: land climatology file %s has none of %r; assuming no snow.",
                self.name,
                self.land_clim_file,
                _SNOW_DEPTH_NAMES,
            )
            snow_depth = jnp.zeros(monthly)

        soil_water = self._load_named(_SOIL_WATER_NAMES)
        if soil_water is None:
            logger.warning(
                "%s: land climatology file %s has none of %r; assuming uniform soil "
                "water availability of 0.5.",
                self.name,
                self.land_clim_file,
                _SOIL_WATER_NAMES,
            )
            soil_water = jnp.full(monthly, 0.5)

        return surface_temperature, snow_depth, soil_water

    def _load_named(self, candidates: tuple[str, ...]) -> jnp.ndarray | None:
        """Load whichever of `candidates` the climatology file carries."""
        name = first_present_variable(self.land_clim_file, candidates)
        if name is None:
            return None
        logger.info(
            "%s: loading %r climatology from %s", self.name, name, self.land_clim_file
        )
        return load_monthly_climatology(self.land_clim_file, name, self.grid)

    def initialize(self) -> Carry:
        """Build the initial land carry."""
        params = self.params
        land = _land_cells(self.grid, params)
        # The month the run starts in; the coupler sets it through `bind`.
        cycle_position = self.start_year_fraction

        surface_temperature = jnp.where(
            land,
            evaluate_cyclic_linear(
                cycle_position, self.surface_temperature_climatology
            ),
            MASKED_SURFACE_TEMPERATURE,
        )

        return {
            "params": params,
            "state": LandState.zeros(
                self.grid.shape,
                land_surface_temperature=surface_temperature,
                snowc=_snow_cover(
                    evaluate_cyclic_linear(
                        cycle_position, self.snow_depth_climatology
                    ),
                    params,
                ),
                soilw=evaluate_cyclic_linear(
                    cycle_position, self.soil_water_climatology
                ),
            ),
            "forcing": LandForcing.zeros(self.grid.shape),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the land surface by one coupling step.

        Follows SPEEDY's ``run_land_model``: interpolate the climatology to the
        start and end of the step, evolve the temperature anomaly about it with
        the surface heat flux and the dissipation coefficient, then add the
        end-of-step climatology back.
        """
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]

        land = _land_cells(self.grid, params)
        land_fraction = _land_fraction(self.grid, params)
        # SPEEDY's dmask: only cells that are mostly land carry an anomaly of
        # their own; the rest follow the climatology exactly.
        anomaly_mask = jnp.where(land_fraction >= params.flandmin, 1.0, 0.0)

        # Heat capacity per unit area, and its reciprocal scaled by the step:
        # SPEEDY's rhcapl. Computed here rather than cached on the model because
        # both the depths and the coupling timestep are things a caller may vary
        # between runs -- and, for the depths, differentiate through.
        heat_capacity = jnp.where(
            self.surface_albedo < params.land_ice_albedo_threshold,
            params.depth_soil * params.soil_volumetric_heat_capacity,
            params.depth_lice * params.land_ice_volumetric_heat_capacity,
        )
        inverse_heat_capacity = time.dt / heat_capacity

        # SPEEDY's cdland, in units of the coupling step.
        dissipation_steps = params.tdland / time.dt
        dissipation = (anomaly_mask * dissipation_steps) / (
            1.0 + anomaly_mask * dissipation_steps
        )

        climatology_begin = evaluate_cyclic_linear(
            time.year_fraction, self.surface_temperature_climatology
        )
        climatology_end = evaluate_cyclic_linear(
            end_of_step(time).year_fraction, self.surface_temperature_climatology
        )
        snow_depth = evaluate_cyclic_linear(
            time.year_fraction, self.snow_depth_climatology
        )
        soil_water = evaluate_cyclic_linear(
            time.year_fraction, self.soil_water_climatology
        )

        # The heat flux arrives in the coupler's upward-positive convention;
        # the land is warmed by what flows into it, hence the negation.
        downward_heat_flux = -forcing.total_heat_flux

        anomaly = state.land_surface_temperature - climatology_begin
        new_anomaly = dissipation * (anomaly + inverse_heat_capacity * downward_heat_flux)
        surface_temperature = jnp.where(
            land, new_anomaly + climatology_end, MASKED_SURFACE_TEMPERATURE
        )

        new_state = state.replace(
            land_surface_temperature=surface_temperature,
            snowc=_snow_cover(snow_depth, params),
            soilw=soil_water,
        )

        diagnostics = {
            "state": new_state,
            "forcing": forcing,
        }
        return {"params": params, **diagnostics}, diagnostics

    def _idealized_land_temperature(self) -> jnp.ndarray:
        """Idealised monthly land-temperature climatology.

        The latitude dependence is taken from the grid's own 2-D latitude
        field rather than reconstructed from a shape tuple, so the axis order
        cannot be confused: an earlier version unpacked `grid.shape` as
        `(n_lat, n_lon)` when it is in fact `(n_lon, n_lat)`, and so laid the
        pole-to-pole profile out along the LONGITUDE axis. On JCM grids the
        two axis lengths differ only by a factor of two, so the shapes lined
        up and the error was silent.

        Returns
        -------
        jnp.ndarray
            Monthly climatology of shape ``(n_lon, n_lat, 12)`` in Kelvin,
            matching the model's ``(lon, lat, time)`` climatology layout.

        """
        lat = self.grid.latitude_radian                      # (n_lon, n_lat)
        months = jnp.arange(_MONTHS_PER_YEAR)

        # Warm equator, cold poles.
        base_T = 273.15 + 25.0 * jnp.cos(lat)

        # Seasonal amplitude is largest at high latitudes, zero at the equator.
        seasonal_amp = 15.0 * jnp.sin(jnp.abs(lat)) ** 2

        # Peak in March (month index 2).
        phase = 2 * jnp.pi * (months - 2) / _MONTHS_PER_YEAR

        return (
            base_T[..., None]
            + seasonal_amp[..., None] * jnp.cos(phase)[None, None, :]
        )

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for land output."""
        state = diagnostics["state"]
        forcing = diagnostics["forcing"]
        dims = ("time",) + self.grid.dims

        return {
            "land_surface_temperature": (
                dims,
                state.land_surface_temperature,
                {
                    "long_name": "Land surface temperature",
                    "units": "K",
                },
            ),
            "snowc": (
                dims,
                state.snowc,
                {
                    "long_name": "Snow cover fraction",
                    "units": "1",
                },
            ),
            "soilw": (
                dims,
                state.soilw,
                {
                    "long_name": "Soil water availability",
                    "units": "1",
                },
            ),
            "total_heat_flux": (
                dims,
                forcing.total_heat_flux,
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m-2",
                    "positive": "upward",
                },
            ),
        }

    def _create_xarray_global_attributes(self) -> dict[str, Any]:
        return {
            "description": "SPEEDY-based slab land surface model output",
            "depth_soil": f"{float(self.params.depth_soil)} m",
            "depth_lice": f"{float(self.params.depth_lice)} m",
            "tdland": f"{float(self.params.tdland)} s",
        }


def _land_fraction(grid: SlabGrid, params: SlabLandParameters) -> jnp.ndarray:
    """SPEEDY's fmask_l: land fraction, snapped to 0 or 1 near the ends."""
    threshold = params.land_threshold
    fractional_mask = grid.fractional_mask
    return jnp.where(
        fractional_mask >= threshold,
        jnp.where(fractional_mask > (1.0 - threshold), 1.0, fractional_mask),
        0.0,
    )


def _land_cells(grid: SlabGrid, params: SlabLandParameters) -> jnp.ndarray:
    """Boolean mask of the cells this model integrates (SPEEDY's bmask_l)."""
    return grid.fractional_mask >= params.land_threshold


def _snow_cover(snow_depth: jnp.ndarray, params: SlabLandParameters) -> jnp.ndarray:
    """Snow cover fraction from snow depth (SPEEDY's sd2sc closure)."""
    return jnp.minimum(1.0, snow_depth / params.snow_depth_to_cover_scale)
