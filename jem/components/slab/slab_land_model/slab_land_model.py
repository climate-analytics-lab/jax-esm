"""Slab land model with a two-layer soil water bucket.

The model has these parts:

- Surface temperature: a one-layer slab heated by the surface heat flux. It
  either runs freely or relaxes to a climatology (SPEEDY's scheme).
- Snow: snow depth is either read from a climatology or held at its initial
  value. Snow cover is diagnosed from the depth.
- Soil water: a two-layer leaky bucket filled by precipitation and emptied
  by evaporation, drainage and overflow above field capacity.
- Soil water availability: SPEEDY's ``soilw_am``, diagnosed from the bucket
  or read from a climatology.
- Surface albedo: bare soil and vegetation albedos blended by the
  vegetation fraction, or read from a climatology. It is output only for
  now.

The slab heat capacity mixes soil and glacier ice in proportion to
``glacier_fraction``, a single value or a time-invariant map read from the
climatology file.
"""

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tree_math
import xarray as xr

from jem.base.component import Carry, CouplingTime, Diagnostics, TimeAxis
from jem.components.slab.base import (
    MASKED_SURFACE_TEMPERATURE,
    SlabModelBase,
    first_present_variable,
    forcing_variable,
    load_monthly_climatology,
    load_static_field,
    role_attrs,
)
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_land_model.params import (
    SlabLandParameters,
)
from jem.utils.cycles import evaluate_cyclic_linear
from jem.utils.esmf_regrid import ESMFRegridder

logger = logging.getLogger(__name__)
_MONTHS_PER_YEAR = 12

@tree_math.struct
class LandState:
    """Land state.

    Attributes
    ----------
    land_surface_temperature : (n_lon, n_lat) array
        Surface temperature (K). Set to ``MASKED_SURFACE_TEMPERATURE`` on
        cells with no land.
    snow_depth : (n_lon, n_lat) array
        Snow depth (mm water equivalent).
    snow_cover : (n_lon, n_lat) array
        Snow cover fraction in [0, 1], diagnosed from ``snow_depth``.
    soil_water_content : (n_lon, n_lat, 2) array
        Volumetric soil water content (m3 m-3) of the surface and deep layer.
    vegetation_fraction : (n_lon, n_lat) array
        Fraction of the cell covered by vegetation (dimensionless, in
        [0, 1]). It is kept as state for future vegetation dynamics, but
        nothing evolves it yet: it starts at 0 and stays there.

    """

    land_surface_temperature: jnp.ndarray
    snow_depth: jnp.ndarray
    snow_cover: jnp.ndarray
    soil_water_content: jnp.ndarray
    vegetation_fraction: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        land_surface_temperature=None,
        snow_depth=None,
        snow_cover=None,
        soil_water_content=None,
        vegetation_fraction=None,
        n_soil_layers=2,
    ):
        return cls(
            land_surface_temperature if land_surface_temperature is not None else jnp.zeros(shape),
            snow_depth if snow_depth is not None else jnp.zeros(shape),
            snow_cover if snow_cover is not None else jnp.zeros(shape),
            soil_water_content if soil_water_content is not None else jnp.zeros(shape + (n_soil_layers,)),
            vegetation_fraction if vegetation_fraction is not None else jnp.zeros(shape),
        )


@tree_math.struct
class LandForcing:
    """Land forcing, filled by exchangers.

    Attributes
    ----------
    total_heat_flux : (n_lon, n_lat) array
        Net surface heat flux (W m-2), positive upward.
    precipitation : (n_lon, n_lat) array
        Precipitation rate (kg m-2 s-1, the same as mm s-1 of water).
    evaporation : (n_lon, n_lat) array
        Evaporation rate (kg m-2 s-1), positive upward; negative is dew.

    """

    total_heat_flux: jnp.ndarray
    precipitation: jnp.ndarray
    evaporation: jnp.ndarray

    @classmethod
    def zeros(cls, shape, total_heat_flux=None, precipitation=None, evaporation=None):
        return cls(
            total_heat_flux if total_heat_flux is not None else jnp.zeros(shape),
            precipitation if precipitation is not None else jnp.zeros(shape),
            evaporation if evaporation is not None else jnp.zeros(shape),
        )


@tree_math.struct
class LandDerived:
    """What the land diagnoses every step and other components may read.

    Attributes
    ----------
    surface_albedo : (n_lon, n_lat) array
        Surface albedo (dimensionless, in [0, 1]). The climatology when
        ``surface_albedo_from_clim_file=True``; otherwise
        ``baresoil_albedo * (1 - vegetation_fraction) + vegetation_albedo * vegetation_fraction``.
        It is written to the output, but nothing in the model or the coupling
        reads it yet.
    deep_drainage : (n_lon, n_lat) array
        Water draining out of the bottom of the deep soil layer, which leaves
        the land model (kg m-2 s-1). Zero where there is no land.
    soil_water_overflow : (n_lon, n_lat) array
        Water above field capacity that left the two soil layers this step,
        summed over both layers (kg m-2 s-1). It leaves the land model as
        runoff. Zero where there is no land.
    runoff_to_ocean : array
        ``deep_drainage + soil_water_overflow`` passed through the model's
        ``runoff_to_ocean_mapping``: the same water, as the flux the ocean
        receives (kg m-2 s-1).
    unmet_evaporation : (n_lon, n_lat) array
        The part of ``forcing.evaporation`` the surface soil layer had no
        water left to supply (kg m-2 s-1, positive upward, never negative).
        The atmosphere evaporated it anyway, so it is water the coupled
        system created. Zero where there is no land.
    soil_water_availability : (n_lon, n_lat) array
        Fraction of the available soil water the soil holds (dimensionless,
        in [0, 1]), the land's counterpart of SPEEDY's ``soilw_am``. With
        depths ``d``, field capacities ``fc``, wilting points ``wp`` and
        vegetation fraction ``f_veg``, it is
        ``(d1 max(W1 - wp1, 0) + f_veg d2 max(W2 - wp2, 0)) / (d1 (fc1 - wp1) + d2 (fc2 - wp2))``,
        clipped to [0, 1].
        The deep layer counts only through vegetation, whose roots reach it.
        With the default parameters (surface wilting point 0, depth ratio
        ``d2 / d1 = 3``) this is SPEEDY's ``soilw_am``; a non-zero surface
        wilting point also leaves out the surface water below it. Computed
        from the soil water at the end of the step. With
        ``soil_water_availability_from_clim_file=True`` it is the climatology
        at the current time instead, also clipped to [0, 1]. Zero where there
        is no land.

    """

    surface_albedo: jnp.ndarray
    deep_drainage: jnp.ndarray
    soil_water_overflow: jnp.ndarray
    runoff_to_ocean: jnp.ndarray
    unmet_evaporation: jnp.ndarray
    soil_water_availability: jnp.ndarray

    @classmethod
    def zeros(cls, shape, surface_albedo=None, deep_drainage=None, soil_water_overflow=None, runoff_to_ocean=None, unmet_evaporation=None, soil_water_availability=None):
        return cls(
            surface_albedo if surface_albedo is not None else jnp.zeros(shape),
            deep_drainage if deep_drainage is not None else jnp.zeros(shape),
            soil_water_overflow if soil_water_overflow is not None else jnp.zeros(shape),
            runoff_to_ocean if runoff_to_ocean is not None else jnp.zeros(shape),
            unmet_evaporation if unmet_evaporation is not None else jnp.zeros(shape),
            soil_water_availability if soil_water_availability is not None else jnp.zeros(shape),
        )


class SlabLandModel(SlabModelBase):
    """Slab land model with a two-layer soil water bucket.

    Only cells with some land (``grid.binary_land_mask == 1``) are
    integrated. Other cells hold ``MASKED_SURFACE_TEMPERATURE`` and zero
    soil water.

    Surface temperature
        A single slab whose heat capacity mixes soil and glacier ice. With
        ``f = glacier_fraction`` (a scalar, or a map read from the
        climatology file with ``glacier_fraction_from_clim_file=True``)::

            C = (1 - f) * soil_thermal_depth * soil_volumetric_heat_capacity
                + f * glacier_depth * glacier_volumetric_heat_capacity

        With ``F`` the downward heat flux, the default update is::

            T_new = T + dt * F / C

        With ``land_surface_temperature_from_clim_file=True`` the model uses
        SPEEDY's scheme instead. The anomaly ``T' = T - T_clim`` is forced by
        ``F`` and damped on ``temperature_relaxation_timescale`` (``tau``)::

            T'_new = tau / (tau + dt) * (T' + dt * F / C)
            T_new  = T'_new + T_clim(end of step)

    Snow
        With ``snow_depth_from_clim_file=True`` the snow depth is the
        climatology at the current time. Otherwise it stays at its initial
        value. Snow cover is
        ``min(1, snow_depth / snow_depth_to_cover_scale)``.

    Soil water
        By default, a two-layer leaky bucket. With ``W1``, ``W2`` the
        volumetric water content, ``d1``, ``d2`` the layer depths,
        ``tau1``, ``tau2`` the drainage timescales, ``P`` the
        precipitation and ``E`` the evaporation::

            dW1/dt = (P - E) / (rho_w * d1) - max(W1 - wp1, 0) / tau1
            dW2/dt = (d1 / d2) * max(W1 - wp1, 0) / tau1 - max(W2 - wp2, 0) / tau2

        with ``wp1``, ``wp2`` the wilting points: water at or below the
        wilting point is held by the soil and does not drain. Evaporation
        can still take it.

        Evaporation is taken from the surface layer only, after this step's
        precipitation and drainage. It is limited to the water that layer
        holds: the part it cannot supply is published as
        ``derived.unmet_evaporation`` rather than clamped away, so the land's
        water budget still closes and the gap is visible. Dew (``E < 0``)
        adds water. The evaporation is the cell mean the atmosphere
        publishes, so in a partly-land cell the soil also supplies the sea
        part, and there is no snow budget, so sublimation is taken from the
        soil too.

        Water leaving the surface layer enters the deep layer, and water
        leaving the deep layer leaves the model; its rate is published as
        ``derived.deep_drainage``. Both layers are kept non-negative and no
        higher than their field capacity ``soil_water_field_capacity``. The
        water above field capacity leaves the model directly, without
        entering the layer below, and is published as
        ``derived.soil_water_overflow``. The deep drainage and the overflow
        together are routed to the ocean as ``derived.runoff_to_ocean``.

    Soil water availability
        Diagnosed from the bucket's soil water (see
        ``LandDerived.soil_water_availability``). With
        ``soil_water_availability_from_clim_file=True`` it is the climatology
        at the current time instead. The bucket runs either way, and a
        prescribed availability does not feed back on the soil water. Once the
        atmosphere reads it, a prescribed wet availability over dry soil makes
        the atmosphere evaporate water the soil does not hold, which shows up
        as ``derived.unmet_evaporation``.

    Surface albedo
        With ``surface_albedo_from_clim_file=True`` the albedo is the
        climatology at the current time. Otherwise it is the bare soil and
        vegetation albedos weighted by the vegetation fraction,
        ``baresoil_albedo * (1 - vegetation_fraction) + vegetation_albedo * vegetation_fraction``.
        It is written to the output, but nothing in the model or the coupling
        reads it yet.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: SlabLandParameters | None = None,
        *,
        name: str = "lnd",
        land_clim_file: str | None = None,
        soil_water_availability_from_clim_file: bool = False,
        snow_depth_from_clim_file: bool = False,
        land_surface_temperature_from_clim_file: bool = False,
        surface_albedo_from_clim_file: bool = False,
        glacier_fraction_from_clim_file: bool = False,
        soil_water_availability_varname_in_clim_file: str = "soil_water_availability",
        snow_depth_varname_in_clim_file: str = "snow_depth",
        land_surface_temperature_varname_in_clim_file: str = "land_surface_temperature",
        surface_albedo_varname_in_clim_file: str = "surface_albedo",
        glacier_fraction_varname_in_clim_file: str = "glacier_fraction",
        runoff_to_ocean_mapping: str | None = None,
    ):
        """Initialize the land surface model.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid.
        params : SlabLandParameters, optional
            Model parameters, defaulting to
            :meth:`SlabLandParameters.default`. They are checked here, since
            inside ``step`` they are traced and cannot be inspected.
        name : str
            Component name in the coupler's workflow and carry. Default
            ``"lnd"``.
        land_clim_file : str, optional
            netCDF file with 12-month climatologies on the model grid. A field
            with no time dimension is used as the same map in every month. A
            field missing from the file, or every field when no file is given,
            uses a fallback: an idealized temperature climatology, no snow, a
            soil water availability of 0.5 and a surface albedo of 0.2.
        soil_water_availability_from_clim_file : bool
            Read the soil water availability from the climatology instead of
            diagnosing it from the bucket. The bucket still runs. Default
            False.
        snow_depth_from_clim_file : bool
            Read snow depth from the climatology instead of holding it fixed.
            Default False.
        land_surface_temperature_from_clim_file : bool
            Relax the surface temperature to the climatology instead of
            letting it run freely. Default False.
        surface_albedo_from_clim_file : bool
            Read the surface albedo from the climatology instead of blending
            ``params.baresoil_albedo`` and ``params.vegetation_albedo`` by the
            vegetation fraction. Without a ``land_clim_file`` the climatology
            is ``params.baresoil_albedo`` everywhere. Default False.
        glacier_fraction_from_clim_file : bool
            Read a glacier fraction map from ``land_clim_file`` and use it as
            ``params.glacier_fraction``. The variable must have no time
            dimension. Cells with no land are set to 0, so the file may hold
            NaN or a fill value over ocean; a NaN over land is an error.
            Default False.
        soil_water_availability_varname_in_clim_file : str
            Name of the soil water availability variable in
            ``land_clim_file``.
        snow_depth_varname_in_clim_file : str
            Name of the snow depth variable in ``land_clim_file``.
        land_surface_temperature_varname_in_clim_file : str
            Name of the surface temperature variable in ``land_clim_file``.
        surface_albedo_varname_in_clim_file : str
            Name of the surface albedo variable in ``land_clim_file``.
        glacier_fraction_varname_in_clim_file : str
            Name of the glacier fraction variable in ``land_clim_file``.
        runoff_to_ocean_mapping : str or a function
            If it is a string, is a NetCDF file that can be used with
            ESMFRegridder to rout runoff to ocean grid. If it is a function, 
            it accepts river runoff and spits out the mapped flux. 
            Either way the routing stays on the model's own grid: it takes
            and returns a field of ``grid.shape``, moving the runoff from land
            cells to ocean cells of that grid. ``derived.runoff_to_ocean``
            is then on the same grid as everything else this model writes.


        Raises
        ------
        ValueError
            If a parameter is out of range, a field in ``land_clim_file`` is
            not on the model grid, or ``glacier_fraction_from_clim_file`` is
            set without a ``land_clim_file`` or with a glacier fraction that
            varies in time, or if ``runoff_to_ocean_mapping`` does not return
            a field on the model grid.
        FileNotFoundError
            If ``land_clim_file`` does not exist.

        """
        super().__init__(name=name, grid=grid)
        self.params = SlabLandParameters.default() if params is None else params
        self.land_clim_file = land_clim_file
        self.varnames_in_clim_file = {
            "snow_depth": snow_depth_varname_in_clim_file,
            "soil_water_availability": soil_water_availability_varname_in_clim_file,
            "land_surface_temperature": land_surface_temperature_varname_in_clim_file,
            "surface_albedo": surface_albedo_varname_in_clim_file,
            "glacier_fraction": glacier_fraction_varname_in_clim_file,
        }

        self.read_from_clim_file = {
            "snow_depth": snow_depth_from_clim_file,
            "soil_water_availability": soil_water_availability_from_clim_file,
            "land_surface_temperature": land_surface_temperature_from_clim_file,
            "surface_albedo": surface_albedo_from_clim_file,
            "glacier_fraction": glacier_fraction_from_clim_file,
        }

        if runoff_to_ocean_mapping is None:
            runoff_to_ocean_mapping = lambda runoff : runoff
        elif isinstance(runoff_to_ocean_mapping, str):
            regridder = ESMFRegridder(runoff_to_ocean_mapping)
            runoff_to_ocean_mapping = lambda runoff: regridder(runoff)

        self.runoff_to_ocean_mapping = runoff_to_ocean_mapping

        # The routing must stay on this grid: runoff_to_ocean is carried and
        # written out on it, and a field of another shape would change the
        # scan carry between initialize and step. Checked once here, with
        # jax.eval_shape so no data is moved, rather than failing inside scan.
        routed = jax.eval_shape(
            self.runoff_to_ocean_mapping, jax.ShapeDtypeStruct(self.grid.shape, jnp.float32)
        )
        if routed.shape != tuple(self.grid.shape):
            raise ValueError(
                "runoff_to_ocean_mapping must route runoff within the model grid "
                f"{tuple(self.grid.shape)}; it returns a field of shape {routed.shape}."
            )

        # The glacier map goes into the parameters rather than onto the model,
        # so the heat capacity has one source and the map stays a pytree leaf
        # that jax.grad can reach. It is loaded before the parameter checks
        # below, so the map is range-checked like a scalar would be.
        if glacier_fraction_from_clim_file:
            self.params = self.params.replace(
                glacier_fraction=self._load_static_field(self.varnames_in_clim_file["glacier_fraction"])
            )

        # Both timescales are read as a ratio against a length -- the relaxation
        # as `temperature_relaxation_timescale / dt`, the snow cover as `snow_depth / scale` -- and an
        # infinite one is silent where a zero one is not: it damps the land
        # temperature by `inf / (1 + inf)`, which is NaN, and it puts the snow
        # cover at zero everywhere however deep the snow. Finiteness is
        # therefore required as well as sign (the fractions below get it from
        # being bounded on both sides).
        temperature_relaxation_timescale = float(self.params.temperature_relaxation_timescale)
        if not math.isfinite(temperature_relaxation_timescale) or temperature_relaxation_timescale <= 0.0:
            raise ValueError(
                "params.temperature_relaxation_timescale must be a finite positive number of seconds; "
                f"got {temperature_relaxation_timescale!r}."
            )
        snow_depth_to_cover_scale = float(self.params.snow_depth_to_cover_scale)
        if (
            not math.isfinite(snow_depth_to_cover_scale)
            or snow_depth_to_cover_scale <= 0.0
        ):
            raise ValueError(
                "params.snow_depth_to_cover_scale must be a finite positive snow "
                f"depth in mm; got {snow_depth_to_cover_scale!r}."
            )
        # The parameters are validated here, at construction, because inside
        # `step` they are traced values that cannot be inspected. A caller who
        # replaces `carry["params"]` afterwards takes on that responsibility.
        # Both are fractions. A NaN would propagate silently: through the
        # heat capacity into the temperature for the glacier fraction, and
        # into the output for the albedo. The glacier fraction may be a map,
        # so every element is checked, and a map must be on the model grid
        # (anything else would broadcast or fail deep inside `step`).
        for name in ("baresoil_albedo", "vegetation_albedo", "glacier_fraction"):
            fraction = np.asarray(getattr(self.params, name), dtype=float)
            if not np.all(np.isfinite(fraction)) or not np.all(
                (fraction >= 0.0) & (fraction <= 1.0)
            ):
                raise ValueError(
                    f"params.{name} must be finite and in [0, 1]; "
                    f"got values in [{np.nanmin(fraction):g}, {np.nanmax(fraction):g}]."
                )
        for map_name in ("glacier_fraction", "baresoil_albedo", "vegetation_albedo"):
            field_shape = np.shape(getattr(self.params, map_name))
            if field_shape not in ((), self.grid.shape):
                raise ValueError(
                    f"params.{map_name} has shape {field_shape}; it must "
                    f"be a scalar or a map on the grid, {self.grid.shape} (n_lon, n_lat)."
                )
        # The slab heat capacity is depth times volumetric capacity; a zero or
        # negative factor makes the temperature update infinite or unstable.
        # The bucket depths divide the bucket's water budget, so they must be
        # positive too; they are per-layer, so every element is checked.
        for factor_name in (
            "soil_bucket_depth",
            "soil_thermal_depth",
            "glacier_depth",
            "soil_volumetric_heat_capacity",
            "glacier_volumetric_heat_capacity",
        ):
            factor = np.asarray(getattr(self.params, factor_name), dtype=float)
            if not np.all(np.isfinite(factor)) or not np.all(factor > 0.0):
                raise ValueError(
                    f"params.{factor_name} must be finite and strictly positive "
                    f"(it divides the heat or water budget); got {factor!r}."
                )

        # Boundary data is *configuration*, so it is read here rather than in
        # ``initialize()``: that keeps initialize() pure with respect to self
        # and surfaces a bad file at construction.
        (
            self.land_surface_temperature_climatology,
            self.snow_depth_climatology,
            self.soil_water_availability_climatology,
            self.surface_albedo_climatology,
        ) = self._load_climatologies()

    def _load_climatologies(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Load the four monthly climatologies, with idealized fallbacks.

        Fields are read by name through
        :func:`~jem.components.slab.base.load_monthly_climatology`, which
        checks them against the model grid.

        Returns
        -------
        tuple of jnp.ndarray
            Surface temperature (K), snow depth (mm), soil water content
            (m3 m-3) and surface albedo, each of shape ``(n_lon, n_lat, 12)``.

        """
        shape = self.grid.shape
        monthly = shape + (_MONTHS_PER_YEAR,)
        
        varnames_in_clim_file = self.varnames_in_clim_file
        
        if self.land_clim_file is None:
            logger.info(
                "%s: no land climatology file; using an idealized climatology.",
                self.name,
            )
            return (
                self._idealized_land_temperature(),
                jnp.zeros(monthly),
                jnp.full(monthly, 0.5),
                # Every month holds the bare soil albedo, which may be a map.
                jnp.broadcast_to(
                    jnp.broadcast_to(self.params.baresoil_albedo, shape)[..., None], monthly
                ),
            )
        
        if not Path(self.land_clim_file).exists():
            raise FileNotFoundError(
                f"Land climatology file \"{self.land_clim_file!s:s}\" does not exist."
            )

        land_surface_temperature = self._load_named([varnames_in_clim_file["land_surface_temperature"],])
        if land_surface_temperature is None:
            logger.warning(
                "%s: land climatology file %s does not have %s; using an idealized "
                "surface-temperature climatology.",
                self.name,
                self.land_clim_file,
                varnames_in_clim_file["land_surface_temperature"],
            )
            land_surface_temperature = self._idealized_land_temperature()

        snow_depth = self._load_named([varnames_in_clim_file["snow_depth"],])
        if snow_depth is None:
            logger.warning(
                "%s: land climatology file %s does not have %s; assuming no snow.",
                self.name,
                self.land_clim_file,
                varnames_in_clim_file["snow_depth"],
            )
            snow_depth = jnp.zeros(monthly)

        soil_water_availability = self._load_named([varnames_in_clim_file["soil_water_availability"],])
        if soil_water_availability is None:
            logger.warning(
                "%s: land climatology file %s does not have %s; assuming uniform soil "
                "water availability of 0.5.",
                self.name,
                self.land_clim_file,
                varnames_in_clim_file["soil_water_availability"],
            )
            soil_water_availability = jnp.full(monthly, 0.5)

        surface_albedo = self._load_named([varnames_in_clim_file["surface_albedo"],])
        if surface_albedo is None:
            logger.warning(
                "%s: land climatology file %s does not have %s; assuming uniform surface_albedo 0.2 ",
                self.name,
                self.land_clim_file,
                varnames_in_clim_file["surface_albedo"],
            )
            surface_albedo = jnp.full(monthly, 0.2)
 
        return (
            land_surface_temperature,
            snow_depth,
            soil_water_availability,
            surface_albedo,
        )

    def _load_static_field(self, varname: str, off_land_value: float = 0.0) -> jnp.ndarray:
        """Load a time-invariant map from the climatology file.

        Unlike the climatologies there is no fallback: a static field is read
        only when the caller asks for the file's map, so a missing file or
        variable is an error rather than a silent uniform value.

        Parameters
        ----------
        varname : str
            Name of the variable in ``land_clim_file``.
        off_land_value : float
            Value given to cells with no land. Default 0.

        Returns
        -------
        jnp.ndarray
            The field with shape ``(n_lon, n_lat)``.

        """
        if self.land_clim_file is None:
            raise ValueError(
                f"Reading the static field \"{varname}\" needs a land_clim_file."
            )
        if not Path(self.land_clim_file).exists():
            raise FileNotFoundError(
                f"Land climatology file \"{self.land_clim_file!s:s}\" does not exist."
            )
        logger.info(
            "%s: loading %r static field from %s",
            self.name,
            varname,
            self.land_clim_file,
        )
        values = np.asarray(
            load_static_field(self.land_clim_file, varname, self.grid), dtype=float
        )
        # Source files usually leave ocean cells as NaN or a fill value. The
        # land model never integrates those cells, so they are overwritten
        # here; a NaN over land is a broken file and is rejected.
        land = np.asarray(self.grid.binary_land_mask) == 1
        if not np.all(np.isfinite(values[land])):
            raise ValueError(
                f"Land climatology file \"{self.land_clim_file!s:s}\": "
                f"\"{varname}\" has non-finite values on land cells."
            )
        return jnp.asarray(np.where(land, values, off_land_value))

    def _load_named(self, candidates: Sequence[str]) -> jnp.ndarray | None:
        """Load whichever of `candidates` the climatology file carries."""
        name = first_present_variable(self.land_clim_file, candidates)
        if name is None:
            return None
        logger.info(
            "%s: loading %r climatology from %s", self.name, name, self.land_clim_file
        )
        return load_monthly_climatology(self.land_clim_file, name, self.grid)

    def initialize(self,
        params: SlabLandParameters | None = None,
        initialize_from_clim_file: bool = False,
    ) -> Carry:
        """Build the initial land carry.

        By default every cell starts at 288.15 K with no snow, no soil water
        and no vegetation. The initial surface albedo is the blend of
        ``params.baresoil_albedo`` and ``params.vegetation_albedo``, which
        with no vegetation is ``params.baresoil_albedo``.

        Parameters
        ----------
        params : SlabLandParameters, optional
            Parameters to put in ``carry["params"]``. Defaults to the ones
            the model was constructed with.
        initialize_from_clim_file : bool
            Start the surface temperature, snow depth and snow cover from the
            climatology at the month the run starts in
            (:attr:`SlabModelBase.start_year_fraction`). Soil water still
            starts at 0, and the initial albedo is the blend even when
            ``surface_albedo_from_clim_file=True``; the first step replaces
            it with the climatology. Default False.

        Returns
        -------
        Carry
            ``{"params", "state", "forcing", "derived"}``. The forcing starts
            at zero and is filled by the exchangers before the first step.

        """
        if (
            params is not None
            and self.read_from_clim_file["glacier_fraction"]
            and jnp.shape(params.glacier_fraction) != self.grid.shape
        ):
            # Passed parameters win, as they do for every other parameter,
            # but losing the file's map that way is easy to do by accident.
            logger.warning(
                "%s: the params passed to initialize() have a uniform "
                "glacier_fraction, so the map read from %s is not used.",
                self.name,
                self.land_clim_file,
            )
        params = self._initial_params(params)
        land = self.grid.binary_land_mask
        # The month the run starts in; the coupler sets it through `bind`.
        cycle_position = self.start_year_fraction

        state = LandState.zeros(self.grid.shape)
        state.land_surface_temperature = state.land_surface_temperature.at[:].set(273.15 + 15.0)
        state.soil_water_content = state.soil_water_content.at[:].set(0.0)
        state.vegetation_fraction = state.vegetation_fraction.at[:].set(0.0)

        derived = LandDerived.zeros(
            self.grid.shape,
            surface_albedo=params.baresoil_albedo * (1.0 - state.vegetation_fraction) + params.vegetation_albedo * state.vegetation_fraction,
            runoff_to_ocean=self.runoff_to_ocean_mapping(jnp.zeros(self.grid.shape)),
        )


        if initialize_from_clim_file:
            logger.info(
                "Initialize land states (land_surface_temperature and snow_depth"
                " with climatology."
            )
            state.land_surface_temperature = jnp.where(
                land,
                evaluate_cyclic_linear(
                    cycle_position, self.land_surface_temperature_climatology
                ),
                MASKED_SURFACE_TEMPERATURE,
            )

            state.snow_depth = evaluate_cyclic_linear(
                cycle_position, self.snow_depth_climatology
            )
            state.snow_cover = convert_snow_depth_to_snow_cover(state.snow_depth, params)

        return {
            "params": params,
            "state": state,
            "forcing": LandForcing.zeros(self.grid.shape),
            # runoff_to_ocean is built by the same routing step uses, so its
            # dtype matches what step returns and the scan carry is stable.
            "derived": derived,
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the land surface by one coupling step.

        See the class docstring for the equations. The ``*_from_clim_file``
        switches pick the branch for each part at trace time.

        Parameters
        ----------
        carry : Carry
            ``{"params", "state", "forcing", "derived"}`` as built by
            :meth:`initialize`.
        time : CouplingTime
            Start time and length (``dt``) of the step.

        Returns
        -------
        tuple[Carry, Diagnostics]
            The new carry, and ``{"state", "forcing", "derived"}`` for output.

        """
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]
        read_from_clim_file = self.read_from_clim_file
        land_mask = self.grid.binary_land_mask

        # The thermal slab has its own depth, separate from the water bucket's
        # layers: SPEEDY also uses 1 m for heat (hcapl) and 7 cm for soil water.
        heat_capacity = (
            params.soil_thermal_depth * params.soil_volumetric_heat_capacity * (1.0 - params.glacier_fraction)
            + params.glacier_depth * params.glacier_volumetric_heat_capacity * params.glacier_fraction
        )
        dt_inverse_heat_capacity = time.dt / heat_capacity

        # The heat flux arrives in the coupler's upward-positive convention;
        # the land is warmed by what flows into it, hence the negation.
        downward_heat_flux = - forcing.total_heat_flux

        if read_from_clim_file["land_surface_temperature"]:
 
            logger.info("Relax land surface temperature to climatology")
 
            lst_climatology_begin = evaluate_cyclic_linear(
                time.year_fraction, self.land_surface_temperature_climatology
            )
            lst_climatology_end = evaluate_cyclic_linear(
                time.end_of_step().year_fraction, self.land_surface_temperature_climatology
            )

           
            dissipation_steps = params.temperature_relaxation_timescale / time.dt
            dissipation = (land_mask * dissipation_steps) / (
                1.0 + land_mask * dissipation_steps
            )

            anomaly = state.land_surface_temperature - lst_climatology_begin
            new_anomaly = dissipation * (anomaly + dt_inverse_heat_capacity * downward_heat_flux)
            new_land_surface_temperature = jnp.where(
                land_mask, new_anomaly + lst_climatology_end, MASKED_SURFACE_TEMPERATURE
            )
        
        else:
            new_land_surface_temperature = state.land_surface_temperature + dt_inverse_heat_capacity * downward_heat_flux

        new_land_surface_temperature = jnp.where(
            land_mask, new_land_surface_temperature, MASKED_SURFACE_TEMPERATURE
        )

        if read_from_clim_file["snow_depth"]:
            logger.info("Read snow depth from climatology")
            new_snow_depth = evaluate_cyclic_linear(
                time.year_fraction, self.snow_depth_climatology
            )
        else:
            new_snow_depth = state.snow_depth
         

        # =====================================================================
        # Land surface soil moisture evolution (two-layer leaky bucket)
        # =====================================================================
        # Layer 0 (surface): receives precipitation, drains into layer 1.
        # Layer 1 (deep): receives layer 0's drainage, drains out as deep drainage.

        W1 = state.soil_water_content[..., 0]
        W2 = state.soil_water_content[..., 1]
        
        drain1 = jnp.maximum(W1 - params.soil_water_wilting_point[0], 0.0) / params.drainage_timescale[0]
        inflow2 = (params.soil_bucket_depth[0] / params.soil_bucket_depth[1]) * drain1

        # Evaporation comes out of what the surface layer holds after this
        # step's precipitation and drainage, and no more: what it cannot
        # supply is published as unmet_evaporation instead of being
        # clamped away, which would silently create water. Dew (negative
        # evaporation) is always taken in.
        surface_water_mass = 1000.0 * params.soil_bucket_depth[0]   # kg m-2 per unit W1
        W1_before_evaporation = jnp.maximum(
            0.0,
            W1 + (forcing.precipitation / surface_water_mass - drain1) * time.dt,
        )
        supplied_evaporation = jnp.minimum(
            forcing.evaporation, W1_before_evaporation * surface_water_mass / time.dt
        )
        unmet_evaporation = forcing.evaporation - supplied_evaporation

        new_W1 = jnp.maximum(
            0.0,
            W1_before_evaporation - supplied_evaporation / surface_water_mass * time.dt,
        )
        
        # The deep layer's drainage leaves the model. It is published as a
        # mass flux (m3 m-3 s-1 times the layer depth and water density),
        # and routed to the ocean as runoff. It equals what is
        # removed below unless dt exceeds the drainage timescale, when the
        # non-negativity clamp would remove less.
        deep_drainage = jnp.maximum(W2 - params.soil_water_wilting_point[1], 0.0) / params.drainage_timescale[1]
        deep_drainage_flux = 1000.0 * params.soil_bucket_depth[1] * deep_drainage

        new_W2 = jnp.maximum(
            0.0,
            W2 + (inflow2 - deep_drainage) * time.dt,
        )

        new_soil_water_content = jnp.stack([new_W1, new_W2], axis=-1)

        # Neither layer holds more than its field capacity. The excess of
        # each layer leaves the model as runoff, without passing through
        # the layer below, and is routed to the ocean with the deep drainage.
        capped_soil_water_content = jnp.minimum(new_soil_water_content, params.soil_water_field_capacity[None, None, :])
        runoff_due_to_overflow = jnp.where(
            land_mask == 1,
            1000 * jnp.sum( (new_soil_water_content - capped_soil_water_content) * params.soil_bucket_depth[None, None, :] / time.dt, axis=-1),
            0.0
        )
        new_soil_water_content = jnp.where(
            land_mask[..., None],
            capped_soil_water_content,
            0.0
        )

        if read_from_clim_file["soil_water_availability"]:
            logger.info("Read soil water availability from climatology")
            new_soil_water_availability = evaluate_cyclic_linear(
                time.year_fraction, self.soil_water_availability_climatology
            )
        else: 
            d1, d2 = params.soil_bucket_depth
            wp1, wp2 = params.soil_water_wilting_point
            fc1, fc2 = params.soil_water_field_capacity
            swc1 = new_soil_water_content[:, :, 0]
            swc2 = new_soil_water_content[:, :, 1]
            f_veg = state.vegetation_fraction
            new_soil_water_availability = (d1 * jnp.maximum(swc1 - wp1, 0.0) + f_veg * d2 * jnp.maximum(swc2 - wp2, 0.0)) /(d1 * (fc1 - wp1) + d2 * (fc2 - wp2))

        new_soil_water_availability = jnp.where(
            land_mask == 1,
            jnp.clip(new_soil_water_availability, 0.0, 1.0),
            0.0
        )


        # The bucket runs either way; a prescribed availability does not feed
        # back on the soil water.
        deep_drainage_flux = jnp.where(land_mask, deep_drainage_flux, 0.0)
        unmet_evaporation = jnp.where(land_mask, unmet_evaporation, 0.0)
        runoff_to_ocean = self.runoff_to_ocean_mapping(deep_drainage_flux + runoff_due_to_overflow)

        if read_from_clim_file["surface_albedo"]:
            logger.info("Read albedo content from climatology")
            new_surface_albedo = evaluate_cyclic_linear(
                time.year_fraction, self.surface_albedo_climatology
            )
        else:
            new_surface_albedo = params.baresoil_albedo * (1.0 - state.vegetation_fraction) + params.vegetation_albedo * state.vegetation_fraction
 
 
        new_state = state.replace(
            land_surface_temperature=new_land_surface_temperature,
            snow_depth=new_snow_depth,
            snow_cover=convert_snow_depth_to_snow_cover(new_snow_depth, params),
            soil_water_content=new_soil_water_content,
        )

        diagnostics = {
            "state": new_state,
            "forcing": forcing,
            "derived": LandDerived.zeros(
                self.grid.shape,
                surface_albedo=new_surface_albedo,
                deep_drainage=deep_drainage_flux,
                soil_water_overflow=runoff_due_to_overflow,
                runoff_to_ocean=runoff_to_ocean,
                unmet_evaporation=unmet_evaporation,
                soil_water_availability=new_soil_water_availability,
            ),
        }
        return {"params": params, **diagnostics}, diagnostics

    def _idealized_land_temperature(self) -> jnp.ndarray:
        """Idealized monthly land surface temperature climatology.

        ``T = 273.15 + 25 cos(lat) + 15 sin^2(lat) cos(2 pi (month - 2) / 12)``,
        so the equator is warm, the poles are cold, and the seasonal cycle
        grows with latitude. Latitude comes from the grid's 2-D field, which
        is already in ``(n_lon, n_lat)`` order.

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
        derived = diagnostics["derived"]
        dims = ("time",) + self.grid.dims
        layer_dims = ("time",) + self.grid.dims + ("layer", )
        return {
            "land_surface_temperature": (
                dims,
                state.land_surface_temperature,
                {
                    "long_name": "Land surface temperature",
                    "units": "K",
                    **role_attrs("state"),
                },
            ),
            "snow_depth": (
                dims,
                state.snow_depth,
                {
                    "long_name": "Snow depth (water equivalent)",
                    "units": "mm",
                    **role_attrs("state"),
                },
            ),

            # layer 0 is the surface soil layer, layer 1 the deep one; their
            # depths are params.soil_bucket_depth.
            "soil_water_content": (
                layer_dims,
                state.soil_water_content,
                {
                    "long_name": "Volumetric soil water content",
                    "units": "m3 m-3",
                    **role_attrs("state"),
                },
            ),
            forcing_variable("total_heat_flux"): (
                dims,
                forcing.total_heat_flux,
                {
                    "long_name": "Total heat flux the land surface was forced with",
                    "units": "W m-2",
                    "positive": "upward",
                    **role_attrs("forcing"),
                },
            ),
            # The water source of the top soil layer. Prefixed like every
            # forcing so it does not collide with the atmosphere's own
            # `precipitation` when the two datasets are merged.
            forcing_variable("precipitation"): (
                dims,
                forcing.precipitation,
                {
                    "long_name": "Precipitation the land surface was forced with",
                    "units": "kg m-2 s-1",
                    "positive": "downward",
                    **role_attrs("forcing"),
                },
            ),
            "surface_albedo": (
                dims,
                derived.surface_albedo,
                {
                    "long_name": "Surface albedo",
                    "units": "None",
                    "description": "The climatology when surface_albedo_from_clim_file is set; otherwise baresoil_albedo * (1 - vegetation_fraction) + vegetation_albedo * vegetation_fraction",
                    **role_attrs("derived"),
                },
            ),
            "deep_drainage": (
                dims,
                derived.deep_drainage,
                {
                    "long_name": "Water drained out of the deep soil layer",
                    "units": "kg m-2 s-1",
                    **role_attrs("derived"),
                },
            ),
            "soil_water_overflow": (
                dims,
                derived.soil_water_overflow,
                {
                    "long_name": "Soil water above field capacity leaving the soil layers",
                    "units": "kg m-2 s-1",
                    **role_attrs("derived"),
                },
            ),
            "runoff_to_ocean": (
                dims,
                derived.runoff_to_ocean,
                {
                    "long_name": "Deep drainage and soil water overflow routed to the ocean",
                    "units": "kg m-2 s-1",
                    **role_attrs("derived"),
                },
            ),
            "unmet_evaporation": (
                dims,
                derived.unmet_evaporation,
                {
                    "long_name": "Evaporation the surface soil layer had no water to supply",
                    "units": "kg m-2 s-1",
                    "positive": "upward",
                    **role_attrs("derived"),
                },
            ),
            forcing_variable("evaporation"): (
                dims,
                forcing.evaporation,
                {
                    "long_name": "Evaporation the land surface was forced with",
                    "units": "kg m-2 s-1",
                    "positive": "upward",
                    **role_attrs("forcing"),
                },
            ),
            "soil_water_availability": (
                dims,
                derived.soil_water_availability,
                {
                    "long_name": "Soil water availability",
                    "units": "1",
                    **role_attrs("derived"),
                },
            ),

        }

    def to_xarray(self, diagnostics: Diagnostics, time: TimeAxis) -> xr.Dataset:
        """Convert a run's stacked diagnostics to a Dataset.

        As :meth:`SlabModelBase.to_xarray`, with the glacier fraction added
        as a time-independent coordinate next to the land-sea mask.

        The map written is ``self.params.glacier_fraction``, the one the
        model was constructed with (and read from the file, if
        ``glacier_fraction_from_clim_file=True``). A run started with
        ``initialize(params)`` whose glacier fraction differs is not
        reflected here.
        """
        dataset = super().to_xarray(diagnostics, time)
        glacier_fraction = np.broadcast_to(
            np.asarray(self.params.glacier_fraction, dtype=float), self.grid.shape
        )
        return dataset.assign_coords(
            glacier_fraction=(
                self.grid.dims,
                glacier_fraction,
                {"long_name": "Glacier fraction of the land area", "units": "1"},
            )
        )

    def _create_xarray_global_attributes(self) -> dict[str, Any]:
        # No parameter values here: the dynamics read `carry["params"]`, which
        # an experiment may replace or differentiate, while `self.params` is
        # only what the model was constructed with. Metadata that can
        # disagree with the run it describes is worse than none; the carry
        # that produced the output holds the parameters actually used.
        return {"description": "SPEEDY-based slab land surface model output"}


def convert_snow_depth_to_snow_cover(snow_depth: jnp.ndarray, params: SlabLandParameters) -> jnp.ndarray:
    """Snow cover fraction from snow depth (SPEEDY's ``sd2sc`` closure).

    Parameters
    ----------
    snow_depth : jnp.ndarray
        Snow depth (mm water equivalent).
    params : SlabLandParameters
        Supplies ``snow_depth_to_cover_scale`` (mm).

    Returns
    -------
    jnp.ndarray
        ``min(1, snow_depth / snow_depth_to_cover_scale)``.

    """
    return jnp.minimum(1.0, snow_depth / params.snow_depth_to_cover_scale)
