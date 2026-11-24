"""Slab land: a thermal slab, snow, a soil water bucket and an albedo.

The land component for a run whose land is JAX-ESM's own rather than the
atmosphere's built-in one (SPEEDY's slab land, ECHAM's SLIM). It has these
parts:

- Surface temperature: a one-layer slab heated by the surface heat flux. It
  either runs freely or relaxes to a climatology (SPEEDY's scheme).
- Snow: snow depth is either read from a climatology or held at its initial
  value. Snow cover is diagnosed from the depth.
- Soil water: the two-layer leaky bucket of
  :mod:`~jem.components.slab.bucket_land_model`, whose pure functions this
  model calls, so the soil water is one implementation however the land's
  temperature is evolved.
- Soil water availability: SPEEDY's ``soilw_am``, diagnosed from the bucket
  or read from a climatology.
- Vegetation fraction: a uniform value, or read from a climatology. Nothing
  evolves it yet.
- Surface albedo: bare soil and vegetation albedos blended by the
  vegetation fraction, or read from a climatology. It is output only for
  now.

The slab heat capacity mixes soil and glacier ice in proportion to
``glacier_fraction``, a single value or a time-invariant map read from the
climatology file.

The model was written by Tien-Yiao Hsu.
"""

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
from jem.components.slab.bucket_land_model.bucket_land_model import (
    initial_soil_water,
    soil_water_availability,
    step_soil_water,
    validate_bucket_parameters,
)
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_land_model.params import SlabLandParameters
from jem.utils.cycles import evaluate_cyclic_linear

logger = logging.getLogger(__name__)

_MONTHS_PER_YEAR = 12

#: Units a snow depth climatology file may be written in; see
#: ``SlabLandModel(snow_depth_units_in_clim_file=...)``.
SNOW_DEPTH_UNITS = ("mm", "cover")

#: The climatologies a land file may carry, as the names the constructor's
#: ``*_from_clim_file`` flags and ``*_varname_in_clim_file`` arguments use.
CLIMATOLOGY_NAMES = (
    "land_surface_temperature",
    "snow_depth",
    "soil_water_availability",
    "surface_albedo",
    "vegetation_fraction",
)

#: The climatologies ``initialize_from_clim_file`` starts the state from.
INITIAL_CONDITION_CLIMATOLOGIES = (
    "land_surface_temperature",
    "snow_depth",
    "soil_water_availability",
)


@tree_math.struct
class LandState:
    """What the land evolves.

    Attributes
    ----------
    land_surface_temperature : (n_lon, n_lat) array
        Surface temperature (K). Set to ``MASKED_SURFACE_TEMPERATURE`` on
        cells with no land.
    snow_depth : (n_lon, n_lat) array
        Snow depth (mm water equivalent). Zero where there is no land.
    snow_cover : (n_lon, n_lat) array
        Snow cover fraction in [0, 1], diagnosed from ``snow_depth``.
    soil_water_content : (n_lon, n_lat, 2) array
        Volumetric soil water content (m3 m-3) of the surface and deep layer.
        Zero where there is no land.
    vegetation_fraction : (n_lon, n_lat) array
        Fraction of the land covered by vegetation (dimensionless, in
        [0, 1]). It is kept as state for future vegetation dynamics, but
        nothing evolves it yet: it starts from ``params.vegetation_fraction``
        (or the climatology at the start month with
        ``vegetation_fraction_from_clim_file=True``) and then is the
        climatology at the current time, or held. Zero where there is no
        land.

    """

    land_surface_temperature: jnp.ndarray
    snow_depth: jnp.ndarray
    snow_cover: jnp.ndarray
    soil_water_content: jnp.ndarray
    vegetation_fraction: jnp.ndarray


@tree_math.struct
class LandForcing:
    """What the atmosphere hands the land, filled by an exchanger.

    The heat flux and the evaporation are the atmosphere's *land tile*: per
    unit of land area, over the land part of the cell -- not the grid-cell
    mean, which would charge the soil with the sea's share of a coast cell.

    Attributes
    ----------
    total_heat_flux : (n_lon, n_lat) array
        Net surface heat flux over the land part of the cell (W m-2),
        positive upward (out of the land).
    precipitation : (n_lon, n_lat) array
        Precipitation (kg m-2 s-1, positive down). One value per cell: the
        atmosphere's precipitation falls on land and sea alike.
    evaporation : (n_lon, n_lat) array
        Evaporation over the land part of the cell (kg m-2 s-1, positive up;
        negative is dew).

    """

    total_heat_flux: jnp.ndarray
    precipitation: jnp.ndarray
    evaporation: jnp.ndarray

    @classmethod
    def zeros(cls, shape):
        return cls(jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))


@tree_math.struct
class LandDerived:
    """What the land diagnoses each step, for the atmosphere, the ocean and output.

    The soil water's fields are those of
    :class:`~jem.components.slab.bucket_land_model.BucketLandDerived`, with
    the same names and meanings, so an exchange that serves a bucket land
    serves this land too. Every flux is per unit of land area (kg m-2 s-1)
    and zero where the cell has no land.

    Attributes
    ----------
    surface_albedo : (n_lon, n_lat) array
        Surface albedo (dimensionless, in [0, 1]). The climatology when
        ``surface_albedo_from_clim_file=True``; otherwise
        ``baresoil_albedo * (1 - vegetation_fraction) + vegetation_albedo *
        vegetation_fraction``. It is written to the output, but nothing in the
        model or the coupling reads it yet.
    soil_water_availability : (n_lon, n_lat) array
        SPEEDY's ``soilw_am``: the water the two layers can give, as a
        fraction of what they hold between wilting point and field capacity,
        ``(d1 max(W1 - wp1, 0) + veg d2 max(W2 - wp2, 0)) / (d1 (fc1 - wp1) +
        d2 (fc2 - wp2))``, clipped to [0, 1], from the soil water at the end
        of the step and this step's vegetation fraction. The deep layer
        counts only through vegetation, whose roots reach it. With
        ``soil_water_availability_from_clim_file=True`` it is the climatology
        at the current time instead, also clipped to [0, 1].
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
        evaporated it anyway, so it is water the coupled system created.
    runoff_mass_rate : () array
        The runoff over the whole grid (kg s-1): ``sum(area * land_fraction *
        runoff)``. An ocean that takes it spreads it over its own wet area.

    """

    surface_albedo: jnp.ndarray
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
            for name in ("surface_albedo", "soil_water_availability",
                         "deep_drainage", "soil_water_overflow", "runoff",
                         "unmet_evaporation")
        }
        return cls(**fields, runoff_mass_rate=overrides.get(
            "runoff_mass_rate", jnp.zeros(())))


class SlabLandModel(SlabModelBase):
    """Slab land: temperature, snow and a soil water bucket on the land part of a cell.

    Only cells with some land (``grid.land_mask``, ``fractional_mask > 0``)
    are integrated, forced by the atmosphere's flux over the land part of the
    cell. Other cells hold ``MASKED_SURFACE_TEMPERATURE`` and zero soil
    water, snow, vegetation and albedo.

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
        ``min(1, snow_depth / snow_depth_to_cover_scale)``. A climatology
        written as snow depth divided by the cover scale (how jax-gcm's
        packaged boundary files store ``snowc``) is read with
        ``snow_depth_units_in_clim_file="cover"``: the depth is then the
        file's value times ``params.snow_depth_to_cover_scale``.

    Soil water
        The bucket of :class:`~jem.components.slab.bucket_land_model.BucketLandModel`
        (:func:`~jem.components.slab.bucket_land_model.step_soil_water`; see
        there for the processes and their order), forced by the
        precipitation and by the evaporation over the land part of the cell,
        with this model's vegetation fraction (a field of its state) as the
        deep layer's vegetation weighting. Water is conserved to rounding,
        per unit of land area. There is no snow budget, so sublimation is
        taken from the soil too.

    Soil water availability
        Diagnosed from the bucket's soil water. With
        ``soil_water_availability_from_clim_file=True`` it is the climatology
        at the current time instead. The bucket runs either way, and a
        prescribed availability does not feed back on the soil water. Once the
        atmosphere reads it, a prescribed wet availability over dry soil makes
        the atmosphere evaporate water the soil does not hold, which shows up
        as ``derived.unmet_evaporation``.

    Vegetation fraction
        ``params.vegetation_fraction`` (uniform by default) at the start. With
        ``vegetation_fraction_from_clim_file=True`` it is the climatology at
        the current time, clipped to [0, 1] and set to 0 off land, so the
        file may hold NaN or a fill value over ocean. It weights the albedo
        blend and the deep layer's share of the soil water.

    Surface albedo
        With ``surface_albedo_from_clim_file=True`` the albedo is the
        climatology at the current time. Otherwise it is the bare soil and
        vegetation albedos weighted by the vegetation fraction,
        ``baresoil_albedo * (1 - vegetation_fraction) + vegetation_albedo * vegetation_fraction``.
        Either way it is clipped to [0, 1] and set to 0 off land. It is
        written to the output, but nothing in the model or the coupling reads
        it yet.

    Initial condition
        Every land cell starts at 288.15 K with no snow, its soil water at
        ``params.initial_soil_water_fill`` of the range between wilting point
        and field capacity, and the vegetation fraction of
        ``params.vegetation_fraction``. With ``initialize_from_clim_file=True``
        the temperature and the snow start from their climatologies at the
        month the run starts in, and the soil water from the
        soil-water-availability climatology as
        :class:`~jem.components.slab.bucket_land_model.BucketLandModel` does,
        if the file carries it. The vegetation fraction starts from its
        climatology whenever ``vegetation_fraction_from_clim_file=True``,
        whatever ``initialize_from_clim_file`` is.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: SlabLandParameters | None = None,
        *,
        name: str = "lnd",
        land_clim_file: str | None = None,
        initialize_from_clim_file: bool = False,
        soil_water_availability_from_clim_file: bool = False,
        snow_depth_from_clim_file: bool = False,
        land_surface_temperature_from_clim_file: bool = False,
        surface_albedo_from_clim_file: bool = False,
        glacier_fraction_from_clim_file: bool = False,
        glacier_fraction_from_albedo: bool = False,
        vegetation_fraction_from_clim_file: bool = False,
        soil_water_availability_varname_in_clim_file: str = "soil_water_availability",
        snow_depth_varname_in_clim_file: str = "snow_depth",
        land_surface_temperature_varname_in_clim_file: str = "land_surface_temperature",
        surface_albedo_varname_in_clim_file: str = "surface_albedo",
        glacier_fraction_varname_in_clim_file: str = "glacier_fraction",
        vegetation_fraction_varname_in_clim_file: str = "vegetation_fraction",
        snow_depth_units_in_clim_file: str = "mm",
    ):
        """Initialize the land surface model.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid, the atmosphere's: the land takes the
            atmosphere's fluxes and hands back its fields without
            regridding. It has to carry cell areas
            (``SlabGrid.cell_solid_angle``) for the runoff's mass rate.
        params : SlabLandParameters, optional
            Model parameters, defaulting to
            :meth:`SlabLandParameters.default`. They are checked here, since
            inside ``step`` they are traced and cannot be inspected.
        name : str
            Component name in the coupler's workflow and carry. Default
            ``"lnd"``, the name the standard coupling wires a land under.
        land_clim_file : str, optional
            netCDF file with 12-month climatologies on the model grid. A field
            with no time dimension is used as the same map in every month. A
            field missing from the file, or every field when no file is given,
            uses a fallback: an idealized temperature climatology, no snow, a
            soil water availability of 0.5, a surface albedo of 0.2 and a
            vegetation fraction of 0.8.
        initialize_from_clim_file : bool
            Start the temperature, the snow and (if the file carries the
            availability) the soil water from their climatologies at the
            month the run starts in. :meth:`initialize` takes the same
            argument, which wins when given. Default False.
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
        glacier_fraction_from_albedo : bool
            Derive the glacier map from the background albedo in
            ``land_clim_file`` (the variable ``surface_albedo_varname_in_clim_file``
            names, averaged over the year): a land cell at or above
            ``params.glacier_albedo_threshold`` is all glacier and the rest
            all soil. This is SPEEDY's own test for land ice, for a boundary
            file that carries a background albedo and no glacier map
            (jax-gcm's packaged ones). The map becomes
            ``params.glacier_fraction``. Default False.
        vegetation_fraction_from_clim_file : bool
            Read the vegetation fraction from the climatology instead of
            using ``params.vegetation_fraction``. The initial state starts
            from the climatology too. Default False.
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
        vegetation_fraction_varname_in_clim_file : str
            Name of the vegetation fraction variable in ``land_clim_file``.
        snow_depth_units_in_clim_file : {"mm", "cover"}
            What the snow depth variable holds: ``"mm"``, snow depth in mm of
            water, or ``"cover"``, snow depth divided by
            ``params.snow_depth_to_cover_scale`` (jax-gcm's packaged boundary
            files store their ``snowc`` so; ``min(1, value)`` is the cover
            fraction). The depth is then the file's value times the scale,
            evaluated with the parameters in the carry, so it follows the
            parameter.

        Raises
        ------
        ValueError
            If a parameter is out of range or is a map off the model grid,
            the grid carries no cell areas, ``snow_depth_units_in_clim_file``
            is not one of its two values, a field in ``land_clim_file`` is
            not on the model grid or has NaNs over land where the model uses
            it, ``glacier_fraction_from_clim_file`` is set without a
            ``land_clim_file`` or with a glacier fraction that varies in time,
            or ``glacier_fraction_from_albedo`` is set together with it, or
            without a ``land_clim_file`` that carries the albedo.
        FileNotFoundError
            If ``land_clim_file`` does not exist.

        """
        super().__init__(name=name, grid=grid)
        if snow_depth_units_in_clim_file not in SNOW_DEPTH_UNITS:
            raise ValueError(
                f"snow_depth_units_in_clim_file must be one of {SNOW_DEPTH_UNITS!r}; "
                f"got {snow_depth_units_in_clim_file!r}."
            )
        if grid.cell_solid_angle is None:
            raise ValueError(
                f"{name}: the grid carries no cell areas (SlabGrid.cell_solid_angle), "
                "which the runoff's mass rate needs. Build it with "
                "SlabGrid.from_coords or from a SCRIP file with grid_area."
            )
        self.params = SlabLandParameters.default() if params is None else params
        self.land_clim_file = land_clim_file
        self.initialize_from_clim_file = initialize_from_clim_file
        self.snow_depth_units_in_clim_file = snow_depth_units_in_clim_file
        self.varnames_in_clim_file = {
            "snow_depth": snow_depth_varname_in_clim_file,
            "soil_water_availability": soil_water_availability_varname_in_clim_file,
            "land_surface_temperature": land_surface_temperature_varname_in_clim_file,
            "surface_albedo": surface_albedo_varname_in_clim_file,
            "glacier_fraction": glacier_fraction_varname_in_clim_file,
            "vegetation_fraction": vegetation_fraction_varname_in_clim_file,
        }
        # Read at trace time to pick a branch: static, never a traced value.
        self.read_from_clim_file = {
            "snow_depth": snow_depth_from_clim_file,
            "soil_water_availability": soil_water_availability_from_clim_file,
            "land_surface_temperature": land_surface_temperature_from_clim_file,
            "surface_albedo": surface_albedo_from_clim_file,
            "glacier_fraction": glacier_fraction_from_clim_file,
            "vegetation_fraction": vegetation_fraction_from_clim_file,
        }

        # The glacier map goes into the parameters rather than onto the model,
        # so the heat capacity has one source and the map stays a pytree leaf
        # that jax.grad can reach. It is loaded before the parameter checks
        # below, so the map is range-checked like a scalar would be.
        if glacier_fraction_from_clim_file and glacier_fraction_from_albedo:
            raise ValueError(
                "glacier_fraction_from_clim_file and glacier_fraction_from_albedo "
                "both set the glacier map; choose one."
            )
        #: Whether ``params.glacier_fraction`` is a map this model read, which
        #: parameters passed to ``initialize`` would replace.
        self._glacier_map_is_read = (
            glacier_fraction_from_clim_file or glacier_fraction_from_albedo)
        if glacier_fraction_from_clim_file:
            self.params = self.params.replace(
                glacier_fraction=self._load_static_field(
                    self.varnames_in_clim_file["glacier_fraction"])
            )
        if glacier_fraction_from_albedo:
            self.params = self.params.replace(
                glacier_fraction=self._glacier_fraction_from_albedo())
        self._validate_parameters()

        # Boundary data is *configuration*, so it is read here rather than in
        # ``initialize()``: that keeps initialize() pure with respect to self
        # and surfaces a bad file at construction.
        self._soil_water_availability_in_file = False
        (
            self.land_surface_temperature_climatology,
            self.snow_depth_climatology,
            self.soil_water_availability_climatology,
            self.surface_albedo_climatology,
            self.vegetation_fraction_climatology,
        ) = self._load_climatologies()
        self._log_modes()

    def _validate_parameters(self) -> None:
        """Refuse parameters the model cannot integrate, naming the field.

        The parameters are validated at construction because inside ``step``
        they are traced values that cannot be inspected. A caller who
        replaces ``carry["params"]`` afterwards takes on that responsibility.
        """
        params = self.params
        # The soil water's own parameters are the bucket's, checked by the
        # bucket's own validation.
        validate_bucket_parameters(params)

        # Both timescales are read as a ratio against a length -- the relaxation
        # as `temperature_relaxation_timescale / dt`, the snow cover as
        # `snow_depth / scale` -- and an infinite one is silent where a zero
        # one is not: it damps the land temperature by `inf / (1 + inf)`, which
        # is NaN, and it puts the snow cover at zero everywhere however deep
        # the snow. Finiteness is therefore required as well as sign (the
        # fractions below get it from being bounded on both sides).
        temperature_relaxation_timescale = float(params.temperature_relaxation_timescale)
        if (not math.isfinite(temperature_relaxation_timescale)
                or temperature_relaxation_timescale <= 0.0):
            raise ValueError(
                "params.temperature_relaxation_timescale must be a finite positive number of seconds; "
                f"got {temperature_relaxation_timescale!r}."
            )
        snow_depth_to_cover_scale = float(params.snow_depth_to_cover_scale)
        if not math.isfinite(snow_depth_to_cover_scale) or snow_depth_to_cover_scale <= 0.0:
            raise ValueError(
                "params.snow_depth_to_cover_scale must be a finite positive snow "
                f"depth in mm; got {snow_depth_to_cover_scale!r}."
            )
        # The albedos and the glacier fraction are fractions. A NaN would
        # propagate silently: through the heat capacity into the temperature
        # for the glacier fraction, and into the output for the albedo.
        for fraction_name in ("baresoil_albedo", "vegetation_albedo", "glacier_fraction",
                              "glacier_albedo_threshold"):
            fraction = np.asarray(getattr(params, fraction_name), dtype=float)
            if not np.all(np.isfinite(fraction)) or not np.all(
                (fraction >= 0.0) & (fraction <= 1.0)
            ):
                raise ValueError(
                    f"params.{fraction_name} must be finite and in [0, 1]; "
                    f"got values in [{np.min(fraction):g}, {np.max(fraction):g}]."
                )
        # A map must be on the model grid: anything else would broadcast or
        # fail deep inside `step`.
        for map_name in ("glacier_fraction", "baresoil_albedo", "vegetation_albedo",
                         "vegetation_fraction"):
            field_shape = np.shape(getattr(params, map_name))
            if field_shape not in ((), self.grid.shape):
                raise ValueError(
                    f"params.{map_name} has shape {field_shape}; it must "
                    f"be a scalar or a map on the grid, {self.grid.shape} (n_lon, n_lat)."
                )
        # The slab heat capacity is depth times volumetric capacity; a zero or
        # negative factor makes the temperature update infinite or unstable.
        for factor_name in (
            "soil_thermal_depth",
            "glacier_depth",
            "soil_volumetric_heat_capacity",
            "glacier_volumetric_heat_capacity",
        ):
            factor = np.asarray(getattr(params, factor_name), dtype=float)
            if not np.all(np.isfinite(factor)) or not np.all(factor > 0.0):
                raise ValueError(
                    f"params.{factor_name} must be finite and strictly positive "
                    f"(it divides the heat budget); got {factor!r}."
                )

    def _log_modes(self) -> None:
        """Say once, at construction, which mode each part of the land runs in.

        Logged here and not in ``step``: a log call inside a traced step fires
        once at trace time, which reads as if it described every step.
        """
        for part in CLIMATOLOGY_NAMES:
            if part == "land_surface_temperature":
                mode = ("relaxed to the climatology"
                        if self.read_from_clim_file[part] else "free-running")
            elif part == "soil_water_availability":
                mode = ("read from the climatology"
                        if self.read_from_clim_file[part] else "diagnosed from the bucket")
            elif part == "snow_depth":
                mode = ("read from the climatology"
                        if self.read_from_clim_file[part] else "held at its initial value")
            elif part == "vegetation_fraction":
                mode = ("read from the climatology"
                        if self.read_from_clim_file[part] else "held at params.vegetation_fraction")
            else:
                mode = ("read from the climatology"
                        if self.read_from_clim_file[part]
                        else "blended from the bare soil and vegetation albedos")
            logger.info("%s: %s is %s.", self.name, part.replace("_", " "), mode)
        logger.info(
            "%s: the run starts from %s.", self.name,
            "the climatologies" if self.initialize_from_clim_file else "uniform values",
        )

    def _load_climatologies(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Load the five monthly climatologies, with idealized fallbacks.

        Fields are read by name through
        :func:`~jem.components.slab.base.load_monthly_climatology`, which
        checks them against the model grid. A field the model uses (a
        ``*_from_clim_file`` flag or the initial condition reads it) must not
        have NaNs over land.

        Returns
        -------
        tuple of jnp.ndarray
            Surface temperature (K), snow depth (in the file's units, see
            ``snow_depth_units_in_clim_file``), soil water availability,
            surface albedo and vegetation fraction, each of shape
            ``(n_lon, n_lat, 12)``.

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
                # Every month holds the bare soil albedo, which may be a map.
                jnp.broadcast_to(
                    jnp.broadcast_to(
                        jnp.asarray(self.params.baresoil_albedo, dtype=float), shape
                    )[..., None],
                    monthly,
                ),
                jnp.full(monthly, 0.8),
            )

        if not Path(self.land_clim_file).exists():
            raise FileNotFoundError(
                f"Land climatology file \"{self.land_clim_file!s:s}\" does not exist."
            )

        fallbacks: dict[str, tuple[Callable[[], jnp.ndarray], str]] = {
            "land_surface_temperature": (
                self._idealized_land_temperature, "an idealized surface-temperature climatology"),
            "snow_depth": (lambda: jnp.zeros(monthly), "no snow"),
            "soil_water_availability": (
                lambda: jnp.full(monthly, 0.5), "a uniform soil water availability of 0.5"),
            "surface_albedo": (
                lambda: jnp.full(monthly, 0.2), "a uniform surface albedo of 0.2"),
            "vegetation_fraction": (
                lambda: jnp.full(monthly, 0.8), "a uniform vegetation fraction of 0.8"),
        }
        loaded = {}
        for part, (fallback, description) in fallbacks.items():
            variable = self.varnames_in_clim_file[part]
            # What the model reads from this climatology: a part it was asked
            # to prescribe, or one the initial condition starts from.
            used = self.read_from_clim_file[part] or (
                self.initialize_from_clim_file and part in INITIAL_CONDITION_CLIMATOLOGIES)
            if first_present_variable(self.land_clim_file, (variable,)) is None:
                # A missing field is worth a warning only where the fallback
                # stands in for something the model was told to read.
                logger.log(
                    logging.WARNING if used else logging.DEBUG,
                    "%s: land climatology file %s does not have %s; using %s.",
                    self.name, self.land_clim_file, variable, description,
                )
                loaded[part] = fallback()
                continue
            logger.info(
                "%s: loading %r climatology from %s", self.name, variable, self.land_clim_file
            )
            climatology = load_monthly_climatology(self.land_clim_file, variable, self.grid)
            if part == "soil_water_availability":
                self._soil_water_availability_in_file = True
            if used and bool(jnp.any(jnp.isnan(climatology) & self.grid.land_mask[..., None])):
                raise ValueError(
                    f"Land climatology \"{self.land_clim_file!s:s}\": \"{variable}\" has "
                    "NaNs over land points of this grid: the file's land mask and the "
                    "grid's disagree."
                )
            loaded[part] = climatology
        return tuple(loaded[part] for part in CLIMATOLOGY_NAMES)  # type: ignore[return-value]

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
        land = np.asarray(self.grid.land_mask)
        if not np.all(np.isfinite(values[land])):
            raise ValueError(
                f"Land climatology file \"{self.land_clim_file!s:s}\": "
                f"\"{varname}\" has non-finite values on land cells."
            )
        return jnp.asarray(np.where(land, values, off_land_value), dtype=float)

    def _glacier_fraction_from_albedo(self) -> jnp.ndarray:
        """Return the glacier map SPEEDY's land-ice test gives the file's albedo.

        One where the annual-mean background albedo of a land cell is at or
        above ``params.glacier_albedo_threshold``, zero elsewhere. The
        threshold is read here, once: it decides a map, not a term of the
        step.

        Returns
        -------
        jnp.ndarray
            The glacier fraction with shape ``(n_lon, n_lat)``, 0 or 1.

        """
        variable = self.varnames_in_clim_file["surface_albedo"]
        if self.land_clim_file is None:
            raise ValueError(
                "glacier_fraction_from_albedo needs a land_clim_file carrying the "
                f"background albedo \"{variable}\"."
            )
        if not Path(self.land_clim_file).exists():
            raise FileNotFoundError(
                f"Land climatology file \"{self.land_clim_file!s:s}\" does not exist."
            )
        if first_present_variable(self.land_clim_file, (variable,)) is None:
            raise ValueError(
                f"glacier_fraction_from_albedo: land climatology file "
                f"\"{self.land_clim_file!s:s}\" does not have \"{variable}\" "
                "(surface_albedo_varname_in_clim_file names the variable)."
            )
        albedo = np.asarray(
            load_monthly_climatology(self.land_clim_file, variable, self.grid),
            dtype=float).mean(axis=-1)
        land = np.asarray(self.grid.land_mask)
        if not np.all(np.isfinite(albedo[land])):
            raise ValueError(
                f"Land climatology file \"{self.land_clim_file!s:s}\": "
                f"\"{variable}\" has non-finite values on land cells."
            )
        threshold = float(self.params.glacier_albedo_threshold)
        glacier = land & (np.where(land, albedo, 0.0) >= threshold)
        logger.info(
            "%s: %d of %d land cells are glacier (background albedo %r >= %g).",
            self.name, int(glacier.sum()), int(land.sum()), variable, threshold,
        )
        return jnp.asarray(glacier, dtype=float)

    def _snow_depth(self, file_values: jnp.ndarray, params: SlabLandParameters) -> jnp.ndarray:
        """Turn the snow climatology's values into a snow depth (mm).

        The scale is a parameter, so a file written as depth over the cover
        scale is converted here, where the parameters in the carry are in
        hand, and not once at construction.
        """
        if self.snow_depth_units_in_clim_file == "cover":
            return file_values * params.snow_depth_to_cover_scale
        return file_values

    def _vegetation_fraction(
        self, params: SlabLandParameters, year_fraction, held=None
    ) -> jnp.ndarray:
        """Return the vegetation fraction: the climatology, else ``held``, else the parameter.

        Clipped to [0, 1] and zero off land.
        """
        if self.read_from_clim_file["vegetation_fraction"]:
            vegetation = evaluate_cyclic_linear(year_fraction, self.vegetation_fraction_climatology)
        elif held is not None:
            vegetation = held
        else:
            vegetation = jnp.broadcast_to(
                jnp.asarray(params.vegetation_fraction, dtype=float), self.grid.shape)
        return jnp.where(self.grid.land_mask, jnp.clip(vegetation, 0.0, 1.0), 0.0)

    def _surface_albedo(
        self, params: SlabLandParameters, vegetation_fraction: jnp.ndarray, year_fraction
    ) -> jnp.ndarray:
        """Return the surface albedo: the climatology, else the blend; clipped, zero off land."""
        if self.read_from_clim_file["surface_albedo"]:
            albedo = evaluate_cyclic_linear(year_fraction, self.surface_albedo_climatology)
        else:
            albedo = (params.baresoil_albedo * (1.0 - vegetation_fraction)
                      + params.vegetation_albedo * vegetation_fraction)
        return jnp.where(self.grid.land_mask, jnp.clip(albedo, 0.0, 1.0), 0.0)

    def _soil_water_availability(
        self, diagnosed: jnp.ndarray, year_fraction
    ) -> jnp.ndarray:
        """Return the availability the land reports: the climatology, else ``diagnosed``."""
        if self.read_from_clim_file["soil_water_availability"]:
            availability = jnp.clip(evaluate_cyclic_linear(
                year_fraction, self.soil_water_availability_climatology), 0.0, 1.0)
            return jnp.where(self.grid.land_mask, availability, 0.0)
        return diagnosed

    def initialize(
        self,
        params: SlabLandParameters | None = None,
        initialize_from_clim_file: bool | None = None,
    ) -> Carry:
        """Build the initial land carry; see the class docstring.

        Parameters
        ----------
        params : SlabLandParameters, optional
            Parameters to put in ``carry["params"]``. Defaults to the ones
            the model was constructed with. ``initial_soil_water_fill`` and
            ``vegetation_fraction`` are read here and only here.
        initialize_from_clim_file : bool, optional
            Start the temperature, the snow and the soil water from their
            climatologies at the month the run starts in
            (:attr:`~jem.components.slab.base.SlabModelBase.start_year_fraction`).
            ``None`` takes what the constructor said.

        Returns
        -------
        Carry
            ``{"params", "state", "forcing", "derived"}``. The forcing starts
            at zero and is filled by the exchangers before the first step.

        """
        if initialize_from_clim_file is None:
            initialize_from_clim_file = self.initialize_from_clim_file
        if (
            params is not None
            and self._glacier_map_is_read
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
        land = self.grid.land_mask
        shape = self.grid.shape
        # The month the run starts in; the coupler sets it through `bind`.
        year_fraction = self.start_year_fraction

        vegetation = self._vegetation_fraction(params, year_fraction)
        use_availability = initialize_from_clim_file and self._soil_water_availability_in_file
        soil_water = initial_soil_water(
            land, params, vegetation,
            self.soil_water_availability_climatology if use_availability else None,
            year_fraction,
        )

        # An unforced start is 15 degC, which is also the value the masked
        # cells report.
        temperature = jnp.full(shape, MASKED_SURFACE_TEMPERATURE)
        snow_depth = jnp.zeros(shape)
        if initialize_from_clim_file:
            temperature = jnp.where(
                land,
                evaluate_cyclic_linear(year_fraction, self.land_surface_temperature_climatology),
                MASKED_SURFACE_TEMPERATURE,
            )
            snow_depth = jnp.where(
                land,
                self._snow_depth(
                    evaluate_cyclic_linear(year_fraction, self.snow_depth_climatology), params),
                0.0,
            )

        state = LandState(  # type: ignore[call-arg]
            land_surface_temperature=temperature,
            snow_depth=snow_depth,
            snow_cover=_snow_cover(snow_depth, params),
            soil_water_content=soil_water,
            vegetation_fraction=vegetation,
        )
        derived = LandDerived.zeros(
            shape,
            surface_albedo=self._surface_albedo(params, vegetation, year_fraction),
            soil_water_availability=self._soil_water_availability(
                jnp.where(land, soil_water_availability(soil_water, params, vegetation), 0.0),
                year_fraction,
            ),
        )
        return {
            "params": params,
            "state": state,
            "forcing": LandForcing.zeros(shape),
            "derived": derived,
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the land by one coupling step.

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
        land = self.grid.land_mask
        year_fraction = time.year_fraction

        # The thermal slab has its own depth, separate from the water bucket's
        # layers: SPEEDY also uses 1 m for heat (hcapl) and 7 cm for soil water.
        heat_capacity = (
            params.soil_thermal_depth * params.soil_volumetric_heat_capacity
            * (1.0 - params.glacier_fraction)
            + params.glacier_depth * params.glacier_volumetric_heat_capacity
            * params.glacier_fraction
        )
        dt_inverse_heat_capacity = time.dt / heat_capacity

        # The heat flux arrives in the coupler's upward-positive convention;
        # the land is warmed by what flows into it, hence the negation.
        downward_heat_flux = -forcing.total_heat_flux

        if self.read_from_clim_file["land_surface_temperature"]:
            climatology_begin = evaluate_cyclic_linear(
                year_fraction, self.land_surface_temperature_climatology)
            climatology_end = evaluate_cyclic_linear(
                time.end_of_step().year_fraction, self.land_surface_temperature_climatology)
            dissipation_steps = params.temperature_relaxation_timescale / time.dt
            dissipation = dissipation_steps / (1.0 + dissipation_steps)
            anomaly = state.land_surface_temperature - climatology_begin
            new_anomaly = dissipation * (anomaly + dt_inverse_heat_capacity * downward_heat_flux)
            new_temperature = new_anomaly + climatology_end
        else:
            new_temperature = (
                state.land_surface_temperature + dt_inverse_heat_capacity * downward_heat_flux)
        new_temperature = jnp.where(land, new_temperature, MASKED_SURFACE_TEMPERATURE)

        if self.read_from_clim_file["snow_depth"]:
            new_snow_depth = jnp.where(
                land,
                self._snow_depth(
                    evaluate_cyclic_linear(year_fraction, self.snow_depth_climatology), params),
                0.0,
            )
        else:
            new_snow_depth = state.snow_depth

        new_vegetation = self._vegetation_fraction(
            params, year_fraction, held=state.vegetation_fraction)

        # The soil water is the bucket's, weighted by this step's vegetation.
        new_water, water = step_soil_water(
            state.soil_water_content, forcing.precipitation, forcing.evaporation,
            time.dt, params, new_vegetation, self.grid)

        new_state = LandState(  # type: ignore[call-arg]
            land_surface_temperature=new_temperature,
            snow_depth=new_snow_depth,
            snow_cover=_snow_cover(new_snow_depth, params),
            soil_water_content=new_water,
            vegetation_fraction=new_vegetation,
        )
        derived = LandDerived(  # type: ignore[call-arg]
            surface_albedo=self._surface_albedo(params, new_vegetation, year_fraction),
            soil_water_availability=self._soil_water_availability(
                water.soil_water_availability, year_fraction),
            deep_drainage=water.deep_drainage,
            soil_water_overflow=water.soil_water_overflow,
            runoff=water.runoff,
            unmet_evaporation=water.unmet_evaporation,
            runoff_mass_rate=water.runoff_mass_rate,
        )
        diagnostics = {"state": new_state, "forcing": forcing, "derived": derived}
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
        lat = jnp.asarray(self.grid.latitude_radian, dtype=float)  # (n_lon, n_lat)
        months = jnp.arange(_MONTHS_PER_YEAR)

        # Warm equator, cold poles.
        base_temperature = 273.15 + 25.0 * jnp.cos(lat)

        # Seasonal amplitude is largest at high latitudes, zero at the equator.
        seasonal_amplitude = 15.0 * jnp.sin(jnp.abs(lat)) ** 2

        # Peak in March (month index 2).
        phase = 2 * jnp.pi * (months - 2) / _MONTHS_PER_YEAR

        return (
            base_temperature[..., None]
            + seasonal_amplitude[..., None] * jnp.cos(phase)[None, None, :]
        )

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for the land output."""
        state = diagnostics["state"]
        forcing = diagnostics["forcing"]
        derived = diagnostics["derived"]
        dims = ("time",) + self.grid.dims
        flux = {"units": "kg m-2 s-1"}

        def per_land_area(name, long_name, **attrs):
            return (dims, getattr(derived, name), {
                "long_name": long_name, **flux,
                "comment": "per unit land area; zero where the cell has no land",
                **attrs, **role_attrs("derived")})

        return {
            "land_surface_temperature": (
                dims, state.land_surface_temperature, {
                    "long_name": "Land surface temperature", "units": "K",
                    **role_attrs("state")}),
            "snow_depth": (
                dims, state.snow_depth, {
                    "long_name": "Snow depth (water equivalent)", "units": "mm",
                    **role_attrs("state")}),
            "snow_cover": (
                dims, state.snow_cover, {
                    "long_name": "Snow cover fraction of the land", "units": "1",
                    **role_attrs("state")}),
            # Layer 0 is the surface soil layer, layer 1 the deep one; their
            # depths are params.soil_bucket_depth.
            "soil_water_content": (
                dims + ("soil_layer",), state.soil_water_content, {
                    "long_name": "Volumetric soil water content of the surface "
                                 "and deep layer",
                    "units": "m3 m-3", **role_attrs("state")}),
            "vegetation_fraction": (
                dims, state.vegetation_fraction, {
                    "long_name": "Vegetation fraction of the land", "units": "1",
                    **role_attrs("state")}),
            "surface_albedo": (
                dims, derived.surface_albedo, {
                    "long_name": "Surface albedo", "units": "1",
                    "description": "The climatology when surface_albedo_from_clim_file "
                                   "is set; otherwise the bare soil and vegetation albedos "
                                   "blended by the vegetation fraction",
                    **role_attrs("derived")}),
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
            # Written under the land tile's own names, the ones the atmosphere
            # publishes them by: a sea ice that solves its own energy balance
            # is forced by the same two fluxes over the sea part, and a merge
            # of the two datasets would collide on a name they shared.
            forcing_variable("land_total_heat_flux"): (
                dims, forcing.total_heat_flux, {
                    "long_name": "Net surface heat flux over the land part of the "
                                 "cell the land was forced with",
                    "units": "W m-2", "positive": "up", **role_attrs("forcing")}),
            forcing_variable("precipitation"): (
                dims, forcing.precipitation, {
                    "long_name": "Precipitation the land was forced with",
                    **flux, "positive": "down", **role_attrs("forcing")}),
            forcing_variable("land_evaporation"): (
                dims, forcing.evaporation, {
                    "long_name": "Evaporation over the land part of the cell "
                                 "the land was forced with",
                    **flux, "positive": "up", **role_attrs("forcing")}),
        }

    def to_xarray(self, diagnostics: Diagnostics, time: TimeAxis) -> xr.Dataset:
        """Convert a run's stacked diagnostics to a Dataset.

        As :meth:`~jem.components.slab.base.SlabModelBase.to_xarray`, with the
        glacier fraction added as a time-independent coordinate.

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


def _snow_cover(snow_depth: jnp.ndarray, params: SlabLandParameters) -> jnp.ndarray:
    """Snow cover fraction from snow depth (SPEEDY's ``sd2sc`` closure).

    ``min(1, snow_depth / snow_depth_to_cover_scale)``, with the depth in mm
    water equivalent.
    """
    return jnp.minimum(1.0, snow_depth / params.snow_depth_to_cover_scale)
