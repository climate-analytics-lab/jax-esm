"""Slab land model with a two-layer soil water bucket.

The model has three parts:

- Surface temperature: a one-layer slab heated by the surface heat flux. It
  either runs freely or relaxes to a climatology (SPEEDY's scheme).
- Snow: snow depth is either read from a climatology or held at its initial
  value. Snow cover is diagnosed from the depth.
- Soil water: a two-layer leaky bucket filled by precipitation, or the
  surface layer read from a climatology.

Each cell is soil or land ice, depending on its surface albedo.
"""

import logging
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import tree_math

from jem.base.component import Carry, CouplingTime, Diagnostics
from jem.components.slab.base import (
    MASKED_SURFACE_TEMPERATURE,
    SlabModelBase,
    first_present_variable,
    forcing_variable,
    load_monthly_climatology,
    role_attrs,
)
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_land_model.params import (
    SlabLandParameters,
)
from jem.utils.cycles import evaluate_cyclic_linear

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

    """

    land_surface_temperature: jnp.ndarray
    snow_depth: jnp.ndarray
    snow_cover: jnp.ndarray
    soil_water_content: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        land_surface_temperature=None,
        snow_depth=None,
        snow_cover=None,
        soil_water_content=None,
        n_soil_layers=2,
    ):
        return cls(
            land_surface_temperature if land_surface_temperature is not None else jnp.zeros(shape),
            snow_depth if snow_depth is not None else jnp.zeros(shape),
            snow_cover if snow_cover is not None else jnp.zeros(shape),
            soil_water_content if soil_water_content is not None else jnp.zeros(shape + (n_soil_layers,)),
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

    """

    total_heat_flux: jnp.ndarray
    precipitation: jnp.ndarray

    @classmethod
    def zeros(cls, shape, total_heat_flux=None, precipitation=None):
        return cls(
            total_heat_flux if total_heat_flux is not None else jnp.zeros(shape),
            precipitation if precipitation is not None else jnp.zeros(shape),
        )


class SlabLandModel(SlabModelBase):
    """Slab land model with a two-layer soil water bucket.

    Only cells with some land (``grid.binary_land_mask == 1``) are
    integrated. Other cells hold ``MASKED_SURFACE_TEMPERATURE`` and zero
    soil water.

    Surface temperature
        A single slab with heat capacity ``C``. On soil
        ``C = soil_depth[0] * soil_volumetric_heat_capacity``; on land ice
        (albedo >= ``landice_albedo_threshold``)
        ``C = landice_depth * landice_volumetric_heat_capacity``. With
        ``F`` the downward heat flux, the default update is::

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
        ``tau1``, ``tau2`` the drainage timescales and ``P`` the
        precipitation::

            dW1/dt = P / (rho_w * d1) - W1 / tau1
            dW2/dt = (d1 / d2) * W1 / tau1 - W2 / tau2

        Water leaving the surface layer enters the deep layer, and water
        leaving the deep layer is lost. Both layers are kept non-negative.
        There is no evaporation and no runoff yet.

        With ``soil_water_content_from_clim_file=True`` the surface layer is
        set to the climatology at the current time, and the deep layer keeps
        its previous value.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: SlabLandParameters | None = None,
        *,
        name: str = "lnd",
        surface_albedo: jnp.ndarray | None = None,
        land_clim_file: str | None = None,
        soil_water_content_from_clim_file: bool = False,
        snow_depth_from_clim_file: bool = False,
        land_surface_temperature_from_clim_file: bool = False,
        soil_water_content_varname_in_clim_file: str = "soil_water_content",
        snow_depth_varname_in_clim_file: str = "snow_depth",
        land_surface_temperature_varname_in_clim_file: str = "land_surface_temperature",
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
        surface_albedo : jnp.ndarray, optional
            Surface albedo on the model grid, used to tell land ice from soil.
            Defaults to a uniform ``params.surface_albedo``, which makes every
            cell soil. Nothing in JEM passes the atmosphere's albedo here yet,
            so real ice sheets are also treated as soil (jax-esm#109).
        land_clim_file : str, optional
            netCDF file with 12-month climatologies on the model grid. A field
            missing from the file, or every field when no file is given, uses
            a fallback: an idealized temperature climatology, no snow, and a
            soil water content of 0.5.
        soil_water_content_from_clim_file : bool
            Set the surface soil layer from the climatology instead of
            running the bucket. Default False.
        snow_depth_from_clim_file : bool
            Read snow depth from the climatology instead of holding it fixed.
            Default False.
        land_surface_temperature_from_clim_file : bool
            Relax the surface temperature to the climatology instead of
            letting it run freely. Default False.
        soil_water_content_varname_in_clim_file : str
            Name of the soil water variable in ``land_clim_file``.
        snow_depth_varname_in_clim_file : str
            Name of the snow depth variable in ``land_clim_file``.
        land_surface_temperature_varname_in_clim_file : str
            Name of the surface temperature variable in ``land_clim_file``.

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
        self.varnames_in_clim_file = {
            "snow_depth": snow_depth_varname_in_clim_file,
            "soil_water_content": soil_water_content_varname_in_clim_file,
            "land_surface_temperature": land_surface_temperature_varname_in_clim_file,
        }

        self.read_from_clim_file = {
            "snow_depth": snow_depth_from_clim_file,
            "soil_water_content": soil_water_content_from_clim_file,
            "land_surface_temperature": land_surface_temperature_from_clim_file,
        }

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
        # Both albedo values feed the soil/ice comparison; a NaN in either
        # compares False everywhere and silently selects one material.
        for name in ("surface_albedo", "landice_albedo_threshold"):
            albedo_value = float(getattr(self.params, name))
            if not math.isfinite(albedo_value) or not 0.0 <= albedo_value <= 1.0:
                raise ValueError(
                    f"params.{name} must be a finite albedo in [0, 1]; "
                    f"got {albedo_value!r}."
                )
        # The slab heat capacity is depth times volumetric capacity; a zero or
        # negative factor makes the temperature update infinite or unstable.
        # `soil_depth` is per-layer, so every element is checked: the layer
        # depths are also the divisors of the bucket's water budget.
        for factor_name in (
            "soil_depth",
            "landice_depth",
            "soil_volumetric_heat_capacity",
            "landice_volumetric_heat_capacity",
        ):
            factor = np.asarray(getattr(self.params, factor_name), dtype=float)
            if not np.all(np.isfinite(factor)) or not np.all(factor > 0.0):
                raise ValueError(
                    f"params.{factor_name} must be finite and strictly positive "
                    f"(it is a factor of the slab heat capacity); got {factor!r}."
                )

        # An explicit albedo field is boundary data and lives on the model;
        # without one, `step` builds a uniform field from the *carried*
        # `params.surface_albedo` each step, so the parameter stays live for
        # anyone who replaces `carry["params"]` (it is below
        # `landice_albedo_threshold` by default, so such a model is all
        # soil). Nothing in JEM routes the albedo the atmosphere's boundary
        # data already carries (`jcm.forcing.ForcingData.alb0`) to this
        # argument, so that is what every packaged configuration gets, ice
        # sheets included -- see jax-esm#109.
        self.surface_albedo: jnp.ndarray | None = None
        if surface_albedo is not None:
            surface_albedo = jnp.asarray(surface_albedo)
            if tuple(surface_albedo.shape) != self.grid.shape:
                raise ValueError(
                    f"surface_albedo has shape {tuple(surface_albedo.shape)}, but the "
                    f"grid is {self.grid.shape} (n_lon, n_lat)."
                )
            # The field only ever feeds a comparison against the ice
            # threshold, so a NaN (which compares False, selecting the ice
            # slab) or an out-of-range value would silently choose the wrong
            # heat capacity rather than fail.
            if not bool(jnp.all(jnp.isfinite(surface_albedo))):
                raise ValueError("surface_albedo contains non-finite values.")
            albedo_min = float(surface_albedo.min())
            albedo_max = float(surface_albedo.max())
            if albedo_min < 0.0 or albedo_max > 1.0:
                raise ValueError(
                    "surface_albedo must lie in [0, 1]; got values in "
                    f"[{albedo_min:g}, {albedo_max:g}]."
                )
            self.surface_albedo = surface_albedo

        # Boundary data is *configuration*, so it is read here rather than in
        # ``initialize()``: that keeps initialize() pure with respect to self
        # and surfaces a bad file at construction.
        (
            self.land_surface_temperature_climatology,
            self.snow_depth_climatology,
            self.soil_water_content_climatology,
        ) = self._load_climatologies()

    def _load_climatologies(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Load the three monthly climatologies, with idealized fallbacks.

        Fields are read by name through
        :func:`~jem.components.slab.base.load_monthly_climatology`, which
        checks them against the model grid.

        Returns
        -------
        tuple of jnp.ndarray
            Surface temperature (K), snow depth (mm) and soil water content
            (m3 m-3), each of shape ``(n_lon, n_lat, 12)``.

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

        soil_water_content = self._load_named([varnames_in_clim_file["soil_water_content"],])
        if soil_water_content is None:
            logger.warning(
                "%s: land climatology file %s does not have %s; assuming uniform soil "
                "water content of 0.5.",
                self.name,
                self.land_clim_file,
                varnames_in_clim_file["soil_water_content"],
            )
            soil_water_content = jnp.full(monthly, 0.5)

        return land_surface_temperature, snow_depth, soil_water_content

    def _load_named(self, candidates: tuple[str, ...]) -> jnp.ndarray | None:
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

        By default every cell starts at 288.15 K with no snow and a soil
        water content of 0.5 in both layers.

        Parameters
        ----------
        params : SlabLandParameters, optional
            Parameters to put in ``carry["params"]``. Defaults to the ones
            the model was constructed with.
        initialize_from_clim_file : bool
            Start the surface temperature, snow depth and snow cover from the
            climatology at the month the run starts in
            (:attr:`SlabModelBase.start_year_fraction`). Soil water still
            starts at 0.5. Default False.

        Returns
        -------
        Carry
            ``{"params", "state", "forcing"}``. The forcing starts at zero
            and is filled by the exchangers before the first step.

        """
        params = self._initial_params(params)
        land = self.grid.binary_land_mask
        # The month the run starts in; the coupler sets it through `bind`.
        cycle_position = self.start_year_fraction

        state = LandState.zeros(self.grid.shape)
        state.land_surface_temperature = state.land_surface_temperature.at[:].set(273.15 + 15.0)
        state.soil_water_content = state.soil_water_content.at[:].set(0.5)

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
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the land surface by one coupling step.

        See the class docstring for the equations. The ``*_from_clim_file``
        switches pick the branch for each part at trace time.

        Parameters
        ----------
        carry : Carry
            ``{"params", "state", "forcing"}`` as built by :meth:`initialize`.
        time : CouplingTime
            Start time and length (``dt``) of the step.

        Returns
        -------
        tuple[Carry, Diagnostics]
            The new carry, and ``{"state", "forcing"}`` for output.

        """
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]
        read_from_clim_file = self.read_from_clim_file
        land_mask = self.grid.binary_land_mask

        # Heat capacity per unit area, and its reciprocal scaled by the step:
        # SPEEDY's rhcapl. Computed here rather than cached on the model because
        # both the depths and the coupling timestep are things a caller may vary
        # between runs -- and, for the depths, differentiate through.
        surface_albedo = (
            self.surface_albedo
            if self.surface_albedo is not None
            else jnp.full(self.grid.shape, params.surface_albedo)
        )

        # The thermal slab is the top soil layer only: the temperature is a
        # single-layer prognostic, and with the default 1 m top layer this is
        # exactly SPEEDY's hcapl. The deeper layer holds water, not heat.
        heat_capacity = jnp.where(
            surface_albedo < params.landice_albedo_threshold,
            params.soil_depth[0] * params.soil_volumetric_heat_capacity,
            params.landice_depth * params.landice_volumetric_heat_capacity,
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
         
        if read_from_clim_file["soil_water_content"]:
            logger.info("Read soil water content from climatology")
            new_soil_water_content_surface_grid = evaluate_cyclic_linear(
                time.year_fraction, self.soil_water_content_climatology
            )
            new_soil_water_content = state.soil_water_content.at[:, :, 0].set(new_soil_water_content_surface_grid)

        else:
 
            # =====================================================================
            # Land surface soil moisture evolution (two-layer leaky bucket)
            # =====================================================================
            # Layer 0 (surface): receives precipitation, drains into layer 1.
            # Layer 1 (deep): receives layer 0's drainage, drains out as deep drainage.

            W1 = state.soil_water_content[..., 0]
            W2 = state.soil_water_content[..., 1]
            
            drain1 = W1 / params.drainage_timescale[0]
            inflow2 = (params.soil_depth[0] / params.soil_depth[1]) * drain1

            new_W1 = jnp.maximum(
                0.0,
                W1 + (forcing.precipitation / (1000.0 * params.soil_depth[0]) - drain1) * time.dt,
            )
            
            new_W2 = jnp.maximum(
                0.0,
                W2 + (inflow2 - W2 / params.drainage_timescale[1]) * time.dt,
            )

            new_soil_water_content = jnp.stack([new_W1, new_W2], axis=-1)

        new_soil_water_content = jnp.where(land_mask[..., None], new_soil_water_content, 0.0)

        new_state = state.replace(
            land_surface_temperature=new_land_surface_temperature,
            snow_depth=new_snow_depth,
            snow_cover=convert_snow_depth_to_snow_cover(new_snow_depth, params),
            soil_water_content=new_soil_water_content,
        )

        diagnostics = {
            "state": new_state,
            "forcing": forcing,
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
            # depths are params.soil_depth.
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
        }

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
