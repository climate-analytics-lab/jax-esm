"""Slab ocean model component."""

from pathlib import Path
from typing import Annotated, Any

import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr

from jem import constants
from jem.components.slab.base import _DEFAULT_START_DATETIME, SlabModelBase
from jem.utils import data_structure
from jem.utils.bulk_op import stack_objects
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared

default_land_surface_temperature = 288.15

@data_structure.typed_and_dimensioned
class OceanState:
    sim_time: Annotated[float, (), "zero_dimensional"]
    sea_surface_temperature: Annotated[
        float, ("longitude", "latitude"), "two_dimensional"
    ]
    mixed_layer_depth: Annotated[float, ("longitude", "latitude"), "two_dimensional"]

@data_structure.typed_and_dimensioned
class OceanForcing:
    total_heat_flux: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    q_flux: Annotated[float, ("longitude", "latitude", "month"), "two_dimensional_with_month"]

class SlabOceanModel(SlabModelBase):
    """Slab ocean model with prescribed mixed layer depth and climatology.

    This model simulates sea surface temperature evolution using a simple
    thermodynamic equation with optional relaxation to climatology.
        
    dT/dt = F_net/(rho * cp * h) + forcing


    where:
        T: sea surface temperature
        F_net: total heat flux (positive upward)
        rho: ocean density
        cp: ocean specific heat capacity
        h: mixed layer depth
        forcing: the forcing of temperature. See below for explaination
    
    (1) If `forcing_method` == "None" (or just None), then forcing = 0.

    (2) If `forcing_method` == "Qflux", then traditional Q-flux adjust, i.e., periodic forcing
        over a year, is used:
 
            forcing = Q / (rho * cp * h)

        where variable `Q` will be read from a file given in `Q_flux_file`. If `Q_flux_file`
        is not provided, then Q will be all zeros, which is possible when doing training.
    
    (3) If `forcing_method` == "relaxation", then linear relaxation will be used

            forcing = - (T - T_clim) / tau

        where tau is the relaxation timescale to climatology (can be jnp.inf), and T_clim
        is the climatology read from `SST_clim_file`. If `SST_clim_file` is not provided,
        then T_clim will be all zeros, which is possible when doing training.

    Freeze/melt potential
    ----------------------
    Following CESM's slab-ocean/CICE coupling convention: after the update above,
    `sea_surface_temperature` is clamped so it never drops below `T_freezing` (the seawater
    freezing point), and the heat that clamp removes (or, symmetrically, the heat available
    above freezing) is reported as a single signed diagnostic, `ice_frazil_melt_energy`
    (J/m^2, energy released over this coupling step -- not a flux):

        ice_frazil_melt_energy = (T_freezing - T_unclamped)
            * mixed_layer_depth * ocean_density * ocean_specific_heat_capacity

    Positive values mean the mixed layer would have gone sub-freezing -- that deficit forms
    new (frazil) ice. Negative values mean the mixed layer sits above freezing -- that surplus
    is available to melt existing ice from below. This is exactly CESM's `frzmlt`: one signed
    quantity, computed once per coupling step with no separate relaxation timescale (the
    coupling step itself is the timescale). This ocean model has no ice physics of its own, so
    `ice_frazil_melt_energy` is meant to be consumed by a sea-ice component (e.g.
    `SlabSeaiceModel`) via the coupler.
    """

    def __init__(
        self,
        grid_specification: str = "JCM::T31",
        start_datetime: jdt.Datetime = _DEFAULT_START_DATETIME,
        timestep: float = 86400.0,
        relaxation_time: float = 60 * 86400.0,
        mixed_layer_depth_min: float = 40.0,
        mixed_layer_depth_max: float = 60.0,
        topography_file: str | None = None,
        mask_file: str | None = None,
        SST_clim_file: str | None = None,
        Q_flux_file: str | None = None,
        forcing_method: str | None = None,
        initialization_sea_surface_temperature: float = 288.15,
        mask_value: float = 0.0,
        calendar: str = "365_day",
    ):
        """Initialize slab ocean model.

        Args:
            grid_specification: Grid spec string (e.g., "JCM::T31")
            start_datetime: Simulation start datetime
            timestep: Model timestep in seconds
            relaxation_time: Relaxation timescale to climatology in seconds
            mixed_layer_depth_min: Minimum mixed layer depth in meters
            mixed_layer_depth_max: Maximum mixed layer depth in meters
            topography_file: Optional path to topography NetCDF file
            mask_file: Optional path to land/ocean mask NetCDF file
            SST_clim_file: Optional path to SST climatology NetCDF file
        """
        self.relaxation_time = relaxation_time
        self.mixed_layer_depth_min = mixed_layer_depth_min
        self.mixed_layer_depth_max = mixed_layer_depth_max
        self.SST_clim_file = SST_clim_file
        self.Q_flux_file = Q_flux_file

        super().__init__(
            name="SlabOceanModel",
            grid_specification=grid_specification,
            start_datetime=start_datetime,
            timestep=timestep,
            topography_file=topography_file,
            mask_file=mask_file,
            calendar=calendar,
        )

        # Climatology data (loaded during initialize)
        self.SST_clim = None
        self.time_factor = None
        self.cd_factor = None
        self.forcing_method = forcing_method or "None"
        self.mask_value = mask_value

        self.validate()

    def validate(self):
        super().validate()
        if self.forcing_method == "None":
            # Do nothing
            pass
        elif self.forcing_method == "Qflux":
            if self.Q_flux_file is None:
                print("Notice: `Q_flux_file` is not given. Default values (zeros) will be used.")
            elif not Path(self.Q_flux_file).exists():
                raise FileNotFoundError(f"Q-flux file \"{self.Q_flux_file!s:s}\" is specified but it does not exist.")
        elif self.forcing_method == "relaxation":
            if self.SST_clim_file is None:
                print("Notice: `SST_clim_file` is not given. Default values (zeros) will be used.")
            elif not Path(self.SST_clim_file).exists():
                raise FileNotFoundError(f"SST climatology file \"{self.SST_clim_file!s:s}\" is specified but does not exist.")
            elif (self.relaxation_time < 0) or jnp.isnan(self.relaxation_time):
                raise ValueError("`relaxation_time` must be a positive number or infinity.")
        else:
            raise ValueError(f"Unknown `forcing_method` is given: \"{self.forcing_method!s:s}\" ")

    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for ocean model."""
        decorator = data_structure.build_dataclass_from_typed_and_dimensioned({"two_dimensional": self.grid_shape})
        decorator = data_structure.build_dataclass_from_typed_and_dimensioned(
            {
                "two_dimensional": self.grid_shape,
                "two_dimensional_with_month": self.grid_shape + (12,),
            }
        )
        self.component_state_class = decorator(OceanState)
        self.component_forcing_class = decorator(OceanForcing)

    def _create_variable_registries(self) -> None:
        self.state_variable_registry = {}
        self.forcing_variable_registry = {}

        for target_registry, target_class in [
            (self.state_variable_registry, self.component_state_class),
            (self.forcing_variable_registry, self.component_forcing_class),
        ]:
            for name, _, dimensions, shape in target_class.typed_and_dimensioned_info():
                target_registry[name] = (shape, dimensions)

    def initialize(self):
        """Initialize ocean model fields."""
        nonocn_idx = self.horizontal_grids["T"].bmask != self.mask_value

        # Initialize mixed layer depth with latitudinal variation
        init_mixed_layer_depth = (
            self.mixed_layer_depth_max
            + (self.mixed_layer_depth_min - self.mixed_layer_depth_max)
            * jnp.cos(self.latitude_radian) ** 3
        )

        # Load or create initial SST
        if self.SST_clim_file is not None:
            print("SST climatology file. The given initial SST will be used.")
            print("SST climatology file: ", self.SST_clim_file)
            self.SST_clim = jnp.array(xr.open_dataset(self.SST_clim_file)["sst"])
            init_sea_surface_temperature = self.SST_clim[:, :, 0].copy()
        else:
            print("Boundary does not exist. Idealized initial SST will be used.")
            init_sea_surface_temperature = (
                positive_cosine_cubic_latitude_squared(self.latitude_radian) * 10.0
                + constants.freezing_point_K
            )

        # Apply mask
        init_sea_surface_temperature = init_sea_surface_temperature.at[nonocn_idx].set(
            default_land_surface_temperature
        )

        # Validate mask consistency
        if jnp.sum(jnp.isnan(init_sea_surface_temperature)) == 0:
            print("grid.bmask and SST_clim do share the same mask.")
        else:
            raise ValueError(
                "fmask_ocn and sea_surface_temperature_init do not share the same mask."
            )

        # Set relaxation time to infinity if no climatology
        if self.SST_clim_file is None:
            print("Notice: Climaology SST does not exist. Set relaxation time to inifinity.")
            self.relaxation_time = jnp.inf

        # Compute heat capacity and time factors for Euler backward scheme
        cd = (
            constants.ocean_density
            * constants.ocean_specific_heat_capacity
            * init_mixed_layer_depth
        )

        if self.forcing_method == "relaxation":
            tau = jnp.ones_like(cd) * self.relaxation_time
        else:
            tau = jnp.inf
        
        self.time_factor = (1.0 + self.timestep / tau) ** (-1)
        self.cd_factor = self.timestep / cd
        
        forcing = self.component_forcing_class.zeros()
        forcing = forcing.copy({
            "q_flux": forcing.q_flux
        })
        return {
            "state": self.component_state_class.zeros().copy({
                "mixed_layer_depth": init_mixed_layer_depth,
                "sea_surface_temperature": init_sea_surface_temperature,
            }),
            "forcing": forcing,
            "derived": {
                "ice_frazil_melt_energy": jnp.zeros(self.grid_shape),
            },
        }

    def _create_step_function_body(self):
        """Create the step function for ocean model."""
        start_day_offset = self._compute_start_day_offset()
        ocn_idx = self.horizontal_grids["T"].bmask == self.mask_value
        nonocn_idx = self.horizontal_grids["T"].bmask != self.mask_value

        def step_function(carry, step):
            state = carry["state"]
            forcing = carry["forcing"]
            new_sea_surface_temperature_anom = state.sea_surface_temperature
            total_heat_flux = forcing.total_heat_flux
            predictions = {}
            print(f"Using method: {self.forcing_method}")
            if self.forcing_method == "relaxation":
                sst_clim_beg = jnp.where(
                    ocn_idx,
                    self._interpolate_cyclic(state.sim_time, start_day_offset, self.SST_clim),
                    default_land_surface_temperature,
                )
                sst_clim_end = jnp.where(
                    ocn_idx,
                    self._interpolate_cyclic(state.sim_time + self.timestep, start_day_offset, self.SST_clim),
                    default_land_surface_temperature,
                )
                new_sea_surface_temperature_anom = state.sea_surface_temperature - sst_clim_beg
            elif self.forcing_method == "Qflux":
                snapshot_Qflux = jnp.where(
                    ocn_idx,
                    self._interpolate_cyclic(state.sim_time, start_day_offset, forcing.q_flux),
                    0.0,
                )
                total_heat_flux = total_heat_flux + snapshot_Qflux


            # Euler backward step
            new_sim_time = state.sim_time + self.timestep
            new_sea_surface_temperature_anom = self.time_factor * (
                new_sea_surface_temperature_anom
                + self.cd_factor * (- total_heat_flux)
            )

            # Add climatology back
            new_sea_surface_temperature = new_sea_surface_temperature_anom
            if self.forcing_method == "relaxation":
                new_sea_surface_temperature += sst_clim_end
            
            # Apply land mask
            new_sea_surface_temperature = new_sea_surface_temperature.at[
                nonocn_idx
            ].set(default_land_surface_temperature)

            # Freeze/melt potential (CESM's `frzmlt`): heat surplus/deficit of the mixed
            # layer relative to freezing, for this coupling step. Positive -> forms new ice;
            # negative -> available to melt existing ice from below.
            ice_frazil_melt_energy = jnp.where(
                ocn_idx,
                (constants.seawater_freezing_point_K - new_sea_surface_temperature)
                * state.mixed_layer_depth
                * constants.ocean_density
                * constants.ocean_specific_heat_capacity,
                0.0,
            )

            # The ocean itself never carries a sub-freezing SST -- that deficit was just
            # diverted into ice_frazil_melt_energy above.
            new_sea_surface_temperature = jnp.where(
                ocn_idx,
                jnp.maximum(new_sea_surface_temperature, constants.seawater_freezing_point_K),
                new_sea_surface_temperature,
            )

            new_state = state.copy(
                {
                    "sea_surface_temperature": new_sea_surface_temperature,
                    "sim_time": new_sim_time,
                }
            )

            predictions = {
                "state": new_state,
                "forcing": {
                    "total_heat_flux": total_heat_flux,
                },
                "derived": {
                    "ice_frazil_melt_energy": ice_frazil_melt_energy,
                },
            }
            if self.forcing_method == "Qflux":
                predictions["forcing"]["q_flux"] = snapshot_Qflux

            return {
                "state": new_state,
                "forcing": forcing,
                "derived": {
                    "ice_frazil_melt_energy": ice_frazil_melt_energy,
                },
            }, stack_objects([predictions])

        return step_function

    def _create_xarray_data_vars(self, predictions) -> dict[str, Any]:
        """Create xarray data variables for ocean output."""
        state = predictions["state"]
        forcing = predictions["forcing"]
        derived = predictions["derived"]
        T_grid_dims = ("time",) + self.horizontal_grids["T"].coordinate.dims

        data_vars = {
            "sea_surface_temperature": (
                T_grid_dims,
                state["sea_surface_temperature"],
                {
                    "long_name": "Sea surface temperature",
                    "units": "K",
                }
            ),
            "mixed_layer_depth": (
                T_grid_dims,
                state["mixed_layer_depth"],
                {
                    "long_name": "Mixed layer depth",
                    "units": "m",
                }
            ),
            "total_heat_flux": (
                T_grid_dims,
                forcing["total_heat_flux"],
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m-2",
                    "positive": "upward",
                }
            ),
            "ice_frazil_melt_energy": (
                T_grid_dims,
                derived["ice_frazil_melt_energy"],
                {
                    "long_name": "Freeze/melt potential (frzmlt): positive forms ice, negative melts ice",
                    "units": "J m-2",
                }
            ),
        }

        if self.forcing_method == "Qflux":
            data_vars["q_flux"] = (
                T_grid_dims,
                forcing["q_flux"],
                {
                    "long_name": "Q-flux",
                    "units": "W m-2",
                    "positive": "Heating the ocean",
                }
            )

        return data_vars

    def get_info(self):
        return {
            'relaxation_time' : self.relaxation_time,
        }
