"""Slab ocean model component."""

from typing import Optional, Dict, Any, Annotated
from pathlib import Path

import jax_datetime as jdt
import jax.numpy as jnp
import xarray as xr

from jem import constants
from jem.utils.bulk_op import stack_objects
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared
import jem.utils.data_structure as data_structure
from jem.components.slab.base import SlabModelBase

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
    """

    def __init__(
        self,
        grid_specification: str = "JCM::T31",
        start_datetime: jdt.Datetime = jdt.to_datetime("2001-01-01"),
        timestep: float = 86400.0,
        relaxation_time: float = 60 * 86400.0,
        mixed_layer_depth_min: float = 40.0,
        mixed_layer_depth_max: float = 60.0,
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
        SST_clim_file: Optional[str] = None,
        Q_flux_file: Optional[str] = None,
        forcing_method: Optional[str] = None,
        initialization_sea_surface_temperature: float = 288.15,
        mask_value: float = 0.0,
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
                raise FileNotFoundError(f"Q-flux file \"{str(self.Q_flux_file):s}\" is specified but it does not exist.")
        elif self.forcing_method == "relaxation":
            if self.SST_clim_file is None:
                print("Notice: `SST_clim_file` is not given. Default values (zeros) will be used.")
            elif not Path(self.SST_clim_file).exists():
                raise FileNotFoundError(f"SST climatology file \"{str(self.SST_clim_file):s}\" is specified but does not exist.")
            elif (self.relaxation_time < 0) or jnp.isnan(self.relaxation_time):
                raise ValueError("`relaxation_time` must be a positive number or infinity.")
        else: 
            raise ValueError(f"Unknown `forcing_method` is given: \"{str(forcing_method):s}\" ")
 

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
                positive_cosine_cubic_latitude_squared(self.latitude_radian) * 27.0
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
            raise Exception(
                "Warning: fmask_ocn and sea_surface_temperature_init do not share the same mask."
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
        return dict(
            state=self.component_state_class.zeros().copy({
                "mixed_layer_depth": init_mixed_layer_depth,
                "sea_surface_temperature": init_sea_surface_temperature,
            }),
            forcing=forcing,
        )

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
                # Get climatology at begin and end of timestep
                length_of_a_cycle = self.SST_clim.shape[2]
                clim_beg_idx, clim_end_idx = self._get_climatology_indices(
                    state.sim_time, start_day_offset, length_of_a_cycle
                )

                snapshot_SST_clim_beg = self.SST_clim[:, :, clim_beg_idx]
                snapshot_SST_clim_beg = jnp.where(
                    ocn_idx, snapshot_SST_clim_beg, default_land_surface_temperature
                )

                snapshot_SST_clim_end = self.SST_clim[:, :, clim_end_idx]
                snapshot_SST_clim_end = jnp.where(
                    ocn_idx, snapshot_SST_clim_end, default_land_surface_temperature
                )

                SST_clim_trend = (
                    snapshot_SST_clim_end - snapshot_SST_clim_beg
                ) / 86400.0  # per second

                new_sea_surface_temperature_anom = (
                    state.sea_surface_temperature - snapshot_SST_clim_beg
                )
            elif self.forcing_method == "Qflux":
                
                length_of_a_cycle = forcing.q_flux.shape[2]
                Qflux_beg_idx, _ = self._get_climatology_indices(
                    state.sim_time, start_day_offset, length_of_a_cycle
                )

                snapshot_Qflux = forcing.q_flux[:, :, Qflux_beg_idx]
                snapshot_Qflux = jnp.where(
                    ocn_idx, snapshot_Qflux, 0.0
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
                new_sea_surface_temperature += (
                    snapshot_SST_clim_beg + SST_clim_trend * self.timestep
                )
            
            # Apply land mask
            new_sea_surface_temperature = new_sea_surface_temperature.at[
                nonocn_idx
            ].set(default_land_surface_temperature)

            new_state = state.copy(
                {
                    "sea_surface_temperature": new_sea_surface_temperature,
                    "sim_time": new_sim_time,
                }
            )

            predictions = dict(
                state=new_state,
                forcing=dict(
                    total_heat_flux=total_heat_flux,
                )
            )
            if self.forcing_method == "Qflux":
                predictions["forcing"]["q_flux"] = snapshot_Qflux

            return dict(
                state=new_state,
                forcing=forcing
            ), stack_objects([predictions])

        return step_function

    def _create_xarray_data_vars(self, predictions) -> Dict[str, Any]:
        """Create xarray data variables for ocean output."""
        state = predictions["state"]
        forcing = predictions["forcing"]
        T_grid_dims = ("time",) + self.horizontal_grids["T"].coordinate.dims

        data_vars = dict(
            sea_surface_temperature=(
                T_grid_dims,
                state["sea_surface_temperature"],
                {
                    "long_name": "Sea surface temperature",
                    "units": "K",
                }
            ),
            mixed_layer_depth=(
                T_grid_dims,
                state["mixed_layer_depth"],
                {
                    "long_name": "Mixed layer depth",
                    "units": "m",
                }
            ),
            total_heat_flux=(
                T_grid_dims,
                forcing["total_heat_flux"],
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m-2",
                    "positive": "upward",
                }
            ),
        )

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
