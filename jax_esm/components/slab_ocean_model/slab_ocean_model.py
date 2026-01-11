"""Slab ocean model component."""

from typing import Optional, Dict, Any, Annotated

import jax_datetime as jdt
import jax.numpy as jnp
import xarray as xr

from jax_esm import constants
from jax_esm.utils.bulk_op import stack_objects
from jax_esm.utils.idealized_distribution import positive_cosine_cubic_latitude_squared
from jax_esm.components.slab.base import SlabModelBase
import jax_esm.base.data_structure as data_structure

from jax_esm.base.variable import VariableRegistry, VariableMetadata

default_land_surface_temperature = 288.15

@data_structure.typed_and_dimensioned
class PrognosticData:
    sim_time: Annotated[float, (), "zero_dimensional"]
    sea_surface_temperature: Annotated[
        float, ("latitudinal", "longitude"), "two_dimensional"
    ]
    mixed_layer_depth: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]


@data_structure.typed_and_dimensioned
class AirseaFlux:
    total_heat_flux: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]


@data_structure.typed_and_dimensioned
class OceanState:
    prog: PrognosticData

@data_structure.typed_and_dimensioned
class OceanForcing:
    flux: AirseaFlux
    q_flux: Annotated[float, ("latitudinal", "longitude", "month"), "two_dimensional_with_month"]
    
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
        save_interval: float = 86400.0,
        relaxation_time: float = 60 * 86400.0,
        mixed_layer_depth_min: float = 40.0,
        mixed_layer_depth_max: float = 60.0,
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
        SST_clim_file: Optional[str] = None,
        Q_flux_file: Optional[str] = None,
        forcing_method: Optional[str] = None,
        initialization_sea_surface_temperature: float = 288.15,
        forcing_method: Optional[str] = None,
    ):
        """Initialize slab ocean model.

        Args:
            grid_specification: Grid spec string (e.g., "JCM::T31")
            start_datetime: Simulation start datetime
            timestep: Model timestep in seconds
            save_interval: Output save interval in seconds (unused, kept for compatibility)
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
        self.has_climatology = False
        self.SST_clim = None
        self.time_factor = None
        self.cd_factor = None
        self.forcing_method = forcing_method

        self.validate()

    def validate(self):
        super()._validate()

        if self.forcing_method == "None" or self.forcing_method is None:
            # Do nothing
        
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
            elif ( not self.relaxation_time > 0 ) or not jnp.isinf(self.relaxation_time):
                raise ValueError("`relaxation_time` must be a positive number of infinity.")
        else: 
            raise ValueError(f"Unknown `forcing_method` is given: \"{str(forcing_method):s}\" ")
        
    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for ocean model."""
        decorator = data_structure.build_dataclass_from_typed_and_dimensioned({"two_dimensional": self.grid_shape})
        self.component_state_class = decorator(OceanState)
        self.component_forcing_class = decorator(OceanForcing)

    def _create_variable_registries(self) -> None:
        self.state_variable_registry = VariableRegistry()
        self.forcing_variable_registry = VariableRegistry()

        for target_registry, target_class in [
            (self.state_variable_registry, self.component_state_class),
            (self.forcing_variable_registry, self.component_forcing_class),
        ]:
            for name, _, dimensions, shape in target_class.typed_and_dimensioned_info():
                target_registry.register_variable(
                    VariableMetadata(name=name, shape=shape, dimensions=dimensions)
                )

    def _initialize_fields(self):
        """Initialize ocean model fields."""
        nonocn_idx = self.domain.horizontal_grids["T"].bmask != 0

        # Initialize mixed layer depth with latitudinal variation
        init_mixed_layer_depth = (
            self.mixed_layer_depth_max
            + (self.mixed_layer_depth_min - self.mixed_layer_depth_max)
            * jnp.cos(self.llat_rad) ** 3
        )

        # Load or create initial SST
        init_sea_surface_temperature = (
            positive_cosine_cubic_latitude_squared(self.llat_rad) * 27.0
            + constants.freezing_point_K
        )

        # Apply mask
        init_sea_surface_temperature = init_sea_surface_temperature.at[nonocn_idx].set(
            default_land_surface_temperature
        )

        # Validate mask consistency
        if jnp.sum(jnp.isnan(init_sea_surface_temperature)) == 0:
            print("domain.bmask and SST_clim do share the same mask.")
        else:
            raise Exception(
                "Warning: fmask_ocn and sea_surface_temperature_init do not share the same mask."
            )

        # Compute heat capacity and time factors for Euler backward scheme
        self.cd_factor = self.timestep / (
            constants.ocean_density
            * constants.ocean_specific_heat_capacity
            * init_mixed_layer_depth
        )
        
        if self.forcing_method == "relaxation":
            tau = jnp.ones_like(self.cd_factor) * self.relaxation_time
            self.time_factor = (1.0 + self.timestep / tau) ** (-1)
        else:
            self.time_factor = 1.0
            
        return self.component_state_class.zeros().copy(
            {
                "prog.mixed_layer_depth_max": init_mixed_layer_depth,
                "prog.sea_surface_temperature": init_sea_surface_temperature,
            }
        ), self.component_forcing_class.zeros()

    def _create_step_function_body(self):
        """Create the step function for ocean model."""
        start_day_offset = self._compute_start_day_offset()
        nonocn_idx = self.domain.horizontal_grids["T"].bmask != 0
        
        def step_function(state, forcing, t):
            
            new_sea_surface_temperature_anom = state.prog.sea_surface_temperature
            total_heat_flux = forcing.flux.total_heat_flux

            if self.forcing_method == "relaxation":
                # Get climatology at begin and end of timestep
                length_of_a_cycle = self.SST_clim.shape[2]
                clim_beg_idx, clim_end_idx = self._get_climatology_indices(
                    t, start_day_offset, length_of_a_cycle
                )

                ocn_idx = self.domain.horizontal_grids["T"].bmask == 0
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
                    state.prog.sea_surface_temperature - snapshot_SST_clim_beg
                )

            elif forcing_method == "Qflux":
                
                length_of_a_cycle = self.Qflux.shape[2]
                Qflux_beg_idx, Qflux_end_idx = self._get_climatology_indices(
                    t, start_day_offset, length_of_a_cycle
                )

                ocn_idx = self.domain.horizontal_grids["T"].bmask == 0
                snapshot_Qflux = self.Qflux[:, :, clim_beg_idx]
                snapshot_Qflux = jnp.where(
                    ocn_idx, snapshot_Qflux, 0.0
                )

                total_heat_flux = total_heat_flux + snapshot_Qflux


            # Euler backward step
            new_sim_time = state.prog.sim_time + self.timestep
            new_sea_surface_temperature_anom = self.time_factor * (
                new_sea_surface_temperature_anom
                + self.cd_factor * (- total_heat_flux)
            )

            # Add climatology back if relaxation method is used
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
                    "prog.sea_surface_temperature": new_sea_surface_temperature,
                    "prog.sim_time": new_sim_time,
                }
            )
            return new_state, stack_objects(
                [dict(prog=new_state.prog, forcing=forcing)]
            )

        return step_function

    def _create_xarray_data_vars(self, predictions) -> Dict[str, Any]:
        """Create xarray data variables for ocean output."""
        prog = predictions["prog"]
        forcing = predictions["forcing"]
        T_grid_dims = self._get_grid_dims()

        return dict(
            sea_surface_temperature=(T_grid_dims, prog.sea_surface_temperature),
            mixed_layer_depth=(T_grid_dims, prog.mixed_layer_depth),
            total_heat_flux=(T_grid_dims, forcing.flux.total_heat_flux),
        )
