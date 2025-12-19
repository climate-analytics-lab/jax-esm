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

@data_structure.schema
class PrognosticData:
    sim_time: Annotated[float, (), "zero_dimensional"]
    sea_surface_temperature: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    mixed_layer_depth: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]


@data_structure.schema
class AirseaFlux:
    total_heat_flux: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]


@data_structure.schema
class OceanState:
    prog : PrognosticData


@data_structure.schema
class OceanForcing:
    flux : AirseaFlux

class SlabOceanModel(SlabModelBase):
    """Slab ocean model with prescribed mixed layer depth and climatology.

    This model simulates sea surface temperature evolution using a simple
    thermodynamic equation with optional relaxation to climatology.

    Physics:
        dT/dt = Q/(rho * cp * h) - (T - T_clim)/tau

    where:
        T: sea surface temperature
        Q: total heat flux (positive upward)
        rho: ocean density
        cp: ocean specific heat capacity
        h: mixed layer depth
        tau: relaxation timescale to climatology
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

        self.validate()

    def validate(self):
        super()._validate()


    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for ocean model."""
        decorator = data_structure.build_dataclass({"two_dimensional": self.grid_shape})
        self.component_state_class = decorator(OceanState)
        self.component_forcing_class = decorator(OceanForcing)
    
    def _create_variable_registries(self) -> None:
        self.state_variable_registry = VariableRegistry()
        self.forcing_variable_registry = VariableRegistry()

        for target_registry, target_class in [
            (self.state_variable_registry, self.component_state_class),
            (self.forcing_variable_registry, self.component_forcing_class),
        ]:
            for (name, _, dimensions, shape) in target_class.schema_info():
                target_registry.register_variable(VariableMetadata(name=name, shape=shape, dimensions=dimensions))
            
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
        if self.SST_clim_file is not None:
            print("SST climatology file. The given initial SST will be used.")
            print("SST climatology file: ", self.SST_clim_file)
            self.SST_clim = jnp.array(xr.open_dataset(self.SST_clim_file)["sst"])
            self.has_climatology = True
            init_sea_surface_temperature = self.SST_clim[:, :, 0].copy()
        else:
            print("Boundary does not exist. Idealized initial SST will be used.")
            init_sea_surface_temperature = (
                positive_cosine_cubic_latitude_squared(self.llat_rad) * 27.0
                + constants.freezing_point_K
            ) 

        # Apply mask
        init_sea_surface_temperature = init_sea_surface_temperature.at[nonocn_idx].set(0)

        # Validate mask consistency
        if jnp.sum(jnp.isnan(init_sea_surface_temperature)) == 0:
            print("domain.bmask and SST_clim do share the same mask.")
        else:
            raise Exception(
                "Warning: fmask_ocn and sea_surface_temperature_init do not share the same mask."
            )

        # Set relaxation time to infinity if no climatology
        if not self.has_climatology:
            print("Climaology SST does not exist. Set relaxation time to inifinity.")
            self.relaxation_time = jnp.inf

        # Compute heat capacity and time factors for Euler backward scheme
        cd = (
            constants.ocean_density
            * constants.ocean_specific_heat_capacity
            * init_mixed_layer_depth
        )
        tau = jnp.ones_like(cd) * self.relaxation_time
        self.time_factor = (1.0 + self.timestep / tau) ** (-1)
        self.cd_factor = self.timestep / cd

        return self.component_state_class.zeros().copy({
            "prog.mixed_layer_depth_max" : init_mixed_layer_depth,
            "prog.sea_surface_temperature" : init_sea_surface_temperature,
        }), self.component_forcing_class.zeros()

    def _create_step_function_body(self):
        """Create the step function for ocean model."""
        start_day_offset = self._compute_start_day_offset()
        nonocn_idx = self.domain.horizontal_grids["T"].bmask != 0

        def step_function(state, forcing, t):
            new_sea_surface_temperature_anom = state.prog.sea_surface_temperature

            if self.has_climatology:
                # Get climatology at begin and end of timestep
                length_of_a_cycle = self.SST_clim.shape[2]
                clim_beg_idx, clim_end_idx = self._get_climatology_indices(
                    t, start_day_offset, length_of_a_cycle
                )

                ocn_idx = self.domain.horizontal_grids["T"].bmask == 0
                snapshot_SST_clim_beg = self.SST_clim[:, :, clim_beg_idx]
                snapshot_SST_clim_beg = jnp.where(
                    ocn_idx, snapshot_SST_clim_beg, constants.freezing_point_K + 15.0
                )

                snapshot_SST_clim_end = self.SST_clim[:, :, clim_end_idx]
                snapshot_SST_clim_end = jnp.where(
                    ocn_idx, snapshot_SST_clim_end, constants.freezing_point_K + 15.0
                )

                SST_clim_trend = (
                    snapshot_SST_clim_end - snapshot_SST_clim_beg
                ) / 86400.0  # per second

                new_sea_surface_temperature_anom = (
                    state.prog.sea_surface_temperature - snapshot_SST_clim_beg
                )

            # Euler backward step
            new_sim_time = state.prog.sim_time + self.timestep
            new_sea_surface_temperature_anom = self.time_factor * (
                new_sea_surface_temperature_anom
                + self.cd_factor * ( - forcing.flux.total_heat_flux )
            )

            # Add climatology back
            new_sea_surface_temperature = new_sea_surface_temperature_anom
            if self.has_climatology:
                new_sea_surface_temperature += (
                    snapshot_SST_clim_beg + SST_clim_trend * self.timestep
                )

            # Apply land mask
            new_sea_surface_temperature = new_sea_surface_temperature.at[nonocn_idx].set(
                constants.freezing_point_K
            )

            new_state = state.copy({
                "prog.sea_surface_temperature" : new_sea_surface_temperature,
                "prog.sim_time" : new_sim_time,
            })
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
