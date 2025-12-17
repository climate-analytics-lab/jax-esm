"""Slab land model component."""

from typing import Optional, Dict, Any

import jax_datetime as jdt
import jax.numpy as jnp
import xarray as xr

from jax_esm import constants
from jax_esm.utils.bulk_op import stack_objects
from jax_esm.utils.idealized_distribution import positive_cosine_cubic_latitude_squared
from jax_esm.components.slab.base import SlabModelBase
from jax_esm.utils.component_variable_tools import (
    create_variable_container_class,
    create_field_group_class,
)



class SlabLandModel(SlabModelBase):
    """Slab land model with prescribed depth and climatology.

    This model simulates land surface temperature evolution using a simple
    thermodynamic equation with optional relaxation to climatology.
    Adapted from SlabOceanModel.

    Physics:
        dT/dt = Q/(rho * cp * h) - (T - T_clim)/tau

    where:
        T: land surface temperature
        Q: total heat flux (positive upward)
        rho: land density
        cp: land specific heat capacity
        h: land depth (thermal)
        tau: relaxation timescale to climatology
    """

    def __init__(
        self,
        grid_specification: str = "JCM::T31",
        start_datetime: jdt.Datetime = jdt.to_datetime("2001-01-01"),
        timestep: float = 86400.0,
        save_interval: float = 86400.0,
        relaxation_time: float = 60 * 86400.0,
        land_depth_min: float = 40.0,
        land_depth_max: float = 60.0,
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
        land_clim_file: Optional[str] = None,
    ):
        """Initialize slab land model.

        Args:
            grid_specification: Grid spec string (e.g., "JCM::T31")
            start_datetime: Simulation start datetime
            timestep: Model timestep in seconds
            save_interval: Output save interval in seconds (unused, kept for compatibility)
            relaxation_time: Relaxation timescale to climatology in seconds
            land_depth_min: Minimum land depth in meters
            land_depth_max: Maximum land depth in meters
            topography_file: Optional path to topography NetCDF file
            mask_file: Optional path to land/ocean mask NetCDF file
            land_clim_file: Optional path to land surface temperature climatology NetCDF file
        """
        self.relaxation_time = relaxation_time
        self.land_depth_min = land_depth_min
        self.land_depth_max = land_depth_max
        self.land_clim_file = land_clim_file

        super().__init__(
            name="SlabLandModel",
            grid_specification=grid_specification,
            start_datetime=start_datetime,
            timestep=timestep,
            topography_file=topography_file,
            mask_file=mask_file,
        )

        # Climatology data (loaded during initialize)
        self.has_climatology = False
        self.land_surface_temperature_clim = None
        self.time_factor = None
        self.cd_factor = None

    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for land model."""
        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="state",
                fields=[
                    ("sim_time", float, ()),
                    ("land_surface_temperature", float, self.grid_shape),
                    ("land_depth", float, self.grid_shape),
                ],
            ),
            phydata_cls=create_field_group_class(
                cls_name="phydata",
                fields=[],
            ),
        )

        self.component_forcing_class = create_component_forcing_class(
            cls_name="forcing",
            flux_cls=create_field_group_class(
                cls_name="flux",
                fields=[
                    ("total_heat_flux", float, self.grid_shape),
                ],
            ),
            scalar_cls=create_field_group_class(
                cls_name="scalar",
                fields=[],
            ),
        )

    def _initialize_fields(self):
        """Initialize land model fields."""
        land_index = self.domain.bmask == 1
        nonland_index = self.domain.bmask != 1

        print("Total land grid count: ", land_index.sum())

        # Initialize land depth with latitudinal variation
        init_land_depth = (
            self.land_depth_max
            + (self.land_depth_min - self.land_depth_max)
            * jnp.cos(self.llat_rad) ** 3
        )

        # Load or create initial land surface temperature
        if self.land_clim_file is not None:
            print("Land climatology file. The given initial land surface temperature will be used.")
            print("Land climatology file: ", self.land_clim_file)
            self.land_surface_temperature_clim = jnp.array(
                xr.open_dataset(self.land_clim_file)["stl"]
            )
            self.has_climatology = True
            init_land_surface_temperature = self.land_surface_temperature_clim[:, :, 0].copy()
        else:
            print("Boundary does not exist. Idealized initial LST will be used.")
            init_land_surface_temperature = (
                positive_cosine_cubic_latitude_squared(self.llat_rad) * 27.0
                + constants.freezing_point_K
            )

        # Apply mask
        init_land_surface_temperature = init_land_surface_temperature.at[nonland_index].set(0)

        # Validate mask consistency
        if jnp.sum(jnp.isnan(init_land_surface_temperature)) == 0:
            print("domain.bmask and land_surface_temperature_clim do share the same mask.")
        else:
            raise Exception(
                "Warning: fmask_ocn and sst_init do not share the same mask."
            )

        # Set relaxation time to infinity if no climatology
        if not self.has_climatology:
            print("Climaology land surface temperature does not exist. Set relaxation time to inifinity.")
            self.relaxation_time = jnp.inf

        # Compute heat capacity and time factors for Euler backward scheme
        cd = (
            constants.land_density
            * constants.land_specific_heat_capacity
            * init_land_depth
        )
        tau = jnp.ones_like(cd) * self.relaxation_time
        self.time_factor = (1.0 + self.timestep / tau) ** (-1)
        self.cd_factor = self.timestep / cd

        return self.component_state_class.zeros().copy(
            prog_kwargs=dict(
                land_depth=init_land_depth,
                land_surface_temperature=init_land_surface_temperature,
            ),
        ), self.component_forcing_class.zeros()

    def _create_step_function_body(self):
        """Create the step function for land model."""
        start_day_offset = self._compute_start_day_offset()
        nonland_index = self.domain.bmask != 1
        land_index = self.domain.bmask == 1

        def step_function(state, forcing, t):
            # Compute anomaly if climatology is given
            new_land_surface_temperature_anomaly = state.prog.land_surface_temperature

            if self.has_climatology:
                # Get climatology at begin and end of timestep
                length_of_a_cycle = self.land_surface_temperature_clim.shape[2]
                clim_beg_idx, clim_end_idx = self._get_climatology_indices(
                    t, start_day_offset, length_of_a_cycle
                )

                snapshot_land_surface_temperature_clim_beg = (
                    self.land_surface_temperature_clim[:, :, clim_beg_idx]
                )
                snapshot_land_surface_temperature_clim_beg = jnp.where(
                    land_index,
                    snapshot_land_surface_temperature_clim_beg,
                    constants.freezing_point_K + 15.0,
                )

                snapshot_land_surface_temperature_clim_end = (
                    self.land_surface_temperature_clim[:, :, clim_end_idx]
                )
                snapshot_land_surface_temperature_clim_end = jnp.where(
                    land_index,
                    snapshot_land_surface_temperature_clim_end,
                    constants.freezing_point_K + 15.0,
                )

                land_surface_temperature_clim_trend = (
                    snapshot_land_surface_temperature_clim_end
                    - snapshot_land_surface_temperature_clim_beg
                ) / 86400.0  # per second

                new_land_surface_temperature_anomaly = (
                    state.prog.land_surface_temperature
                    - snapshot_land_surface_temperature_clim_beg
                )

            # Euler backward step
            new_sim_time = state.prog.sim_time + self.timestep
            new_land_surface_temperature_anomaly = self.time_factor * (
                new_land_surface_temperature_anomaly
                + self.cd_factor * (-(forcing.flux.total_heat_flux))
            )

            # Add climatology back
            new_land_surface_temperature = new_land_surface_temperature_anomaly
            if self.has_climatology:
                new_land_surface_temperature += (
                    snapshot_land_surface_temperature_clim_beg
                    + land_surface_temperature_clim_trend * self.timestep
                )

            # Apply ocean mask
            new_land_surface_temperature = new_land_surface_temperature.at[nonland_index].set(0)

            new_state = state.copy(
                prog_kwargs=dict(
                    land_surface_temperature=new_land_surface_temperature,
                    sim_time=new_sim_time,
                ),
            )
            return new_state, stack_objects(
                [dict(prog=new_state.prog, forcing=forcing)]
            )

        return step_function

    def _create_xarray_data_vars(self, predictions) -> Dict[str, Any]:
        """Create xarray data variables for land output."""
        prog = predictions["prog"]
        forcing = predictions["forcing"]
        T_grid_dims = self._get_grid_dims()

        return dict(
            land_surface_temperature=(T_grid_dims, prog.land_surface_temperature),
            land_depth=(T_grid_dims, prog.land_depth),
            total_heat_flux=(T_grid_dims, forcing.flux.total_heat_flux),
        )
