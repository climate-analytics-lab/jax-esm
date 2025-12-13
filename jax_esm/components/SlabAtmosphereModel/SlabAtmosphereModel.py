"""Slab atmosphere model component."""

from typing import Optional, Dict, Any

import jax_datetime as jdt
import jax.numpy as jnp

from jax_esm import constants
from jax_esm.utils.bulk_op import stack_objects
from jax_esm.utils.idealized_distribution import positive_cosine_cubic_latitude_squared
from jax_esm.components.slab.base import SlabModelBase
from jax_esm.components.base import (
    create_component_state_class,
    create_component_forcing_class,
    create_field_group_class,
)


class SlabAtmosphereModel(SlabModelBase):
    """Slab atmosphere model for simple air-sea-land heat exchange.

    This model simulates mean air temperature evolution using a bulk
    aerodynamic formulation for sensible heat flux from the surface.

    Physics:
        dT_air/dt = (H_ocean + H_land) / (M_air * cp_air)

    where:
        T_air: mean air temperature
        H_ocean, H_land: sensible heat fluxes from surface
        M_air: atmospheric column mass
        cp_air: specific heat capacity at constant pressure
    """

    def __init__(
        self,
        grid_specification: str = "JCM::T31",
        timestep: float = 3600.0 * 6,
        start_datetime: jdt.Datetime = jdt.to_datetime("2001-01-01"),
        save_interval: float = 86400.0,
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
    ):
        """Initialize slab atmosphere model.

        Args:
            grid_specification: Grid spec string (e.g., "JCM::T31")
            timestep: Model timestep in seconds
            start_datetime: Simulation start datetime
            save_interval: Output save interval in seconds (unused, kept for compatibility)
            topography_file: Optional path to topography NetCDF file
            mask_file: Optional path to land/ocean mask NetCDF file
        """
        self.save_interval = save_interval

        super().__init__(
            name="SlabAtmosphereModel",
            grid_specification=grid_specification,
            start_datetime=start_datetime,
            timestep=timestep,
            topography_file=topography_file,
            mask_file=mask_file,
        )

        # Atmospheric constants
        self.total_air_column_mass = constants.atmosphere_column_mass
        self.heat_capacity_under_constant_pressure = (
            constants.atmosphere_specific_heat_capacity_under_constant_pressure
        )

        # Computed during initialization
        self.cd_factor = None

    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for atmosphere model."""
        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="state",
                fields=[
                    ("sim_time", float, ()),
                    ("mean_air_temperature", float, self.grid_shape),
                    ("mean_zonal_wind_velocity", float, self.grid_shape),
                    ("mean_meridional_wind_velocity", float, self.grid_shape),
                ],
            ),
            phydata_cls=create_field_group_class(
                cls_name="phydata",
                fields=[
                    ("hfluxn", float, self.grid_shape + (2,)),
                ],
            ),
        )

        self.component_forcing_class = create_component_forcing_class(
            cls_name="forcing",
            flux_cls=create_field_group_class(
                cls_name="flux",
                fields=[],
            ),
            scalar_cls=create_field_group_class(
                cls_name="scalar",
                fields=[
                    ("bare_land_albedo", float, self.grid_shape),
                    ("sea_ice_concentration", float, self.grid_shape),
                    ("soil_moisture", float, self.grid_shape),
                    ("snow_cover", float, self.grid_shape),
                    ("land_surface_temperature", float, self.grid_shape),
                    ("sea_surface_temperature", float, self.grid_shape),
                ],
            ),
        )

    def _initialize_fields(self):
        """Initialize atmosphere model fields."""
        # Initialize air temperature with latitudinal variation
        init_mean_air_temperature = (
            positive_cosine_cubic_latitude_squared(self.llat_rad) * 17.0
            + constants.freezing_point_K
        )
        init_mean_zonal_wind_velocity = jnp.zeros_like(self.llat_rad) + 10.0
        init_mean_meridional_wind_velocity = jnp.zeros_like(self.llat_rad)

        # Compute heat capacity factor for Euler forward scheme
        cd = (
            constants.atmosphere_column_mass
            * constants.atmosphere_specific_heat_capacity_under_constant_pressure
        )
        self.cd_factor = self.timestep / cd

        return self.component_state_class.zeros().copy(
            prog_kwargs=dict(
                mean_air_temperature=init_mean_air_temperature,
                mean_zonal_wind_velocity=init_mean_zonal_wind_velocity,
                mean_meridional_wind_velocity=init_mean_meridional_wind_velocity,
            ),
        ), self.component_forcing_class.zeros()

    def _create_step_function_body(self):
        """Create the step function for atmosphere model."""
        land_index = self.domain.bmask == 1
        ocean_index = self.domain.bmask == 0

        def step_function(state, forcing, t):
            # Compute wind speed
            wind_speed = (
                state.prog.mean_zonal_wind_velocity ** 2
                + state.prog.mean_meridional_wind_velocity ** 2
            ) ** 0.5

            # Bulk aerodynamic formula for ocean sensible heat flux
            ocean_sensible_heat_flux = (
                constants.surface_air_density
                * constants.bulk_drag_coefficient
                * wind_speed
                * constants.atmosphere_specific_heat_capacity_under_constant_pressure
                * (
                    forcing.scalar.sea_surface_temperature
                    - state.prog.mean_air_temperature
                )
            )

            # Bulk aerodynamic formula for land sensible heat flux
            land_sensible_heat_flux = (
                constants.surface_air_density
                * constants.bulk_drag_coefficient
                * wind_speed
                * constants.atmosphere_specific_heat_capacity_under_constant_pressure
                * (
                    forcing.scalar.land_surface_temperature
                    - state.prog.mean_air_temperature
                )
            )

            # Apply masks
            ocean_sensible_heat_flux = ocean_sensible_heat_flux.at[land_index].set(0.0)
            land_sensible_heat_flux = land_sensible_heat_flux.at[ocean_index].set(0.0)

            latent_heat_flux = 0.0

            total_heat_flux = (
                ocean_sensible_heat_flux + land_sensible_heat_flux + latent_heat_flux
            )
            
            # Update temperature
            new_sim_time = state.prog.sim_time + self.timestep
            new_mean_air_temperature = (
                state.prog.mean_air_temperature + self.cd_factor * total_heat_flux
            )
            new_hfluxn = state.phydata.hfluxn.at[:, :, 0].set(total_heat_flux)

            new_state = state.copy(
                prog_kwargs=dict(
                    mean_air_temperature=new_mean_air_temperature,
                    sim_time=new_sim_time,
                ),
                phydata_kwargs=dict(
                    hfluxn=new_hfluxn,
                ),
            )
            return new_state, stack_objects(
                [dict(prog=new_state.prog, phydata=new_state.phydata, forcing=forcing)]
            )

        return step_function

    def _create_xarray_data_vars(self, predictions) -> Dict[str, Any]:
        """Create xarray data variables for atmosphere output."""
        prog = predictions["prog"]
        phydata = predictions["phydata"]
        forcing = predictions["forcing"]

        return dict(
            hfluxn=(["time", "longitude", "latitude", "two"], phydata.hfluxn),
            mean_air_temperature=(
                ["time", "longitude", "latitude"],
                prog.mean_air_temperature,
            ),
            mean_zonal_wind_velocity=(
                ["time", "longitude", "latitude"],
                prog.mean_zonal_wind_velocity,
            ),
            mean_meridional_wind_velocity=(
                ["time", "longitude", "latitude"],
                prog.mean_meridional_wind_velocity,
            ),
            bare_land_albedo=(
                ["time", "longitude", "latitude"],
                forcing.scalar.bare_land_albedo,
            ),
            sea_ice_concentration=(
                ["time", "longitude", "latitude"],
                forcing.scalar.sea_ice_concentration,
            ),
            soil_moisture=(
                ["time", "longitude", "latitude"],
                forcing.scalar.soil_moisture,
            ),
            snow_cover=(
                ["time", "longitude", "latitude"],
                forcing.scalar.snow_cover,
            ),
            land_surface_temperature=(
                ["time", "longitude", "latitude"],
                forcing.scalar.land_surface_temperature,
            ),
            sea_surface_temperature=(
                ["time", "longitude", "latitude"],
                forcing.scalar.sea_surface_temperature,
            ),
        )
