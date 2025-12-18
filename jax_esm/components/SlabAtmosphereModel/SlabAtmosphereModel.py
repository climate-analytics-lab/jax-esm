"""Slab atmosphere model component."""

from typing import Optional, Dict, Any, Annotated

import jax_datetime as jdt
import jax.numpy as jnp

from jax_esm import constants
from jax_esm.utils.bulk_op import stack_objects
from jax_esm.utils.idealized_distribution import positive_cosine_cubic_latitude_squared
from jax_esm.components.slab.base import SlabModelBase
import jax_esm.base.data_structure as data_structure

@data_structure.schema
class PrognosticData:
    sim_time: Annotated[float, (), "zero_dimensional"]
    mean_air_temperature: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    mean_zonal_wind_velocity: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    mean_meridional_wind_velocity: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]

@data_structure.schema
class PhysicsData:
    hfluxn: Annotated[float, ("latitudinal", "longitude"), "heatflux_dimension"]

@data_structure.schema
class AtmosphereState:
    prog : PrognosticData
    phydata : PhysicsData

@data_structure.schema
class SurfaceForcing:
    bulk_drag_coefficient : Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    bare_land_albedo : Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    sea_ice_concentration : Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    soil_moisture : Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    snow_cover : Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    land_surface_temperature : Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    sea_surface_temperature: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]


@data_structure.schema
class AtmosphereForcing:
    scalar : SurfaceForcing



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

        self.validate()

    def validate(self):
        super()._validate()


    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for atmosphere model."""
        decorator = data_structure.build_dataclass({
            "two_dimensional": self.grid_shape,
            "heatflux_dimension": self.grid_shape + (2,),
        })
        self.component_state_class = decorator(AtmosphereState)
        self.component_forcing_class = decorator(AtmosphereForcing)

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

        return self.component_state_class.zeros().copy({
            "prog.mean_air_temperature" : init_mean_air_temperature,
            "prog.mean_zonal_wind_velocity" : init_mean_zonal_wind_velocity,
            "prog.mean_meridional_wind_velocity" : init_mean_meridional_wind_velocity,
        }), self.component_forcing_class.zeros().copy({
            "scalar.bulk_drag_coefficient" : jnp.array(1e-3),
        })

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

            print("bulk_drag_coefficient = ", forcing.scalar.bulk_drag_coefficient)
            # Bulk aerodynamic formula for ocean sensible heat flux
            ocean_sensible_heat_flux = (
                constants.surface_air_density
                * forcing.scalar.bulk_drag_coefficient
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
                * forcing.scalar.bulk_drag_coefficient
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
            new_state = state.copy({
                "prog.mean_air_temperature" : new_mean_air_temperature,
                "prog.sim_time" : new_sim_time,
                "phydata.hfluxn" : new_hfluxn,
            })
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
