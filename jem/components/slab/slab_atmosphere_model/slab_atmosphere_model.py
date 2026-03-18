"""Slab atmosphere model component."""

from typing import Optional, Dict, Any, Annotated

import jax_datetime as jdt
import jax.numpy as jnp

from jem import constants
from jem.utils.bulk_op import stack_objects
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared
from jem.components.slab.base import SlabModelBase
import jem.utils.data_structure as data_structure

@data_structure.typed_and_dimensioned
class AtmosphereState:
    sim_time: Annotated[float, (), "zero_dimensional"]
    mean_air_temperature: Annotated[
        float, ("latitudinal", "longitude"), "two_dimensional"
    ]
    mean_zonal_wind_velocity: Annotated[
        float, ("latitudinal", "longitude"), "two_dimensional"
    ]
    mean_meridional_wind_velocity: Annotated[
        float, ("latitudinal", "longitude"), "two_dimensional"
    ]

@data_structure.typed_and_dimensioned
class AtmosphereForcing:
    land_surface_temperature: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    sea_surface_temperature: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    total_heat_flux: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
    bulk_drag_coefficient: Annotated[float, (), "zero_dimensional"]

@data_structure.typed_and_dimensioned
class AtmosphereDerived:
    internal_total_heat_flux: Annotated[
        float, ("latitudinal", "longitude"), "two_dimensional"
    ]

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
        timestep: float = 86400.0,
        start_datetime: jdt.Datetime = jdt.to_datetime("2001-01-01"),
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
    ):
        """Initialize slab atmosphere model.

        Args:
            grid_specification: Grid spec string (e.g., "JCM::T31")
            timestep: Model timestep in seconds
            start_datetime: Simulation start datetime
            topography_file: Optional path to topography NetCDF file
            mask_file: Optional path to land/ocean mask NetCDF file
        """

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
        super().validate()

    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for atmosphere model."""
        decorator = data_structure.build_dataclass_from_typed_and_dimensioned(
            {
                "zero_dimensional": (),
                "two_dimensional": self.grid_shape,
                "heatflux_dimension": self.grid_shape + (2,),
            }
        )
        self.component_state_class = decorator(AtmosphereState)
        self.component_derived_class = decorator(AtmosphereDerived)
        self.component_forcing_class = decorator(AtmosphereForcing)

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
        """Initialize atmosphere model fields."""
        # Initialize air temperature with latitudinal variation
        init_mean_air_temperature = (
            positive_cosine_cubic_latitude_squared(self.latitude_radian) * 17.0
            + constants.freezing_point_K
        )
        init_mean_zonal_wind_velocity = jnp.zeros_like(self.latitude_radian) + 10.0
        init_mean_meridional_wind_velocity = jnp.zeros_like(self.latitude_radian)

        # Compute heat capacity factor for Euler forward scheme
        cd = (
            constants.atmosphere_column_mass
            * constants.atmosphere_specific_heat_capacity_under_constant_pressure
        )
        self.cd_factor = self.timestep / cd

        return dict(
            state=self.component_state_class.zeros().copy({
                "mean_air_temperature": init_mean_air_temperature,
                "mean_zonal_wind_velocity": init_mean_zonal_wind_velocity,
                "mean_meridional_wind_velocity": init_mean_meridional_wind_velocity,
            }),
            derived=self.component_derived_class.zeros(),
            forcing=self.component_forcing_class.zeros().copy({
                "bulk_drag_coefficient": jnp.array(1e-3),
            })
        )

    def _create_step_function_body(self):
        """Create the step function for atmosphere model."""
        land_index = self.horizontal_grids["T"].bmask == 1
        ocean_index = self.horizontal_grids["T"].bmask == 0

        def step_function(carry, step):
            state = carry["state"]
            forcing = carry["forcing"]
 
            # Compute wind speed
            wind_speed = (
                state.mean_zonal_wind_velocity**2
                + state.mean_meridional_wind_velocity**2
            ) ** 0.5

            # Bulk aerodynamic formula for ocean sensible heat flux
            ocean_sensible_heat_flux = (
                constants.surface_air_density
                * forcing.bulk_drag_coefficient
                * wind_speed
                * constants.atmosphere_specific_heat_capacity_under_constant_pressure
                * (
                    forcing.sea_surface_temperature
                    - state.mean_air_temperature
                )
            )
            # Bulk aerodynamic formula for land sensible heat flux
            land_sensible_heat_flux = (
                constants.surface_air_density
                * forcing.bulk_drag_coefficient
                * wind_speed
                * constants.atmosphere_specific_heat_capacity_under_constant_pressure
                * (
                    forcing.land_surface_temperature
                    - state.mean_air_temperature
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
            new_sim_time = state.sim_time + self.timestep
            new_mean_air_temperature = (
                state.mean_air_temperature + self.cd_factor * total_heat_flux
            )

            new_state = state.copy(
                {
                    "sim_time": new_sim_time,
                    "mean_air_temperature": new_mean_air_temperature,
                    "total_heat_flux": total_heat_flux,
                }
            )

            new_derived = self.component_derived_class.zeros().copy({
                "internal_total_heat_flux" : total_heat_flux,
            })

            return dict(
                state=new_state,
                derived=new_derived,
                forcing=forcing,
            ), stack_objects(
                [dict(state=new_state, forcing=forcing)]
            )

        return step_function

    def _create_xarray_data_vars(self, predictions) -> Dict[str, Any]:
        """Create xarray data variables for atmosphere output."""
        state = predictions["state"]
        forcing = predictions["forcing"]
        T_grid_dims = ("time",) + self.horizontal_grids["T"].coordinate.dims

        return dict(
            total_heat_flux=(
                T_grid_dims,
                forcing.total_heat_flux,
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m^-2",
                    "positive": "upward",
                }                   
            ),
            mean_air_temperature=(
                T_grid_dims,
                state.mean_air_temperature,
                {
                    "long_name": "Mean air column temperature",
                    "units": "K",
                }
            ),
            mean_zonal_wind_velocity=(
                T_grid_dims,
                state.mean_zonal_wind_velocity,
                {
                    "long_name": "Mean velocity of the air column in zonal direction",
                    "units": "m s^-1",
                    "positive": "east",
                } 
            ),
            mean_meridional_wind_velocity=(
                T_grid_dims,
                state.mean_meridional_wind_velocity,
                {
                    "long_name": "Mean velocity of the air column in meridional direction",
                    "units": "m s^-1",
                    "positive": "north",
                }
            ),
        )
