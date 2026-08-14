"""Slab atmosphere model component."""

from typing import Any

import jax.numpy as jnp
import jax_datetime as jdt
import tree_math

from jem import constants
from jem.components.slab.base import _DEFAULT_START_DATETIME, SlabModelBase
from jem.components.slab.grid import SlabGrid
from jem.utils.bulk_op import stack_objects
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared


@tree_math.struct
class AtmosphereState:
    sim_time: jnp.ndarray
    mean_air_temperature: jnp.ndarray
    mean_zonal_wind_velocity: jnp.ndarray
    mean_meridional_wind_velocity: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        sim_time=None,
        mean_air_temperature=None,
        mean_zonal_wind_velocity=None,
        mean_meridional_wind_velocity=None,
    ):
        return cls(
            sim_time if sim_time is not None else jnp.zeros(()),
            mean_air_temperature if mean_air_temperature is not None else jnp.zeros(shape),
            mean_zonal_wind_velocity if mean_zonal_wind_velocity is not None else jnp.zeros(shape),
            mean_meridional_wind_velocity if mean_meridional_wind_velocity is not None else jnp.zeros(shape),
        )


@tree_math.struct
class AtmosphereForcing:
    land_surface_temperature: jnp.ndarray
    sea_surface_temperature: jnp.ndarray
    total_heat_flux: jnp.ndarray
    bulk_drag_coefficient: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        land_surface_temperature=None,
        sea_surface_temperature=None,
        total_heat_flux=None,
        bulk_drag_coefficient=None,
    ):
        return cls(
            land_surface_temperature if land_surface_temperature is not None else jnp.zeros(shape),
            sea_surface_temperature if sea_surface_temperature is not None else jnp.zeros(shape),
            total_heat_flux if total_heat_flux is not None else jnp.zeros(shape),
            bulk_drag_coefficient if bulk_drag_coefficient is not None else jnp.zeros(()),
        )


@tree_math.struct
class AtmosphereDerived:
    internal_total_heat_flux: jnp.ndarray

    @classmethod
    def zeros(cls, shape, internal_total_heat_flux=None):
        return cls(
            internal_total_heat_flux if internal_total_heat_flux is not None else jnp.zeros(shape),
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
        grid: SlabGrid,
        timestep: float = 86400.0,
        start_datetime: jdt.Datetime = _DEFAULT_START_DATETIME,
        calendar: str = "365_day",
    ):
        """Initialize slab atmosphere model.

        Args:
            grid: The model's grid. See jem.components.slab.grid.SlabGrid.
            timestep: Model timestep in seconds
            start_datetime: Simulation start datetime
        """

        super().__init__(
            name="SlabAtmosphereModel",
            grid=grid,
            start_datetime=start_datetime,
            timestep=timestep,
            calendar=calendar,
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

    def initialize(self):
        """Initialize atmosphere model fields."""
        # Initialize air temperature with latitudinal variation
        init_mean_air_temperature = (
            positive_cosine_cubic_latitude_squared(self.grid.latitude_radian) * 17.0
            + constants.freezing_point_K
        )
        init_mean_zonal_wind_velocity = jnp.zeros_like(self.grid.latitude_radian) + 10.0
        init_mean_meridional_wind_velocity = jnp.zeros_like(self.grid.latitude_radian)

        # Compute heat capacity factor for Euler forward scheme
        cd = (
            constants.atmosphere_column_mass
            * constants.atmosphere_specific_heat_capacity_under_constant_pressure
        )
        self.cd_factor = self.timestep / cd

        return {
            "state": AtmosphereState.zeros(
                self.grid.shape,
                mean_air_temperature=init_mean_air_temperature,
                mean_zonal_wind_velocity=init_mean_zonal_wind_velocity,
                mean_meridional_wind_velocity=init_mean_meridional_wind_velocity,
            ),
            "derived": AtmosphereDerived.zeros(self.grid.shape),
            "forcing": AtmosphereForcing.zeros(
                self.grid.shape,
                bulk_drag_coefficient=jnp.array(1e-3),
            ),
        }

    def _create_step_function_body(self):
        """Create the step function for atmosphere model."""
        land_index = self.grid.binary_mask == 1
        ocean_index = self.grid.binary_mask == 0

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

            new_state = state.replace(
                sim_time=new_sim_time,
                mean_air_temperature=new_mean_air_temperature,
            )

            new_derived = AtmosphereDerived.zeros(
                self.grid.shape, internal_total_heat_flux=total_heat_flux,
            )

            return {
                "state": new_state,
                "derived": new_derived,
                "forcing": forcing,
            }, stack_objects(
                [{"state": new_state, "derived": new_derived, "forcing": forcing}]
            )

        return step_function

    def _create_xarray_data_vars(self, predictions) -> dict[str, Any]:
        """Create xarray data variables for atmosphere output."""
        state = predictions["state"]
        forcing = predictions["forcing"]
        derived = predictions["derived"]
        T_grid_dims = ("time",) + self.grid.dims

        return {
            "total_heat_flux": (
                T_grid_dims,
                forcing.total_heat_flux,
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m^-2",
                    "positive": "upward",
                }
            ),
            "internal_total_heat_flux": (
                T_grid_dims,
                derived.internal_total_heat_flux,
                {
                    "long_name": "Internally-computed total heat flux (ocean + land sensible)",
                    "units": "W m^-2",
                    "positive": "upward",
                }
            ),
            "mean_air_temperature": (
                T_grid_dims,
                state.mean_air_temperature,
                {
                    "long_name": "Mean air column temperature",
                    "units": "K",
                }
            ),
            "mean_zonal_wind_velocity": (
                T_grid_dims,
                state.mean_zonal_wind_velocity,
                {
                    "long_name": "Mean velocity of the air column in zonal direction",
                    "units": "m s^-1",
                    "positive": "east",
                } 
            ),
            "mean_meridional_wind_velocity": (
                T_grid_dims,
                state.mean_meridional_wind_velocity,
                {
                    "long_name": "Mean velocity of the air column in meridional direction",
                    "units": "m s^-1",
                    "positive": "north",
                }
            ),
        }
