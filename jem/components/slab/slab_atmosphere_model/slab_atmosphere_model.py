"""Slab atmosphere model component."""

from typing import Any

import jax.numpy as jnp
import jcm.constants as jcm_constants
import tree_math

from jem import constants
from jem.base.component import Carry, CouplingTime, Diagnostics
from jem.components.slab.base import SlabModelBase
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_atmosphere_model.params import SlabAtmosphereParameters
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared


@tree_math.struct
class AtmosphereState:
    mean_air_temperature: jnp.ndarray
    mean_zonal_wind_velocity: jnp.ndarray
    mean_meridional_wind_velocity: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        mean_air_temperature=None,
        mean_zonal_wind_velocity=None,
        mean_meridional_wind_velocity=None,
    ):
        return cls(
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
    aerodynamic formulation for sensible heat flux from the surface. It exists
    to exercise a coupled run without a full atmosphere, not to be a climate
    model: there is no radiation, no moisture and no dynamics.

    Physics:
        dT_air/dt = (H_ocean + H_land) / (M_air * cp_air)

    where:
        T_air: mean air temperature
        H_ocean, H_land: sensible heat fluxes from the surface below
        M_air: atmospheric column mass
        cp_air: specific heat capacity of dry air at constant pressure, from
            ``jcm.constants.cpd`` -- the same value the real atmosphere uses
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: SlabAtmosphereParameters | None = None,
        *,
        name: str = "atm",
    ):
        """Initialize the slab atmosphere model.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid.
        params : SlabAtmosphereParameters, optional
            Tunable parameters; defaults to
            :meth:`SlabAtmosphereParameters.default`.
        name : str
            Component name in the coupler's workflow and carry.

        """
        super().__init__(name=name, grid=grid)
        self.params = (
            SlabAtmosphereParameters.default() if params is None else params
        )

    def initialize(self) -> Carry:
        """Build the initial atmosphere carry."""
        params = self.params
        latitude = self.grid.latitude_radian

        mean_air_temperature = (
            params.initial_temperature_base
            + params.initial_temperature_amplitude
            * positive_cosine_cubic_latitude_squared(latitude)
        )

        return {
            "params": params,
            "state": AtmosphereState.zeros(
                self.grid.shape,
                mean_air_temperature=mean_air_temperature,
                mean_zonal_wind_velocity=jnp.full(
                    self.grid.shape, params.initial_zonal_wind
                ),
                mean_meridional_wind_velocity=jnp.full(
                    self.grid.shape, params.initial_meridional_wind
                ),
            ),
            "derived": AtmosphereDerived.zeros(self.grid.shape),
            "forcing": AtmosphereForcing.zeros(
                self.grid.shape,
                bulk_drag_coefficient=jnp.asarray(constants.bulk_drag_coefficient),
            ),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the air column by one coupling step (Euler forward)."""
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]

        # Land and ocean cells exchange heat with different surfaces, so each
        # bulk flux is computed everywhere and then kept only where its surface
        # actually is. ``binary_mask == 1`` is land (jem CLAUDE.md).
        land = self.grid.binary_mask == 1.0

        wind_speed = (
            state.mean_zonal_wind_velocity**2 + state.mean_meridional_wind_velocity**2
        ) ** 0.5
        bulk_conductance = (
            constants.surface_air_density
            * forcing.bulk_drag_coefficient
            * wind_speed
            * jcm_constants.cpd
        )

        # Positive upward: a surface warmer than the air heats the air, and the
        # column's own heat budget takes the flux with the opposite sign below.
        surface_temperature = jnp.where(
            land, forcing.land_surface_temperature, forcing.sea_surface_temperature
        )
        total_heat_flux = bulk_conductance * (
            surface_temperature - state.mean_air_temperature
        )

        heat_capacity = constants.atmosphere_column_mass * jcm_constants.cpd
        mean_air_temperature = (
            state.mean_air_temperature + time.dt / heat_capacity * total_heat_flux
        )

        new_state = state.replace(mean_air_temperature=mean_air_temperature)
        new_derived = AtmosphereDerived.zeros(
            self.grid.shape, internal_total_heat_flux=total_heat_flux
        )

        diagnostics = {
            "state": new_state,
            "derived": new_derived,
            "forcing": forcing,
        }
        return {"params": params, **diagnostics}, diagnostics

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for atmosphere output."""
        state = diagnostics["state"]
        forcing = diagnostics["forcing"]
        derived = diagnostics["derived"]
        dims = ("time",) + self.grid.dims

        return {
            "total_heat_flux": (
                dims,
                forcing.total_heat_flux,
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m-2",
                    "positive": "upward",
                },
            ),
            "internal_total_heat_flux": (
                dims,
                derived.internal_total_heat_flux,
                {
                    "long_name": "Internally-computed total heat flux (ocean + land sensible)",
                    "units": "W m-2",
                    "positive": "upward",
                },
            ),
            "mean_air_temperature": (
                dims,
                state.mean_air_temperature,
                {
                    "long_name": "Mean air column temperature",
                    "units": "K",
                },
            ),
            "mean_zonal_wind_velocity": (
                dims,
                state.mean_zonal_wind_velocity,
                {
                    "long_name": "Mean velocity of the air column in zonal direction",
                    "units": "m s-1",
                    "positive": "east",
                },
            ),
            "mean_meridional_wind_velocity": (
                dims,
                state.mean_meridional_wind_velocity,
                {
                    "long_name": "Mean velocity of the air column in meridional direction",
                    "units": "m s-1",
                    "positive": "north",
                },
            ),
        }
