"""Slab ocean model component."""

from typing import Optional

import jax_datetime as jdt
import jax
import jax.numpy as jnp

from jax_esm import constants as constants
from jax_esm.utils.bulk_op import stack_objects
from jax_esm.components.domain import Domain
import xarray as xr

from jax_esm.components.base import (
    Component,
    CoupledComponentConfig,
    create_component_state_class,
    create_component_forcing_class,
    create_field_group_class,
)

from jax_esm.utils.idealized_distribution import positive_cosine_cubic_latitude_squared


class SlabAtmosphereModel(Component):
    """
    Slab atmosphere model
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
        super().__init__(
            CoupledComponentConfig(name="SlabAtmosphereModel", timestep=timestep)
        )

        self.total_air_column_mass = constants.atmosphere_column_mass
        self.heat_capacity_under_constant_pressure = (
            constants.atmosphere_specific_heat_capacity_under_constant_pressure
        )

        self.start_datetime = start_datetime
        self.timestep = timestep
        self.save_interval = save_interval
        self.topography_file = topography_file
        self.mask_file = mask_file
        self.domain = Domain.from_grid_specification(
            grid_specification,
            topography_file=topography_file,
            mask_file=mask_file,
        )

        D2_nodal_shape = self.domain.grids["T"].nodal_shape

        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="state",
                fields=[
                    ("sim_time", float, ()),
                    ("mean_air_temperature", float, D2_nodal_shape),
                    ("mean_zonal_wind_velocity", float, D2_nodal_shape),
                    ("mean_meridional_wind_velocity", float, D2_nodal_shape),
                ],
            ),
            phydata_cls=create_field_group_class(
                cls_name="phydata",
                fields=[
                    ("hfluxn", float, D2_nodal_shape + (2,)),
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
                    ("bare_land_albedo", float, D2_nodal_shape),
                    ("sea_ice_concentration", float, D2_nodal_shape),
                    ("soil_moisture", float, D2_nodal_shape),
                    ("snow_cover", float, D2_nodal_shape),
                    ("land_surface_temperature", float, D2_nodal_shape),
                    ("sea_surface_temperature", float, D2_nodal_shape),
                ],
            ),
        )

    def initialize(self):
        # =========================================================================
        # Initialize slab ocean model boundary conditions
        # =========================================================================

        T_grid = self.domain.grids["T"]
        D2_nodal_shape = T_grid.nodal_shape

        lat_dim_idx = next(
            i
            for i, axis_name in enumerate(T_grid.axis_names)
            if axis_name == "latitude"
        )
        lon_dim_idx = next(
            i
            for i, axis_name in enumerate(T_grid.axis_names)
            if axis_name == "longitude"
        )

        self.llat_rad = jnp.repeat(
            jnp.expand_dims(
                T_grid.axis_values[lat_dim_idx],
                axis=lon_dim_idx,
            ),
            repeats=D2_nodal_shape[lon_dim_idx],
            axis=lon_dim_idx,
        )

        self.llon_rad = jnp.repeat(
            jnp.expand_dims(
                T_grid.axis_values[lon_dim_idx],
                axis=lat_dim_idx,
            ),
            repeats=D2_nodal_shape[lat_dim_idx],
            axis=lat_dim_idx,
        )

        init_mean_air_temperature = (
            positive_cosine_cubic_latitude_squared(self.llat_rad) * 27.0 + 273.15
        )
        init_mean_zonal_wind_velocity = jnp.zeros_like(self.llat_rad) + 10.0
        init_mean_meridional_wind_velocity = jnp.zeros_like(self.llat_rad)

        # Compute heat capacity cd, and time factor for Euler backward scheme
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
        )

    def generate_step_function(
        self,
        jitted: bool = True,
    ):
        land_index = self.domain.bmask == 1
        ocean_index = self.domain.bmask == 0

        def step_function(state, forcing, t):
            # Simple bulk formula
            surface_air_density = 1.22  # kg / m^3
            drag_coefficient = 1e-3  # scalar

            ocean_sensible_heat_flux = (
                surface_air_density
                * drag_coefficient
                * (
                    (
                        state.prog.mean_zonal_wind_velocity**2
                        + state.prog.mean_meridional_wind_velocity**2
                    )
                    ** 0.5
                )
                * constants.atmosphere_specific_heat_capacity_under_constant_pressure
                * (
                    forcing.scalar.sea_surface_temperature
                    - state.prog.mean_air_temperature
                )
            )

            land_sensible_heat_flux = (
                surface_air_density
                * drag_coefficient
                * (
                    (
                        state.prog.mean_zonal_wind_velocity**2
                        + state.prog.mean_meridional_wind_velocity**2
                    )
                    ** 0.5
                )
                * constants.atmosphere_specific_heat_capacity_under_constant_pressure
                * (
                    forcing.scalar.land_surface_temperature
                    - state.prog.mean_air_temperature
                )
            )

            ocean_sensible_heat_flux = ocean_sensible_heat_flux.at[land_index].set(0.0)
            land_sensible_heat_flux = land_sensible_heat_flux.at[ocean_index].set(0.0)

            latent_heat_flux = 0.0

            total_heat_flux = (
                ocean_sensible_heat_flux + land_sensible_heat_flux + latent_heat_flux
            )

            def sub_step_function(T, sim_time):
                return T + self.cd_factor * total_heat_flux, None

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

        return jax.jit(step_function) if jitted else step_function

    def validate(self):
        pass

    def predictions_to_xarray(
        self,
        predictions,
    ):
        """
        A tool function that converts a trajectory into an xarray Dataset.

        Args:
            predictions : The predictions returned from `forward_func`

        Returns:
            ds : The resulting xarray dataset.
        """
        prog = predictions["prog"]
        phydata = predictions["phydata"]
        forcing = predictions["forcing"]
        T_grid_axis_names = self.domain.grids["T"].axis_names
        start_datetime_str = self.start_datetime.to_pydatetime().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        ds = xr.Dataset(
            data_vars=dict(
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
            ),
            coords=dict(
                time=(
                    [
                        "time",
                    ],
                    prog.sim_time / 3600.0,
                    {"units": f"hours since {start_datetime_str:s}"},
                ),
                latitude2D=(T_grid_axis_names, self.llat_rad * 180 / jnp.pi),
                longitude2D=(T_grid_axis_names, self.llon_rad * 180 / jnp.pi),
            ),
        )

        return ds
