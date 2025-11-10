"""Slab ocean model component."""

from typing import Dict, Tuple, Any, List, Optional

from datetime import datetime
import jax
import jax.numpy as jnp

from jax_esm import constants as constants
from jax_esm.utils.bulk_op import stack_objects
from jax_esm.components.domain import Domain
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

from jax_esm.components.base import (
    Component,
    ComponentConfig,
    create_component_state_class,
    create_component_forcing_class,
    create_field_group_class,
)

from jax_esm.utils.idealized_distribution import positive_cosine_cubic_latitude_squared

class SlabAtmosphereModel(Component):
    
    """
    Slab ocean model with prescribed mixed layer depth and climatology.
    """

    @classmethod
    def generate_default_configuration(
        cls,
        grid_specification:str="Veros::4deg",
        dict_form = False,
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
    ):
        
        config_dict = dict(
            name="default_config",
            timestep=1800.0,
            start_dt = datetime(year=2025, month=1, day=1),
            substeps = 24,
            save_interval = 5,
            domain = Domain.from_grid_specification(
                grid_specification,
                topography_file = topography_file,
                mask_file = mask_file,
            ),
            params=dict(
                relaxation_time = 60 * 86400.0,
            ),
        )
        
        if dict_form:
            return config_dict
        else:
            return ComponentConfig(**config_dict)


    @classmethod
    def generate_default_model(
        cls,
        grid_specification:str="JCM::T31",
        topography_file:Optional[str]=None,
        mask_file:Optional[str]=None,
    ):

        return SlabAtmosphereModel(
            SlabAtmosphereModel.generate_default_configuration(
                grid_specification=grid_specification,
                topography_file = topography_file,
                mask_file = mask_file,
        ))

    def __init__(
        self,
        config: ComponentConfig,
    ):
        """Initialize slab ocean model."""
        
        super().__init__(config)

        self.total_air_column_mass = constants.atmosphere_column_mass
        self.heat_capacity_under_constant_pressure = constants.atmosphere_specific_heat_capacity_under_constant_pressure

        self.start_dt = config.start_dt
        self.timestep = config.timestep
        self.substeps = config.substeps
        self.subtimestep = self.timestep / self.substeps

        D2_nodal_shape = config.domain.grids["T"].nodal_shape
        
        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("mean_air_temperature", float, D2_nodal_shape),
                    ("mean_zonal_wind_velocity", float, D2_nodal_shape),
                    ("mean_meridional_wind_velocity", float, D2_nodal_shape),
                ],
            ),

            phydata_cls =  create_field_group_class(
                cls_name = "phydata",
                fields = [
                    ("total_heat_flux", float, D2_nodal_shape),
                ],
            ),
        )

        self.component_forcing_class = create_component_forcing_class(
            cls_name = "forcing",
            flux_cls = create_field_group_class(
                cls_name = "flux",
                fields = [
                ],
            ),
            scalar_cls = create_field_group_class(
                cls_name = "scalar",
                fields = [
                    ("sea_surface_temperature", float, D2_nodal_shape),
                ],
            ),
        )
                
    def initialize(self):

        # =========================================================================
        # Initialize slab ocean model boundary conditions
        # =========================================================================
       
        T_grid = self.config.domain.grids["T"] 
        D2_nodal_shape = T_grid.nodal_shape
        config = self.config

        lat_dim_idx = next( i for i, axis_name in enumerate(T_grid.axis_names) if axis_name == "latitude")
        lon_dim_idx = next( i for i, axis_name in enumerate(T_grid.axis_names) if axis_name == "longitude")

        self.llat_rad = jnp.repeat(
            jnp.expand_dims(
                T_grid.axis_values[lat_dim_idx],
                axis = lon_dim_idx,
            ),
            repeats = D2_nodal_shape[lon_dim_idx],
            axis = lon_dim_idx,
        )
       
        self.llon_rad = jnp.repeat(
            jnp.expand_dims(
                T_grid.axis_values[lon_dim_idx],
                axis = lat_dim_idx,
            ),
            repeats = D2_nodal_shape[lat_dim_idx],
            axis = lat_dim_idx,
        )

        init_mean_air_temperature = positive_cosine_cubic_latitude_squared(self.llat_rad) * 27.0 + 273.15
        init_mean_zonal_wind_velocity = jnp.zeros_like(self.llat_rad) + 10.0
        init_mean_meridional_wind_velocity = jnp.zeros_like(self.llat_rad)

        # Compute heat capacity cd, and time factor for Euler backward scheme
        cd = constants.atmosphere_column_mass * constants.atmosphere_specific_heat_capacity_under_constant_pressure 
        self.cd_factor = self.subtimestep / cd

        return self.component_state_class.zeros().copy(
            prog_kwargs = dict(
                mean_air_temperature = init_mean_air_temperature,
                mean_zonal_wind_velocity = init_mean_zonal_wind_velocity,
                mean_meridional_wind_velocity = init_mean_meridional_wind_velocity,
            ),
        )

    def generate_step_function(
        self,
        jitted: bool = True,
    ):

        def step_function(state, forcing, t):
 
            # Simple bulk formula
            surface_air_density = 1.22 # kg / m^3
            drag_coefficient = 1e-3 # scalar
            sensible_heat_flux = surface_air_density * drag_coefficient * ((state.prog.mean_zonal_wind_velocity ** 2 + state.prog.mean_meridional_wind_velocity**2)**0.5) * constants.atmosphere_specific_heat_capacity_under_constant_pressure * (forcing.scalar.sea_surface_temperature - state.prog.mean_air_temperature)

            total_heat_flux = sensible_heat_flux

            def sub_step_function(T, sim_time):
                return T + self.cd_factor * total_heat_flux, None

            sub_sim_times = state.prog.sim_time + jnp.arange(self.substeps) * self.subtimestep
            new_sim_time = state.prog.sim_time + self.timestep
            new_MAT, _ = jax.lax.scan(
                sub_step_function,
                state.prog.mean_air_temperature,
                xs = sub_sim_times,    
            )

            new_state = state.copy(
                prog_kwargs = dict(
                    mean_air_temperature = new_MAT,
                    sim_time = new_sim_time,
                ),
                phydata_kwargs = dict(
                    total_heat_flux = total_heat_flux,
                ),
            )
            return new_state, stack_objects( [ dict(prog=new_state.prog, phydata=new_state.phydata, forcing=forcing) ] )
            
        return jax.jit(step_function) if jitted else step_function
        
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
        prog      = predictions["prog"]
        phydata   = predictions["phydata"]
        forcing   = predictions["forcing"]
        T_grid_axis_names = self.config.domain.grids["T"].axis_names
        T_grid_dims = ("time",) + T_grid_axis_names


        ds = xr.Dataset(
            data_vars = dict(
                mean_air_temperature = (["time", "longitude", "latitude"], prog.mean_air_temperature),
                mean_zonal_wind_velocity = (["time", "longitude", "latitude"], prog.mean_zonal_wind_velocity),
                mean_meridional_wind_velocity = (["time", "longitude", "latitude"], prog.mean_meridional_wind_velocity),
                total_heat_flux = (["time", "longitude", "latitude"], phydata.total_heat_flux),
            ), 
            coords = dict(
                time = (["time",], prog.sim_time),
                latitude2D = (T_grid_axis_names, self.llat_rad * 180/jnp.pi),
                longitude2D = (T_grid_axis_names, self.llon_rad * 180/jnp.pi),
            ),
        )
        
        return ds
