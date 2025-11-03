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

def generate_idealized_mean_air_temperature(llat, llon):
    return jnp.where(jnp.abs(llat) < jnp.pi/3, 27*jnp.cos(3*llat/2)**2, 0) + 273.15


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

        self.total_air_column_mass = constants.air_column_mass        # kg / m^2
        self.heat_capacity_under_constant_pressure = constants.atm_cp # J / kg / K

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
                ],
            ),
        )

        self.component_forcing_class = create_component_forcing_class(
            cls_name = "forcing",
            flux_cls = create_field_group_class(
                cls_name = "flux",
                fields = [
                    ("total_heat_flux", float, D2_nodal_shape),
                ],
            ),
            scalar_cls = create_field_group_class(
                cls_name = "scalar",
                fields = [
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

        llat_rad = jnp.repeat(
            jnp.expand_dims(
                T_grid.axis_values[0],
                axis = 1,
            ),
            repeats = D2_nodal_shape[1],
            axis = 1,
        )
       
        llon_rad = jnp.repeat(
            jnp.expand_dims(
                T_grid.axis_values[1],
                axis = 0,
            ),
            repeats = D2_nodal_shape[0],
            axis = 0,
        )
        
        init_mean_air_temperature = generate_idealized_mean_air_temperature(llat_rad, llon_rad)

        # Compute heat capacity cd, and time factor for Euler backward scheme
        cd = total_air_column_mass * heat_capacity_under_constant_pressure 
        self.cd_factor = self.subtimestep / cd

        return self.component_state_class.zeros().copy(
            prog_kwargs = dict(
                mean_air_temperature = init_mean_air_temperature,
                mean_zonal_wind_velocity = mean_zonal_wind_velocity,
                mean_meridional_wind_velocity = mean_meridional_wind_velocity,
            ),
        )

    def generate_step_function(
        self,
        jitted: bool = True,
    ):

        def step_function(state, forcing, t):
            
            new_MAT = state.prog.mean_air_temperature
                
            new_sim_time = state.prog.sim_time
            for step in range(self.substeps):
                new_MAT = new_MAT + self.cd_factor * (
                    forcing.flux.total_heat_flux
                )
                new_sim_time += self.subtimestep
            
            new_state = state.copy(
                prog_kwargs = dict(
                    mean_air_temperature = new_MAT,
                    sim_time = new_sim_time,
                ),
            )
            return new_state, stack_objects( [ dict(prog=new_state.prog, forcing=forcing) ] )
            
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
        prog    = predictions["prog"]
        forcing = predictions["forcing"]
        ds = xr.Dataset(
            data_vars = dict(
                mean_air_temperature = (["time", "lon", "lat"], prog.mean_air_temperature),
                mean_zonal_wind_velocity = (["time", "lon", "lat"], prog.mean_zonal_wind_velocity),
                mean_meridional_wind_velocity = (["time", "lon", "lat"], prog.mean_meridional_wind_velocity),
                total_heat_flux = (["time", "lon", "lat"], forcing.flux.total_heat_flux),
            ), 
            coords = dict(
                time = (["time",], prog.sim_time),
            ),
        )
        
        return ds
