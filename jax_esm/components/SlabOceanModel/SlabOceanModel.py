"""Slab ocean model component."""

from typing import Dict, Tuple, Any, List, Optional

import jax_datetime as jdt
import jax
import jax.numpy as jnp

from jax_esm import constants as constants
from jax_esm.utils.bulk_op import stack_objects
from jax_esm.components.domain import Domain
from pathlib import Path
import xarray as xr
import numpy as np

from jax_esm.components.base import (
    Component,
    CoupledComponentConfig,
    create_component_state_class,
    create_component_forcing_class,
    create_field_group_class,
)

from jax_esm.utils.idealized_distribution import positive_cosine_cubic_latitude_squared

class SlabOceanModel(Component):
    
    """
    Slab ocean model with prescribed mixed layer depth and climatology.
    """

    @classmethod
    def generate_default_configuration(
        cls,
        grid_specification:str="JCM::T31",
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
        SST_clim_file : Optional[str] = None,
        relaxation_time : float = 60 * 86400.0,
    ):
        
        config_dict = dict(
            name="default_config",
            timestep=1800.0,
            start_dt = jdt.to_datetime('2000-01-01'),
            save_interval = 5,
            domain = Domain.from_grid_specification(
                grid_specification,
                topography_file = topography_file,
                mask_file = mask_file,
            ),
            relaxation_time = relaxation_time,
            SST_clim_file = SST_clim_file,
        )
        
        return config_dict


    @classmethod
    def generate_default_model(
        cls,
        grid_specification:str="JCM::T31",
        topography_file:Optional[str]=None,
        mask_file:Optional[str]=None,
    ):

        return SlabOceanModel(
            SlabOceanModel.generate_default_configuration(
                grid_specification=grid_specification,
                topography_file = topography_file,
                mask_file = mask_file,
        ))

    def __init__(
        self,
        start_dt : jdt.Datetime,
        timestep : float,
        save_interval : float,
        domain : Domain,
        relaxation_time : float,
        SST_clim_file : Optional[str] = None,
        mixed_layer_depth_min : float = 40.0,
        mixed_layer_depth_max : float = 60.0,
    ):
        """Initialize slab ocean model."""
        
        super().__init__(CoupledComponentConfig(name="SlabOceanModel", timestep=timestep))

        self.relaxation_time = relaxation_time
        self.start_dt = start_dt
        self.timestep = timestep
        self.domain = domain
        self.mixed_layer_depth_min = mixed_layer_depth_min
        self.mixed_layer_depth_max = mixed_layer_depth_max
        self.SST_clim_file = SST_clim_file
        D2_nodal_shape = self.domain.grids["T"].nodal_shape
        
        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("T", float, D2_nodal_shape),
                    ("mld", float, D2_nodal_shape),
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
       
        T_grid = self.domain.grids["T"] 
        D2_nodal_shape = T_grid.nodal_shape

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

        lnd_idx = self.domain.bmask == 1
        ocn_idx = self.domain.bmask == 0
        
        # initialize mld
        init_mld = self.mixed_layer_depth_max + (self.mixed_layer_depth_min - self.mixed_layer_depth_max) * jnp.cos(self.llat_rad)**3
        init_T = None

        self.has_climatology = False
        self.SST_clim = None

        if self.SST_clim_file is not None:
            print("SST climatology file. The given initial SST will be used.")
            print("SST climatology file: ", self.SST_clim_file)            
            self.SST_clim = jnp.array(xr.open_dataset(self.SST_clim_file)["sst"])
            self.has_climatology = True
            init_T = self.SST_clim[:, :, 0].copy()
        else:
            print("Boundary does not exist. Idealized initial SST will be used.")
            init_T = positive_cosine_cubic_latitude_squared(self.llat_rad) * 27.0 + 273.15 + 15

        
        init_T = init_T.at[lnd_idx].set(273.15+15)
        if jnp.sum( jnp.isnan(init_T) ) == 0:
            print("domain.bmask and SST_clim do share the same mask.")
        else:
            raise Exception("Warning: fmask_ocn and sst_init do not share the same mask.")

        # Compute heat capacity cd, and time factor for Euler backward scheme
        cd = constants.ocean_density * constants.ocean_specific_heat_capacity * init_mld 
        tau = jnp.ones_like(cd) * self.relaxation_time
        self.time_factor = ( 1.0 + self.timestep / tau )**(-1)
        self.cd_factor = self.timestep / cd

        return self.component_state_class.zeros().copy(
            prog_kwargs = dict(
                mld = init_mld,
                T = init_T,
            ),
        )

    def generate_step_function(
        self,
        jitted: bool = True,
    ):

        # Find day of the year to locate climatology
        #ref_dt = pd.Timestamp(year=self.start_dt.year, month=self.start_dt.month, day=1)
        #start_dt_offset = jnp.int_(jnp.floor( ( self.start_dt - ref_dt ) / pd.Timedelta(days=1) ))
        
        def step_function(state, forcing, t):
            new_Tanom = state.prog.T
            if self.has_climatology:
                
                days_after_start = jnp.floor( state.prog.sim_time / 86400.0 ).astype(jnp.int32)
                
                clim_day_beg = 0#start_dt_offset + days_after_start
                clim_day_end = jnp.mod(clim_day_beg + 1, self.SST_clim.shape[2])
                
                ocn_idx = self.domain.bmask == 0
                snapshot_SST_clim_beg = self.SST_clim[:, :, clim_day_beg]
                snapshot_SST_clim_beg = jnp.where(ocn_idx, snapshot_SST_clim_beg, 273.15+15)
                
                snapshot_SST_clim_end = self.SST_clim[:, :, clim_day_end]
                snapshot_SST_clim_end = jnp.where(ocn_idx, snapshot_SST_clim_end, 273.15+15)
                
                SST_clim_trend = (snapshot_SST_clim_end - snapshot_SST_clim_beg) / 86400.0  # convert to per second
                
                new_Tanom = state.prog.T - snapshot_SST_clim_beg
                
            new_sim_time = state.prog.sim_time + self.timestep
            new_Tanom = self.time_factor * ( new_Tanom + self.cd_factor * ( - (
                     forcing.flux.total_heat_flux
            )))
            
            new_T = new_Tanom 
            if self.has_climatology:
                new_T += snapshot_SST_clim_beg + SST_clim_trend * self.timestep
                
            new_state = state.copy(
                prog_kwargs = dict(
                    T = new_T,
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

        T_grid_axis_names = self.domain.grids["T"].axis_names
        T_grid_dims = ("time",) + T_grid_axis_names

        ds = xr.Dataset(
            data_vars = dict(
                T   = (T_grid_dims, prog.T),
                mld = (T_grid_dims, prog.mld),
                total_heat_flux = (T_grid_dims, forcing.flux.total_heat_flux),
            ), 
            coords = dict(
                time = (["time",], prog.sim_time),
                latitude2D = (T_grid_axis_names, self.llat_rad * 180/jnp.pi),
                longitude2D = (T_grid_axis_names, self.llon_rad * 180/jnp.pi),
            ),
        )
        
        return ds
