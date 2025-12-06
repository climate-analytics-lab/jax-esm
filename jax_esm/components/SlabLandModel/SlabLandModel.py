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

class SlabLandModel(Component):
    
    """
    A simple land model adapated from SlabOceanModel
    """

    def __init__(
        self,
        grid_specification:str="JCM::T31",
        start_datetime : jdt.Datetime = jdt.to_datetime("2001-01-01"),
        timestep : float = 86400.0,
        save_interval : float = 86400.0,
        relaxation_time : float = 60 * 86400.0,
        land_depth_min : float = 40.0,
        land_depth_max : float = 60.0,
        topography_file:Optional[str]=None,
        mask_file:Optional[str]=None,
        land_clim_file : Optional[str] = None,
    ):
        """Initialize slab ocean model."""
        
        super().__init__(CoupledComponentConfig(name="SlabLandModel", timestep=timestep))

        self.relaxation_time = relaxation_time
        self.start_datetime = start_datetime
        self.timestep = timestep
        self.land_depth_min = land_depth_min
        self.land_depth_max = land_depth_max
        self.topography_file = topography_file
        self.mask_file = mask_file
        self.land_clim_file = land_clim_file
        self.domain = Domain.from_grid_specification(
            grid_specification,
            topography_file = topography_file,
            mask_file = mask_file,
        )


        D2_nodal_shape = self.domain.grids["T"].nodal_shape
        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("land_surface_temperature", float, D2_nodal_shape),
                    ("land_depth", float, D2_nodal_shape),
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
        # Initialize slab land model boundary conditions
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

        land_index = self.domain.bmask == 1
        nonland_index = self.domain.bmask != 1
       
        print("Total land grid count: ", land_index.sum())
 
        # initialize land_depth
        init_land_depth = self.land_depth_max + (self.land_depth_min - self.land_depth_max) * jnp.cos(self.llat_rad)**3
        init_land_surface_temperature = None

        self.has_climatology = False
        self.land_surface_temperature_clim = None

        if self.land_clim_file is not None:
            print("Land climatology file. The given initial land surface temperature will be used.")
            print("Land climatology file: ", self.land_clim_file)            
            self.land_surface_temperature_clim = jnp.array(xr.open_dataset(self.land_clim_file)["stl"])
            self.has_climatology = True
            init_land_surface_temperature = self.land_surface_temperature_clim[:, :, 0].copy()
        else:
            print("Boundary does not exist. Idealized initial SST will be used.")
            init_land_surface_temperature = positive_cosine_cubic_latitude_squared(self.llat_rad) * 27.0 + 273.15 + 15

        
        init_land_surface_temperature = init_land_surface_temperature.at[nonland_index].set(0)
        if jnp.sum( jnp.isnan(init_land_surface_temperature) ) == 0:
            print("domain.bmask and land_surface_temperature_clim do share the same mask.")
        else:
            raise Exception("Warning: fmask_ocn and sst_init do not share the same mask.")

        if not self.has_climatology:
            print("Climaology land surface temperature does not exist. Set relaxation time to inifinity.")
            self.relaxation_time = jnp.inf
        
        # Compute heat capacity cd, and time factor for Euler backward scheme
        cd = constants.land_density * constants.land_specific_heat_capacity * init_land_depth
        tau = jnp.ones_like(cd) * self.relaxation_time
        self.time_factor = ( 1.0 + self.timestep / tau )**(-1)
        self.cd_factor = self.timestep / cd

        return self.component_state_class.zeros().copy(
            prog_kwargs = dict(
                land_depth = init_land_depth,
                land_surface_temperature = init_land_surface_temperature,
            ),
        )

    def generate_step_function(
        self,
        jitted: bool = True,
    ):

        # Find day of the year to locate climatology
        ref_year = self.start_datetime.to_pydatetime().year
        ref_dt = jdt.to_datetime(f"{ref_year:d}-01-01")
        start_day_offset = ( self.start_datetime - ref_dt ) / jdt.to_timedelta(1, "second")
       
        nonland_index = self.domain.bmask != 1
        land_index = self.domain.bmask == 1
 
        def step_function(state, forcing, t):
            
            # Compute anomaly later if climatology is given
            new_land_surface_temperature_anomaly = state.prog.land_surface_temperature
            if self.has_climatology:
                
                # Compute SST clim at begin and end time
                length_of_a_cycle = self.land_surface_temperature_clim.shape[2]
                clim_beg_idx = jnp.mod(jnp.int_( jnp.floor( (start_day_offset + t                ) / 86400.0) ), length_of_a_cycle)
                clim_end_idx = jnp.mod(jnp.int_( jnp.floor( (start_day_offset + t + self.timestep) / 86400.0) ), length_of_a_cycle)
                
                snapshot_land_surface_temperature_clim_beg = self.land_surface_temperature_clim[:, :, clim_beg_idx]
                snapshot_land_surface_temperature_clim_beg = jnp.where(land_index, snapshot_land_surface_temperature_clim_beg, 273.15+15)
                
                snapshot_land_surface_temperature_clim_end = self.land_surface_temperature_clim[:, :, clim_end_idx]
                snapshot_land_surface_temperature_clim_end = jnp.where(land_index, snapshot_land_surface_temperature_clim_end, 273.15+15)
                
                land_surface_temperature_clim_trend = (snapshot_land_surface_temperature_clim_end - snapshot_land_surface_temperature_clim_beg) / 86400.0  # convert to per second
                
                new_land_surface_temperature_anomaly = state.prog.land_surface_temperature - snapshot_land_surface_temperature_clim_beg
                
            new_sim_time = state.prog.sim_time + self.timestep
            new_land_surface_temperature_anomaly = self.time_factor * ( new_land_surface_temperature_anomaly + self.cd_factor * ( - (
                     forcing.flux.total_heat_flux
            )))
            
            new_land_surface_temperature = new_land_surface_temperature_anomaly
            if self.has_climatology:
                new_land_surface_temperature += snapshot_land_surface_temperature_clim_beg + land_surface_temperature_clim_trend * self.timestep
            
            new_land_surface_temperature = new_land_surface_temperature.at[nonland_index].set(0)
            new_state = state.copy(
                prog_kwargs = dict(
                    land_surface_temperature = new_land_surface_temperature,
                    sim_time = new_sim_time,
                ),
            )
            return new_state, stack_objects( [ dict(prog=new_state.prog, forcing=forcing) ] )
            
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
        prog    = predictions["prog"]
        forcing = predictions["forcing"]

        T_grid_axis_names = self.domain.grids["T"].axis_names
        T_grid_dims = ("time",) + T_grid_axis_names
        start_datetime_str = self.start_datetime.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")

        ds = xr.Dataset(
            data_vars = dict(
                land_surface_temperature = (T_grid_dims, prog.land_surface_temperature),
                land_depth = (T_grid_dims, prog.land_depth),
                total_heat_flux = (T_grid_dims, forcing.flux.total_heat_flux),
            ), 
            coords = dict(
                time = (["time",], prog.sim_time/3600.0, {"units": f"hours since {start_datetime_str:s}"}),
                latitude2D = (T_grid_axis_names, self.llat_rad * 180/jnp.pi),
                longitude2D = (T_grid_axis_names, self.llon_rad * 180/jnp.pi),
            ),
        )
        
        return ds
