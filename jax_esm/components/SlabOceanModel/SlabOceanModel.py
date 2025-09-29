"""Slab ocean model component."""

from typing import Dict, Tuple, Any, List

import jax
import jax.numpy as jnp

from jax_esm import constants as constants
from jax_esm.utils.meta_prog_class import createFieldsClass
from jax_esm.utils.bulk_op import stack_objects

from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

from jax_esm.components.base import (
    Component,
    ComponentConfig,
    create_component_state_class,
    create_field_group_class,
)


class SlabOceanModel(Component):
    """
    Slab ocean model with prescribed mixed layer depth and climatology.
    """
        
    def __init__(
        self,
        config: ComponentConfig,
    ):
        """Initialize slab ocean model."""
        
        super().__init__(config)

        self.ocn_rho = constants.ocn_rho # Seawater density (kg / m^3)
        self.ocn_cp = constants.ocn_cp   # Seawater specific heat capacity (J/kg/K)

        self.coords = config.params["coords"]
        self.geometry = config.params["geometry"]
        self.relaxation_time = config.params["relaxation_time"]

        self.start_dt = config.start_dt
        self.timestep = config.timestep
        self.substeps = config.substeps
        self.subtimestep = self.timestep / self.substeps

         
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        
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
                    ("heatflx", float, D2_nodal_shape),
                ],
            ),
        )


                
    def initialize(self):

        # =========================================================================
        # Initialize slab ocean model boundary conditions
        # =========================================================================
        
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        config = self.config
        
        llon_rad = jnp.repeat(
            jnp.expand_dims(
                self.coords.horizontal.longitudes,
                axis = 1,
            ),
            repeats = D2_nodal_shape[1],
            axis = 1,
        )

        llat_rad = jnp.repeat(
            jnp.expand_dims(
                self.coords.horizontal.latitudes,
                axis = 0,
            ),
            repeats = D2_nodal_shape[0],
            axis = 0,
        )

        # initialize mld
        mld_max = config.params["mld_max"] if "mld_max" in config.params else 60.0
        mld_min = config.params["mld_min"] if "mld_min" in config.params else 40.0

        init_mld = mld_max + (mld_min - mld_max) * jnp.cos(llat_rad)**3
        init_T = None
        self.SST_clim = None
        self.fmask_ocn = jnp.ones_like(init_mld)
        
        if "boundaries" in config.params and config.params["boundaries"] is not None:

            print("Boundary exists. ")
            print("Boundary file: ", config.params["boundary_file"])
            
            boundaries = config.params["boundaries"]
            thrsh = 0.3

            self.SST_clim = jnp.array(xr.open_dataset(config.params["boundary_file"])["sst"])
            
            # Fractional and binary land masks
            fmask_lnd = boundaries.fmask
            #bmask_lnd = jnp.where(fmask_lnd >= thrsh, 1.0, 0.0)
    
            # Update fmask_lnd based on the conditions
            fmask_lnd = jnp.where(
                fmask_lnd >= thrsh,
                1.0,
                0.0,
            )

            fmask_ocn = 1.0 - fmask_lnd
            
            #init_mld = init_mld.at[fmask_ocn == 0].set(jnp.nan)
            init_T = self.SST_clim[:, :, 0].copy().at[fmask_ocn == 0].set(273.15+15)#.set(jnp.nan)
            
            if jnp.any( jnp.isnan(init_T) == (fmask_ocn == 0) ):
                print("fmask_ocn and init_T do share the same mask.")
            else:
                raise Exception("Warning: fmask_ocn and sst_init do not share the same mask.")

            self.fmask_ocn = fmask_ocn
            
        else:
            
            T_max = config.params["T_max"] if "T_max" in config.params else 273.15 + 30.0
            T_min = config.params["T_min"] if "T_min" in config.params else 273.15 + 5.0            
            init_T   = T_min + (T_max - T_min) * jnp.cos(llat_rad - 20 * jnp.pi / 180.0)**3 + 5.0 * jnp.cos(llon_rad)

        
        # Compute cd and time factor
        
        cd = self.ocn_rho * self.ocn_cp * init_mld 
        tau = jnp.ones_like(cd) * self.relaxation_time
    
        self.time_factor = ( 1.0 + self.subtimestep / tau )**(-1)
        self.cd_factor = self.subtimestep / cd

        return self.component_state_class.zeros().copy(
            prog_kwargs = dict(
                mld = init_mld,
                T = init_T,
            ),
        )

        
    def gen_step_fn(
        self,
    ):

        # Find day of the year to locate climatology
        ref_dt = pd.Timestamp(year=self.start_dt.year, month=self.start_dt.month, day=1)
        start_dt_offset = jnp.int_(jnp.floor( ( self.start_dt - ref_dt ) / pd.Timedelta(days=1) ))
        
        @jax.jit
        def step_fn(cpl, t):


            days_after_start = jnp.floor( cpl.ocn.prog.sim_time / 86400.0 ).astype(jnp.int32)
            
            # This snapshot SST will be frozen
            snapshot_SST_clim = self.SST_clim[:, :, start_dt_offset + days_after_start]
            snapshot_SST_clim = jnp.where(self.fmask_ocn != 0, snapshot_SST_clim, 273.15+15)

            new_Tanom = cpl.ocn.prog.T - snapshot_SST_clim

            
            for step in range(self.substeps):
                new_Tanom = self.time_factor * ( new_Tanom + self.cd_factor * ( - (
                    cpl.flx.phydata.hfluxn[:, :, 0]
                )))

                cpl.ocn.prog = cpl.ocn.prog.copy(
                    sim_time = cpl.ocn.prog.sim_time + self.subtimestep
                )
            
            new_T = new_Tanom + snapshot_SST_clim

            new_state = cpl.ocn.copy(
                prog_kwargs = dict(
                    T = new_T,
                ),
            )
            
            return new_state, stack_objects( [ dict(prog=new_state.prog) , ] )
            
        return step_fn

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
        st = predictions["state"]
        ds = xr.Dataset(
            data_vars = dict(
                T   = (["time", "lon", "lat"], st.T),
                mld = (["time", "lon", "lat"], st.mld),
            ), 
            coords = dict(
                time = (["time",], st.sim_time),
            ),
        )
        
        return ds
