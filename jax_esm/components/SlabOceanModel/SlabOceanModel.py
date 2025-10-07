"""Slab ocean model component."""

from typing import Dict, Tuple, Any, List

import jax
import jax.numpy as jnp

from jax_esm import constants as constants
from jax_esm.utils.bulk_op import stack_objects

from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from jcm.boundaries import _fixed_ssts as gen_idealized_sst


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

            print("Boundary exists. The given initial SST will be used.")
            print("Boundary file: ", config.params["boundary_file"])
            
            boundaries = config.params["boundaries"]
            thrsh = 0.3

            self.SST_clim = jnp.array(xr.open_dataset(config.params["boundary_file"])["sst"])
                            
            # Update fmask_lnd based on the conditions
            fmask_lnd = jnp.where(
                boundaries.fmask >= thrsh,
                1.0,
                0.0,
            )

            fmask_ocn = 1.0 - fmask_lnd
            
            init_T = self.SST_clim[:, :, 0].copy().at[fmask_ocn == 0].set(273.15+15)
            
            if jnp.any( jnp.isnan(init_T) == (fmask_ocn == 0) ):
                print("fmask_ocn and init_T do share the same mask.")
            else:
                raise Exception("Warning: fmask_ocn and sst_init do not share the same mask.")

            self.fmask_ocn = fmask_ocn
            
        else:
            print("Boundary does not exist. Idealized initial SST will be used.")
            init_T = gen_idealized_sst(self.coords.horizontal)

        
        # Compute heat capacity cd, and time factor for Euler backward scheme
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
            
            clim_day_beg = start_dt_offset + days_after_start
            clim_day_end = jnp.mod(clim_day_beg + 1, self.SST_clim.shape[2])
            
            snapshot_SST_clim_beg = self.SST_clim[:, :, clim_day_beg]
            snapshot_SST_clim_beg = jnp.where(self.fmask_ocn != 0, snapshot_SST_clim_beg, 273.15+15)
            
            snapshot_SST_clim_end = self.SST_clim[:, :, clim_day_end]
            snapshot_SST_clim_end = jnp.where(self.fmask_ocn != 0, snapshot_SST_clim_end, 273.15+15)

            SST_clim_trend = (snapshot_SST_clim_end - snapshot_SST_clim_beg) / 86400.0

            new_Tanom = cpl.ocn.prog.T - snapshot_SST_clim_beg

            new_sim_time = cpl.ocn.prog.sim_time
            for step in range(self.substeps):
                new_Tanom = self.time_factor * ( new_Tanom + self.cd_factor * ( - (
                    cpl.flx.phydata.heatflx
                )))
                new_sim_time += self.subtimestep
                
                
            new_T = new_Tanom + snapshot_SST_clim_beg + SST_clim_trend * self.config.timestep

            new_state = cpl.ocn.copy(
                prog_kwargs = dict(
                    T = new_T,
                    sim_time = new_sim_time,
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
        prog    = predictions["prog"]
        ds = xr.Dataset(
            data_vars = dict(
                T   = (["time", "lon", "lat"], prog.T),
                mld = (["time", "lon", "lat"], prog.mld),
            ), 
            coords = dict(
                time = (["time",], prog.sim_time),
            ),
        )
        
        return ds
