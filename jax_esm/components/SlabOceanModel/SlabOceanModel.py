"""Slab ocean model component."""

from typing import Dict, Tuple, Any
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax import Array
from jax_esm import constants as constants
from jax_esm.components.util import createPhysicsStateClass, createStateDiagClass
from jax_esm.components.util import stack_objects

from jcm.geometry import Geometry
import tree_math

from dataclasses import make_dataclass

from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

from jax_esm.components.base import (
    Component,
    ComponentConfig,
)



class SlabOceanModel(Component):
    """
    
    Slab ocean model with prescribed mixed layer depth and climatology.

    """

    @classmethod
    def createStateClass(
        cls,
        D2_nodal_shape: Tuple,
        D3_nodal_shape: Tuple,
        cls_name      : str = "SOMState",
    ):
        """
        It creates a state class dynamically with given dimension.
        This class will have the following methods: `zeros`, `ones`, and `copy`. 

        Args:
        
            D2_nodal_shape: Two-dimension shape tuple (y, x).
            D3_nodal_shape: Three-dimension shape tuple (z, y, x).

        Returns:

            stateClass: The resulting state class.
        
        """
        
        new_cls = createPhysicsStateClass(
            cls_name = cls_name,
            fields = [
                ("T", float, D2_nodal_shape),
                ("mld", float, D2_nodal_shape),
            ],
        )
    
        return new_cls


    @classmethod
    def createDiagClass(
        cls,
        D2_nodal_shape,
        D3_nodal_shape,
        cls_name = "SOMSDiag",
    ):
        
        """
        It creates a diag class dynamically with given dimension.
        This class will have the following methods: `zeros`, `ones`, and `copy`. 

        Args:
        
            D2_nodal_shape: Two-dimension shape tuple (y, x).
            D3_nodal_shape: Three-dimension shape tuple (z, y, x).

        Returns:

            stateClass: The resulting state class.
        
        """        
        new_cls = createPhysicsStateClass(
            cls_name = cls_name,
            fields = [
                ("heatflx", float, D2_nodal_shape),
            ],
        )
    
        return new_cls


        
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

        self.timestep = config.timestep
        self.substeps = config.substeps
        self.subtimestep = self.timestep / self.substeps

        
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        
        self.stateClass = self.__class__.createStateClass(
            D2_nodal_shape = D2_nodal_shape,
            D3_nodal_shape = D3_nodal_shape,
        )

        self.diagClass = self.__class__.createDiagClass(
            D2_nodal_shape = D2_nodal_shape,
            D3_nodal_shape = D3_nodal_shape,
        )

        self.stateDiagClass = createStateDiagClass(
            state_cls = self.stateClass,
            diag_cls = self.diagClass,
        )
        self.state_diag = self.stateDiagClass.zeros()

        # =========================================================================
        # Initialize slab ocean model boundary conditions
        # =========================================================================

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

        #lat_rad = grid.latitudes
        #lon_rad = grid.longitudes
        
        #llon_rad, llat_rad = jnp.meshgrid(lon_rad, lat_rad, indexing="ij")
        
        # initialize mld
        mld_max = config.params["mld_max"] if "mld_max" in config.params else 60.0
        mld_min = config.params["mld_min"] if "mld_min" in config.params else 40.0

        
        init_mld = mld_max + (mld_min - mld_max) * jnp.cos(llat_rad)**3
        init_T = None
        self.SST_clim = None
        self.fmask_ocn = jnp.ones_like(init_mld)
        
        if "boundaries" in config.params and config.params["boundaries"] is not None:

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

        self.state_diag = self.state_diag.copy(
            state_kwargs = dict(
                mld = init_mld,
                T = init_T,
            ),
        )

        
        self.trajectory = []
        

    def initialize(self):
        self.trajectory = []
    
    def run(self, master=None):
        pass            

    def record(self, state_diag):
        self.state_diag = state_diag
        self.trajectory.append(state_diag.copy())
    
    def genForwardFunc(
        self,
        begin_time,
    ):

        # Find day of the year to locate climatology
        begin_time_dt = pd.Timestamp(begin_time.to_datetime64())
        ref_dt = pd.Timestamp(year=begin_time_dt.year, month=begin_time_dt.month, day=1)
        day_of_year = int(np.floor( ( begin_time_dt - ref_dt ) / pd.Timedelta(days=1) ))
        snapshot_SST_clim = self.SST_clim[:, :, day_of_year].at[self.fmask_ocn == 0].set(273.15+15)
        
        @jax.jit
        def forward_func(cplinfo):

            somstate = cplinfo.ocn.state
            fmstate  = cplinfo.flx.state

            new_Tanom = somstate.T - snapshot_SST_clim
            for step in range(self.substeps):
                new_Tanom = self.time_factor * ( new_Tanom + self.cd_factor * ( - (
                    fmstate.hfluxn[:, :, 0]
                )))
            
            new_T = new_Tanom + snapshot_SST_clim
            
            new_state_diag = cplinfo.ocn.copy(
                state_kwargs = dict(T = new_T),
            )

            return new_state_diag
            
        return forward_func

    def convertTrajectoryToXarray(
        self,
        trajectory = None,
    ):
        """
        A tool function that convert a trajectory into an xarray Dataset.

        Args:
        
            trajectory : A list of object of class `self.stateDiagClass`. If None is given, then use `self.trajectory`. 

        Returns:

            ds : The resulting xarray dataset.
        """
        if trajectory is None:
            trajectory = self.trajectory
        
        stacked = stack_objects(trajectory)  
        ds = xr.Dataset(
            data_vars = dict(
                T   = (["time", "lon", "lat"], stacked.state.T),
                mld = (["time", "lon", "lat"], stacked.state.mld),
            ),
        )
        
        return ds
        
    
    def report(self):
       print("Ocean temperature = ", self.state.T[0]) 
