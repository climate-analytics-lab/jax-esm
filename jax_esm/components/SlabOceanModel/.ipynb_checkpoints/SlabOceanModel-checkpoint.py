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

from jax_esm.components.base import (
    BoundaryFluxes,
    Component,
    ComponentConfig,
    ComponentState,
)

import xarray as xr


class SlabOceanModel(Component):
    """Simple slab ocean model with prescribed mixed layer depth.
    
    This model integrates SST anomalies based on surface heat fluxes
    and relaxes towards a prescribed climatology.
    """

    @classmethod
    def createStateClass(
        cls,
        D2_nodal_shape,
        D3_nodal_shape,
        cls_name = "SOMState",
    ):
        
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

        # initialize mld
        mld_max = config.params["mld_max"] if "mld_max" in config.params else 60.0
        mld_min = config.params["mld_min"] if "mld_min" in config.params else 40.0
        
        T_max = config.params["T_max"] if "T_max" in config.params else 273.15 + 30.0
        T_min = config.params["T_min"] if "T_min" in config.params else 273.15 + 5.0


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

        
        init_mld = mld_max + (mld_min - mld_max) * jnp.cos(llat_rad)**3
        init_T   = T_min + (T_max - T_min) * jnp.cos(llat_rad - 20 * jnp.pi / 180.0)**3 + 5.0 * jnp.cos(llon_rad)
        
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
    
    def genForwardFunc(self, begin_time):
        
        @jax.jit
        def forward_func(cplinfo):

            somstate = cplinfo.ocn.state
            fmstate  = cplinfo.flx.state
            
            new_T = somstate.T
            for step in range(self.substeps):
                new_T = new_T + self.subtimestep * ( - (
 #                   fmstate.swflx_sfc +
 #                   fmstate.lhflx
                    fmstate.hfluxn[:, :, 0]
                ) / ( somstate.mld * self.ocn_rho * self.ocn_cp ) )

            
            new_state_diag = cplinfo.ocn.copy(
                state_kwargs = dict(T = new_T),
            )

            return new_state_diag
            
        return forward_func

    def convertTrajectoryToXarray(
        self,
        trajectory = None,
    ):
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
