"""Slab ocean model component."""

from typing import Dict, Tuple
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax import Array
from jax_esm import constants as constants
from jax_esm.components.PhysicsState import CreatePhysicsStateClass




from jax_esm.components.base import (
    BoundaryFluxes,
    Component,
    ComponentConfig,
    ComponentState,
)

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
    ):
        
        SOMStateClass = CreatePhysicsStateClass(
            cls_name = "SOMState",
            fields = [
                ("T", float, D2_nodal_shape),
                ("mld", float, D2_nodal_shape),
            ],
        )
    
        return SOMStateClass

    
    def __init__(
        self,
        config: ComponentConfig,
    ):
        """Initialize slab ocean model."""
        
        super().__init__(config)

        self.coords = config.params["coords"]
        #namedtuple( "Coords", ["nodal_shape"] )(nodal_shape=(2,2,1))
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

        self.state = self.stateClass.zeros()
        self.state = self.state.copy(
            mld = self.state.mld * 0 + 50.0  # Mixed layer depth (m)
        )
        
        self.ocn_rho = constants.ocn_rho # Seawater density (kg / m^3)
        self.ocn_cp = constants.ocn_cp   # Seawater specific heat capacity (J/kg/K)

        
        """
        # Physical constants
        
        # Model parameters
        self.mixed_layer_depth = config.params.get("mixed_layer_depth", 50.0)  # m
        self.relaxation_time = config.params.get("relaxation_time", 30.0)  # days
        
        # Grid info
        self.nlat = config.grid["nlat"]
        self.nlon = config.grid["nlon"]
        
        # Precompute factors for efficiency
        self.heat_capacity = self.ocn_rho * self.ocn_cp * self.mixed_layer_depth
        self.cd_factor = 1.0 / self.heat_capacity  # K/(W/m²)/s
        
        # Time factor for anomaly evolution (per day)
        self.time_factor_per_day = jnp.exp(-1.0 / self.relaxation_time)
        """

    def run(self, master=None):

        flux_model = master.components["flx"]
        subtimestep = self.timestep / self.substeps

        for step in range(self.subtimestep):
            
            new_T = self.state.T + self.subtimestep * ( - (
                flux_model.state.swflx_sfc +
                flux_model.state.lhflx
            ) / ( self.state.mld * self.ocn_rho * self.ocn_cp ) )
        
            self.state = self.state.copy(
                T = new_T,
            )
            

    def genForwardFunc(self):
        @jax.jit
        def forward_func(cplstate):

            somstate = cplstate.ocn
            fmstate  = cplstate.flx

            new_T = somstate.T
            
            for step in range(self.substeps):
                new_T = new_T + self.subtimestep * ( - (
                    fmstate.swflx_sfc +
                    fmstate.lhflx
                ) / ( somstate.mld * self.ocn_rho * self.ocn_cp ) )
            
            new_somstate = somstate.copy(
                T = new_T,
            )

            return new_somstate

        return forward_func
                
    def report(self):
       print("Ocean temperature = ", self.state.T[0]) 
