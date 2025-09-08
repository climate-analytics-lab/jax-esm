"""Slab ocean model component."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax_esm import constants as constants
from jax_esm.components.base import (
    BoundaryFluxes,
    Component,
    ComponentConfig,
    ComponentState,
)
from jax_esm.components.util import CreatePhysicsStateClass

class SlabAtmosphereModel(Component):
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
        
        StateClass = createPhysicsStateClass(
            cls_name = "SAMState",
            fields = [
                ("T", float, D2_nodal_shape),
                ("U", float, D2_nodal_shape),
            ],
        )
    
        return StateClass

    def __init__(
        self,
        config: ComponentConfig,
    ):
        """Initialize slab atmosphere model."""
        
        super().__init__(config)
        
        self.coords = config.params["coords"]
        
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

        self.column_mass = constants.atm_column_mass   # Mass of entire atmosphere column per unit area (kg / m^2)
        self.cp = constants.atm_cp                    # Air specific heat capacity (J/kg/K)


    def run(self, master=None):
        #print("Slab atmosphere run.")

        dc = self.data_center
        lwflx_toa = dc.getVariable(component="flx", varname="lwflx_toa", by_component="atm")
        swflx_toa = dc.getVariable(component="flx", varname="swflx_toa", by_component="atm")
        swflx_sfc = dc.getVariable(component="flx", varname="swflx_sfc", by_component="atm")
        lhflx     = dc.getVariable(component="flx", varname="lhflx", by_component="atm")
        
        #flux_model = master.components["flx"]
        time_step = master.config["time_step"]
        
        new_T = self.state.T + time_step *( - (lwflx_toa + swflx_toa - swflx_sfc - lhflx) / ( self.column_mass * self.cp ) )       
        self.state = self.state.copy(
            T = new_T,
        )
    
    def genForwardFunc(self):
    
        @jax.jit
        def forward_func(cplstate):
            
            samstate = cplstate.atm
            fmstate = cplstate.flx
            
            lwflx_toa = fmstate.lwflx_toa
            swflx_toa = fmstate.swflx_toa
            swflx_sfc = fmstate.swflx_sfc
            lhflx     = fmstate.lhflx

            new_T = samstate.T
            for step in range(self.substeps):
                #new_T = new_T + self.subtimestep *( - (lwflx_toa + swflx_toa - swflx_sfc - lhflx) / ( self.column_mass * self.cp ) )
                new_T = new_T + self.subtimestep *( - (lwflx_toa + swflx_toa - swflx_sfc - lhflx ) / ( self.column_mass * self.cp ) )      
            
            new_samstate = samstate.copy(
                T = new_T,
            )

            return new_samstate

        return forward_func

            
    def report(self):
       print("Atmoshpere temperature = ", self.state.T[0]) 
        