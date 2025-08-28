"""Speedy Wrapper Class"""

from typing import Dict, Tuple

import jcm
from jcm.model import Model, get_coords
from jcm.boundaries import initialize_boundaries

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


from jax_esm.components.PhysicsState import CreatePhysicsStateClass

from jcm.physics_interface import dynamics_state_to_physics_state, physics_state_to_dynamics_state
from dinosaur import primitive_equations, primitive_equations_states
from jcm.physics_interface import PhysicsState

class Speedy(Component):
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
        
        SpeedyStateClass = CreatePhysicsStateClass(
            cls_name = "SpeedyState",
            fields = [
                ("u_wind", float, D3_nodal_shape),
                ("v_wind", float, D3_nodal_shape),
                ("temperature", float, D3_nodal_shape),
                ("specific_humidity", float, D3_nodal_shape),
                ("geopotential", float, D3_nodal_shape),
                ("normalized_surface_pressure", float, D3_nodal_shape),
                ("surface_temperature", float, D2_nodal_shape),
                ("sw_sfc", float, D2_nodal_shape),
                ("lw_sfc", float, D2_nodal_shape),
            ],
        )
    
        return SpeedyStateClass
    
    def __init__(
        self,
        config: ComponentConfig,
    ):

        super().__init__(config)

        config_speedy = dict(
            time_step = config.timestep / config.substeps / 60.0, # in minutes
            save_interval = config.save_interval / 86400.0,       # in days
            total_time = config.timestep / 86400.0,               # in days
        )


        
        self.model = Model(**config_speedy)
        
        #D3_nodal_shape = self.model.coords.nodal_shape
        #D2_nodal_shape = D3_nodal_shape[1:]
        #self.stateClass = Speedy.createStateClass(
        #    D2_nodal_shape = D2_nodal_shape,
        #    D3_nodal_shape = D3_nodal_shape,
        #)

        state_dynamics = self.model.get_initial_state()
        print("Type of state_dynamics: ", type(state_dynamics))
        
        self.stateClass = PhysicsState #primitive_equations.State
        self.state = dynamics_state_to_physics_state(
            state_dynamics,
            self.model.primitive,
        )
        self.state_dynamics = state_dynamics
        
 

    def run(self, master=None):

        final_state, pred = self.model.unroll(self.speedy_holder["tmp_state"])
        self.speedy_holder["tmp_state"] = final_state
        self.speedy_holder["tmp_pred"]  = pred

    def genForwardFunc(self):
        
        @jax.jit
        def forward_func(atmstate, fmstate):

            #new_atmstate = atmstate
            #atmstate_dynamics = physics_state_to_dynamics_state(atmstate, self.model.primitive)
            final_state, pred = self.model.unroll(atmstate_dynamics)   # Error
            #new_atmstate = dynamics_state_to_physics_state(final_state, self.model.primitive)
            
            return new_atmstate

        return forward_func
    
    def report(self):
        pass
        #print("Atmoshpere temperature = ", self.state.T[0]) 
        