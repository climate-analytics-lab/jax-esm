"""Speedy Wrapper Class"""

from typing import Dict, Tuple

import jcm
from jcm.model import Model, get_coords
from jcm.boundaries import initialize_boundaries
from jcm.date import DateData, Timestamp, Timedelta
from datetime import datetime
from jcm.boundaries import BoundaryData, default_boundaries, populate_parameter_dependent_boundaries

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
from jcm.model import Predictions

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

        self.coupling_timestep = config.timestep  # in secs
        self.save_interval = config.save_interval # in secs
        self.substeps = config.substeps
        config_speedy = dict(
            time_step = self.coupling_timestep / self.substeps / 60.0, # in minutes
        )

        self.model = Model(**config_speedy)
        
        #D3_nodal_shape = self.model.coords.nodal_shape
        #D2_nodal_shape = D3_nodal_shape[1:]
        #self.stateClass = Speedy.createStateClass(
        #    D2_nodal_shape = D2_nodal_shape,
        #    D3_nodal_shape = D3_nodal_shape,
        #)

        #state_dynamics = self.model.get_initial_state()
        #print("Type of state_dynamics: ", type(state_dynamics))
        
        self.stateClass = Predictions

        self.initialize()

        """
        init_dynamics_state = self.model._prepare_initial_state()

        print("type of init_dynamics_state: ", type(init_dynamics_state))
        init_state = dynamics_state_to_physics_state(init_dynamics_state, self.model.primitive)
        """
        
        print("type of init_state: ", type(self.model._final_state_internal))
        self.state = Predictions(
            dynamics = self.model._final_state_internal,
            physics  = None,
            times = None,
        )

    def run(self, master=None):

        final_state, pred = self.model.unroll(self.speedy_holder["tmp_state"])
        self.speedy_holder["tmp_state"] = final_state
        self.speedy_holder["tmp_pred"]  = pred

    def initialize(
        self,
        initial_state: PhysicsState | primitive_equations.State = None,
        boundaries: BoundaryData = None,
        start_date: Timestamp = Timestamp.from_datetime(datetime(2000, 1, 1)),
    ):

        model = self.model
        # Copy from jax-gcm jcm/model.py
        if isinstance(initial_state, primitive_equations.State):
            model.initial_state = dynamics_state_to_physics_state(initial_state, model.primitive)
            model._final_state_internal = initial_state
        else:
            model.initial_state = initial_state
            model._final_state_internal = model._prepare_initial_state(initial_state)
        
        model.start_date = start_date
        model.boundaries = model._prepare_boundaries()
    
    def genForwardFunc(self):

        #@jax.jit
        def forward_func(cplstate):
            
            atmstate = cplstate.atm
            fmstate = cplstate.flx
            ocnstate = cplstate.ocn
            atm_boundary = self.model.boundaries.copy(
                tsea = ocnstate.T,
            )
                
            #print("Begin type(_final_state_internal): ", type(self.model._final_state_internal))
            #print(dir(self.model._final_state_internal))

            #print("type(cplstate.atm.dynamics) = ", type(cplstate.atm.dynamics))
            #self.model._final_state_internal = physics_state_to_dynamics_state(cplstate.atm.dynamics, self.model.primitive)
            final_state = self.model.resume(
                save_interval = self.save_interval / 86400.0, # in days
                total_time = self.coupling_timestep / 86400.0, # in days
                boundaries = atm_boundary,
            )
            
            #print(dir(self.model._final_state_internal))
            #print("End type(_final_state_internal): ", type(self.model._final_state_internal))
            return final_state
                
        return forward_func
    
    def report(self):
        pass
        #print("Atmoshpere temperature = ", self.state.T[0]) 
        