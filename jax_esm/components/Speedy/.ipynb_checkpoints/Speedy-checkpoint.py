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


from jax_esm.components.util import createPhysicsStateClass

from jcm.physics_interface import dynamics_state_to_physics_state, physics_state_to_dynamics_state
from dinosaur import primitive_equations, primitive_equations_states
from jcm.physics_interface import PhysicsState
from jcm.model import Predictions2

class Speedy(Component):
    """Simple slab ocean model with prescribed mixed layer depth.
    
    This model integrates SST anomalies based on surface heat fluxes
    and relaxes towards a prescribed climatology.
    """

    
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
        
        self.stateDiagClass = Predictions2
        self.initialize()
        self.trajectory = []
        
        self.state_diag = Predictions2(
            modal_state = self.model._final_modal_state,
            dynamics = self.model.initial_state,
            physics  = None,
            times = None,
        )

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
            model._final_modal_state = initial_state
        else:
            model.initial_state = initial_state
            model._final_modal_state = model._prepare_initial_modal_state(initial_state)
        
        model.start_date = start_date
        model.boundaries = model._prepare_boundaries()
        

    def record(self, state_diag):
        
        self.state_diag = state_diag

        copy_state_diag = Predictions2(
            modal_state = state_diag.modal_state,
            physics = state_diag.physics.copy(),
            dynamics = state_diag.dynamics.copy(),
            times = state_diag.times.copy(),
        )

        
        self.trajectory.append(copy_state_diag)

        
    def genForwardFunc(self, begin_time):

        @jax.jit
        def forward_func(cplstate):
            
            ocnstate = cplstate.ocn.state
            atm_boundary = self.model.boundaries.copy(
                tsea = ocnstate.T,
            )
            
            integrate_fn = self.model.genIntegrateFn(
                sim_time = cplstate.atm.modal_state.sim_time,
                save_interval = self.save_interval / 86400.0, # in days
                total_time = self.coupling_timestep / 86400.0, # in days
                boundaries = atm_boundary,
            )
            
            predictions2 = integrate_fn(cplstate.atm)
            
            
            return predictions2
                
        return forward_func
    
    def report(self):
        pass
        #print("Atmoshpere temperature = ", self.state.T[0]) 
        