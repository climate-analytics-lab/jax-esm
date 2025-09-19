"""Speedy Wrapper Class"""

from typing import Dict, Tuple

import jcm
from jcm.model import Model, get_coords
from jcm.boundaries import initialize_boundaries
from jcm.date import DateData, Timestamp, Timedelta
from datetime import datetime
from jcm.boundaries import BoundaryData, default_boundaries

import jax
import jax.numpy as jnp
from jax import Array
from jax_esm import constants as constants
from jax_esm.components.base import (
    Component,
    ComponentConfig,
)


from jax_esm.components.util import createPhysicsStateClass

from jcm.physics_interface import dynamics_state_to_physics_state, physics_state_to_dynamics_state
from dinosaur import primitive_equations, primitive_equations_states
from jcm.physics_interface import PhysicsState
#from jcm.model import Predictions2

import tree_math
from typing import Any

@tree_math.struct
class WrappedSpeedyState:
    times : Any
    state : PhysicsState
    diag  : Any 
    snapshot_modal_state : primitive_equations.State



class Speedy(Component):
    
    """
    This is a class wrapping Speedy.
    """

    
    def __init__(
        self,
        config: ComponentConfig,
    ):
        """
        config: Configuration of Speedy.
        """

        super().__init__(config)

        self.coupling_timestep = config.timestep  # in secs
        self.save_interval = config.save_interval # in secs
        self.substeps = config.substeps
        config_speedy = dict(
            time_step = self.coupling_timestep / self.substeps / 60.0, # in minutes
        )

        if "boundaries" in config.params and config.params["boundaries"] is not None:
            boundaries = config.params["boundaries"]
            config_speedy["orography"] = boundaries.orog

        self.model = Model(**config_speedy)
        
        self.stateDiagClass = WrappedSpeedyState
        self.initialize()
        self.trajectory = []
        
        self.state_diag = WrappedSpeedyState(
            snapshot_modal_state = self.model._final_modal_state,
            state = self.model.initial_state,
            diag  = None,
            times = None,
        )

        print("Is snapshot None? ", self.state_diag.snapshot_modal_state is None)
    
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
        model.boundaries = default_boundaries(self.model.coords.horizontal, self.model.orography)

        

    def record(self, state_diag):
        
        self.state_diag = state_diag

        copy_state_diag = WrappedSpeedyState(
            snapshot_modal_state = state_diag.snapshot_modal_state,
            state = state_diag.state.copy(),
            diag = state_diag.diag.copy(),
            times = state_diag.times.copy(),
        )

        
        self.trajectory.append(copy_state_diag)

        
    def genForwardFunc(self, begin_time):

        @jax.jit
        def forward_func(cpl):

            # This is where SST is passed
            ocnstate = cpl.ocn.state
                        
            atm_boundary = self.model.boundaries.copy(
                tsea = ocnstate.T,
            )

            # The current `resume` function is not jittable
            # Therefore, I create a function `genForwardFunc` that
            # essentially is a jittable `resume` function.
            new_snapshot_modal_state, predictions = self.model.run_from_state(
                initial_state = cpl.atm.snapshot_modal_state,
                save_interval = self.save_interval / 86400.0, # in days
                total_time = self.coupling_timestep / 86400.0, # in days
                boundaries = atm_boundary,
            )
            
            new_atm = self.stateDiagClass(
                times = predictions.times,
                state = predictions.dynamics,
                diag  = predictions.physics,
                snapshot_modal_state = new_snapshot_modal_state,
            )
            
            return new_atm
                
        return forward_func
    
    def report(self):
        pass
        #print("Atmoshpere temperature = ", self.state.T[0]) 
        