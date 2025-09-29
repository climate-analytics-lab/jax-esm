"""Speedy Wrapper Class"""

from typing import Dict, Tuple

import jcm
from jcm.model import Model, get_coords
from jcm.boundaries import boundaries_from_file
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


from jcm.physics_interface import dynamics_state_to_physics_state, physics_state_to_dynamics_state
from dinosaur import primitive_equations, primitive_equations_states
from jcm.physics_interface import PhysicsState
from jcm.model import Predictions

from jax_esm.utils.bulk_op import stack_objects

import tree_math
from typing import Any


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

        self.orog = boundaries.orog

        self.model = Model(**config_speedy) 
        self.initialize()

        
    
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

            if initial_state is None:
                model.initial_state = dynamics_state_to_physics_state(model._final_modal_state, model.primitive)
            
        
        model.start_date = start_date
        model.boundaries = default_boundaries(self.model.coords.horizontal, self.model.orography)


    def getInitState(self):
        
        D3_nodal_shape = self.model.geometry.nodal_shape
        #print("self.model.initial_state = ", self.model.initial_state)
        return_obj = dict(
            modal_state = self.model._final_modal_state,
            state = self.model.initial_state,
            phydata = jcm.physics.speedy.physics_data.PhysicsData.zeros(nodal_shape=D3_nodal_shape[1:], node_levels=D3_nodal_shape[0]),
        )
        
        return return_obj
        
    def gen_step_fn(self):

       
        @jax.jit
        def step_fn(cpl, t):

            # This is where SST is passed
            ocnstate = cpl["ocn"]["state"]
                        
            atm_boundary = self.model.boundaries.copy(
                tsea = ocnstate.T,
            )

            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state = cpl["atm"]["modal_state"],
                save_interval = self.save_interval / 86400.0, # in days
                total_time = self.coupling_timestep / 86400.0, # in days
                boundaries = atm_boundary,
            )

            # phydata is a stacked object. What do I do?
            return dict(
                modal_state = new_atm_modal_state,
                state = predictions.dynamics,
                phydata = jax.tree.map(lambda arr: jnp.mean(arr, axis=0), predictions.physics),
            ), predictions
            
        return step_fn

    def predictions_to_xarray(
        self,
        predictions,
    ):
        return self.model.predictions_to_xarray(predictions)
    
    def report(self):
        pass
        
