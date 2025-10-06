"""Speedy Wrapper Class"""

from typing import Dict, Tuple
from dataclasses import dataclass

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
    AbstractComponentState,
)


from jcm.physics_interface import dynamics_state_to_physics_state, physics_state_to_dynamics_state
from dinosaur import primitive_equations, primitive_equations_states
from jcm.physics_interface import PhysicsState
from jcm.model import Predictions

from jax_esm.utils.bulk_op import stack_objects, mean_leaf

import tree_math
from typing import Any

@tree_math.struct
@dataclass
class SpeedyState(AbstractComponentState):
    prog    : PhysicsState
    phydata : Any
    metadata    : primitive_equations_states

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


        self.component_state_class = SpeedyState

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
        D3_nodal_shape = self.model.geometry.nodal_shape
            
        _, init_phydata = self.model.physics.compute_tendencies(
            state      = model.initial_state,
            boundaries = model.boundaries,
            geometry   = model.geometry,
            date       = model._date_from_sim_time(jnp.array(model._final_modal_state.sim_time)),
        )

        # This is a temporary solution to jcm's problem: some of the array's initiated
        # by jcm is int32, but it will change to float32 after step_fn. This causes
        # jax.lax.scan to fail due to data type inconsistency.
        def asfloat32(tree):
            return jax.tree_util.tree_map(lambda arr: arr.astype(jnp.float32), tree)
        
        return SpeedyState(
            prog     = asfloat32(model.initial_state),
            phydata  = asfloat32(init_phydata),
            metadata = model._final_modal_state,
        )

    def gen_step_fn(self):
       
        @jax.jit
        def step_fn(cpl, t):
           
            atm_boundary = self.model.boundaries.copy(
                tsea = cpl.ocn.prog.T
            )

            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state = cpl.atm.metadata,
                save_interval = self.save_interval / 86400.0, # in days
                total_time = self.coupling_timestep / 86400.0, # in days
                boundaries = atm_boundary,
            )

            # phydata is a stacked object. What do I do?
            return SpeedyState(
                prog    = mean_leaf(predictions.dynamics, axis=0),
                phydata = mean_leaf(predictions.physics, axis=0),
                metadata = new_atm_modal_state,
            ), predictions
            
        return step_fn

    def predictions_to_xarray(
        self,
        predictions,
    ):
        return self.model.predictions_to_xarray(predictions)
    
    def report(self):
        pass
        
