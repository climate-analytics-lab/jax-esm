"""JCM Wrapper Class"""

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
    create_component_forcing_class,
    create_field_group_class,
)


from jcm.physics_interface import dynamics_state_to_physics_state, physics_state_to_dynamics_state
from jcm.physics_interface import PhysicsState
from jcm.model import Predictions

import dinosaur
from dinosaur import primitive_equations, primitive_equations_states

from jax_esm.utils.bulk_op import stack_objects, mean_leaf
from jax_esm.components.domain import Domain

import tree_math
from typing import Any

from jcm.geometry import get_coords

@tree_math.struct
@dataclass
class JCMState(AbstractComponentState):
    prog    : PhysicsState
    phydata : Any
    metadata    : primitive_equations_states

class JCM(Component):
    
    """
    This is a class wrapping JCM.
    """

    @classmethod
    def generate_default_configuration(cls, grid_specification:str="JCM::T31", layers:int = 8, dict_form = False, topography_file=None, mask_file=None):

        domain = Domain.from_grid_specification(
            grid_specification,
            topography_file = topography_file,
            mask_file = mask_file,
        )

        domain.meta["layers"] = layers

        config_dict = dict(
            name="default_config",
            timestep = (timestep := 86400.0),
            start_dt = datetime(year=2025, month=1, day=1),
            substeps = (substeps := 2),
            save_interval = timestep / substeps,
            domain = domain,
            params=dict(
                
            ),
        )

        if dict_form:
            return config_dict
        else:
            return ComponentConfig(**config_dict)


    @classmethod
    def generate_default_model(cls, grid_specification:str="JCM::T31", layers:int=8, topography_file=None, mask_file=None):

        return cls(
            cls.generate_default_configuration(
                grid_specification=grid_specification,
                layers = layers,
                topography_file=topography_file,
            )
        )


    
    def __init__(
        self,
        config: ComponentConfig,
    ):
        """
        config: Configuration of JCM.
        """

        super().__init__(config)


        self.component_state_class = JCMState

        self.coupling_timestep = config.timestep  # in secs
        self.save_interval = config.save_interval # in secs
        self.substeps = config.substeps
        config_jcm = dict(
            time_step = self.coupling_timestep / self.substeps / 60.0, # in minutes
            orography = self.config.domain.topography,
            spectral_truncation = self.config.domain.meta["horizontal_resolution"],
            layers = self.config.domain.meta["layers"],
        )

        
        self.model = Model(**config_jcm) 

        D3_nodal_shape = self.model.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
 
        self.component_forcing_class = create_component_forcing_class(
            cls_name = "forcing",
            flux_cls = create_field_group_class(
                cls_name = "flux",
                fields = [
                    ("sensible_heat_flux", float, D2_nodal_shape),
                    ("latent_heat_flux", float, D2_nodal_shape),
                ],
            ),
            scalar_cls = create_field_group_class(
                cls_name = "scalar",
                fields = [
                    ("sea_surface_skin_temperature", float, D2_nodal_shape),
                ],
            ),
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

            if initial_state is None:
                model.initial_state = dynamics_state_to_physics_state(model._final_modal_state, model.primitive)
            
        model.start_date = start_date
        model.boundaries = boundaries or default_boundaries(self.model.coords.horizontal)

        # The following code is a solution to have an initial value for phydata by stepping the model one time.
        # The returned phydata is then used for the initial value.
        _, init_phydata = self.model.physics.compute_tendencies(
            state      = model.initial_state,
            boundaries = model.boundaries,
            geometry   = model.geometry,
            date       = model._date_from_sim_time(jnp.array(model._final_modal_state.sim_time)),
        )

        # This is a temporary solution to jcm's problem: some of the array's initiated
        # by jcm is int32, but it will change to float32 after step_function. This causes
        # jax.lax.scan to fail due to data type inconsistency.
        def asfloat32(tree):
            return jax.tree_util.tree_map(lambda arr: arr.astype(jnp.float32), tree)
        
        return JCMState(
            prog     = asfloat32(model.initial_state),
            phydata  = asfloat32(init_phydata),
            metadata = model._final_modal_state,
        )

    def generate_step_function(self, jitted: bool = True):
       
        def step_function(state, forcing, t):
           
            atm_boundary = self.model.boundaries.copy(
                tsea = jnp.repeat(jnp.expand_dims(forcing.scalar.sea_surface_skin_temperature, axis=2), axis=2, repeats=365),
            )
                
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state = state.metadata,
                save_interval = self.save_interval / 86400.0, # in days
                total_time = self.coupling_timestep / 86400.0, # in days
                boundaries = atm_boundary,
            )

            # phydata is a stacked object, so I take the mean here.
            # Howwever, this action will be done by jcm in the new jcm PR.
            return JCMState(
                prog    = mean_leaf(predictions.dynamics, axis=0),
                phydata = mean_leaf(predictions.physics, axis=0),
                metadata = new_atm_modal_state,
            ), predictions
            
        return jax.jit(step_function) if jitted else step_function

    def predictions_to_xarray(
        self,
        predictions,
    ):
        return self.model.predictions_to_xarray(predictions)
    
    def report(self):
        pass
        
