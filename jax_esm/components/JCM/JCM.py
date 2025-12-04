"""JCM Wrapper Class"""

from dataclasses import dataclass

import jax_datetime as jdt
from jcm.model import Model as RawJCMModel
from jcm.forcing import ForcingData, default_forcing

import jax
import jax.numpy as jnp
from jax_esm import constants as constants
from jax_esm.components.base import (
    Component,
    CoupledComponentConfig,
    ComponentState,
    create_component_forcing_class,
    create_field_group_class,
)


from jcm.physics_interface import dynamics_state_to_physics_state
from jcm.physics_interface import PhysicsState

from dinosaur import primitive_equations, primitive_equations_states

from jax_esm.utils.bulk_op import mean_leaf
from jax_esm.components.domain import Domain

import tree_math
from typing import Any


# This is a temporary solution to jcm's problem: some of the array's initiated
# by jcm is int32, but it will change to float32 after step_function. This causes
# jax.lax.scan to fail due to data type inconsistency.
def asfloat64(tree):
    return jax.tree_util.tree_map(lambda arr: arr.astype(jnp.float64), tree)


@tree_math.struct
@dataclass
class JCMState(ComponentState):
    prog: PhysicsState
    phydata: Any
    metadata: primitive_equations_states


class JCM(Component):
    """
    This is a class wrapping JCM.
    """

    def __init__(
        self,
        model: RawJCMModel,
        coupling_timestep: float = 86400.0,
        save_interval: float = 86400.0,
    ):
        """
        config: Configuration of JCM.
        """

        self.model = model
        super().__init__(
            CoupledComponentConfig(
                name="JCM",
                timestep=coupling_timestep,
            )
        )

        self.domain = Domain.from_grid_specification(
            f"JCM::T{model.coords.horizontal.total_wavenumbers - 2}"
        )
        self.save_interval = save_interval

        if save_interval > coupling_timestep:
            raise ValueError("Error: `save_interval` is larger than model timestep. ")

        D3_nodal_shape = self.model.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]

        self.component_state_class = JCMState
        self.component_forcing_class = create_component_forcing_class(
            cls_name="forcing",
            flux_cls=create_field_group_class(
                cls_name="flux",
                fields=[],
            ),
            scalar_cls=create_field_group_class(
                cls_name="scalar",
                fields=[
                    ("sea_surface_temperature", float, D2_nodal_shape),
                ],
            ),
        )

    def initialize(
        self,
        initial_state: PhysicsState | primitive_equations.State = None,
        forcing: ForcingData = None,
        start_date: jdt.Datetime = jdt.to_datetime("2000-01-01"),
    ):
        model = self.model

        # Copy from jax-gcm jcm/model.py
        if isinstance(initial_state, primitive_equations.State):
            model.initial_state = dynamics_state_to_physics_state(
                initial_state, model.primitive
            )
            model._final_modal_state = initial_state
        else:
            model.initial_state = initial_state
            model._final_modal_state = model._prepare_initial_modal_state(initial_state)

            if initial_state is None:
                model.initial_state = dynamics_state_to_physics_state(
                    model._final_modal_state, model.primitive
                )

        model.start_date = start_date
        model.forcing = forcing or default_forcing(self.model.coords.horizontal)

        # The following code is a solution to have an initial value for phydata by stepping the model one time.
        # The returned phydata is then used for the initial value.
        _, init_phydata = self.model.physics.compute_tendencies(
            state=model.initial_state,
            forcing=model.forcing,
            geometry=model.geometry,
            date=model._date_from_sim_time(
                jnp.array(model._final_modal_state.sim_time)
            ),
        )

        return JCMState(
            prog=asfloat64(model.initial_state),
            phydata=asfloat64(init_phydata),
            metadata=model._final_modal_state,
        )

    def generate_step_function(self, jitted: bool = True):
        def step_function(state, forcing, t):
            atm_forcing = self.model.forcing.copy(
                sea_surface_temperature=forcing.scalar.sea_surface_temperature,
            )
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state=state.metadata,
                save_interval=self.save_interval / 86400.0,  # in days
                total_time=self.config.timestep / 86400.0,  # in days
                forcing=atm_forcing,
            )

            # phydata is a stacked object, so I take the mean here.
            # Howwever, this action will be done by jcm in the new jcm PR.
            return JCMState(
                prog=asfloat64(mean_leaf(predictions.dynamics, axis=0)),
                phydata=asfloat64(mean_leaf(predictions.physics, axis=0)),
                metadata=new_atm_modal_state,
            ), predictions

        return jax.jit(step_function) if jitted else step_function

    def predictions_to_xarray(
        self,
        predictions,
    ):
        return self.model.predictions_to_xarray(predictions)

    def report(self):
        pass
