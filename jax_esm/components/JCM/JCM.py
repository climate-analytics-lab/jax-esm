"""JCM Wrapper Class"""

from dataclasses import dataclass

import jax_datetime as jdt
from jcm.model import Model as RawJCMModel
from jcm.forcing import ForcingData
from jcm.physics.speedy.physics_data import PhysicsData

import jax
import jax.numpy as jnp
from jax import Array
from jax_esm import constants as constants
from jax_esm.components.base import (
    CoupledComponent,
    CoupledComponentConfig,
)

from jcm.physics_interface import dynamics_state_to_physics_state
from jcm.physics_interface import PhysicsState

from dinosaur import primitive_equations, primitive_equations_states

from jax_esm.utils.bulk_op import mean_leaf
from jax_esm.base.domain import Domain
from jax_esm.base.variable import VariableMetadata, VariableRegistry

import tree_math
from typing import Any, Dict


# This is a temporary solution to jcm's problem: some of the array's initiated
# by jcm is int32, but it will change to float32 after step_function. This causes
# jax.lax.scan to fail due to data type inconsistency.
def asfloat64(tree):
    # return jax.tree_util.tree_map(lambda arr: arr.astype(jnp.float64), tree)

    return jax.tree_util.tree_map(lambda arr: jnp.array(arr).astype(jnp.float64), tree)


@tree_math.struct
@dataclass
class JCMState:
    prog: PhysicsState
    phydata: Any
    extra: Dict[str, Array]
    metadata: primitive_equations_states


class JCM(CoupledComponent):
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
        self.component_forcing_class = ForcingData

        self.state_variable_registry = VariableRegistry([
            VariableMetadata(name="extra.total_heat_flux", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
        ])

        self.forcing_variable_registry = VariableRegistry([
            VariableMetadata(name="sea_surface_temperature", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
            VariableMetadata(name="sice_am", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
            VariableMetadata(name="snowc_am", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
            VariableMetadata(name="soilw_am", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
            VariableMetadata(name="stl_am", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
        ])

    def initialize(
        self,
        initial_state: PhysicsState | primitive_equations.State = None,
        start_date: jdt.Datetime = jdt.to_datetime("2000-01-01"),
    ):
        _modal_state = asfloat64(self.model._prepare_initial_modal_state())
        self.model._final_modal_state = _modal_state
        return JCMState(
            metadata=_modal_state,
            phydata=asfloat64(
                PhysicsData.zeros(
                    self.model.coords.horizontal.nodal_shape,
                    self.model.coords.vertical.layers,
                )
            ),
            prog=dynamics_state_to_physics_state(_modal_state, self.model.primitive),
            extra={
                "total_heat_flux" : jnp.zeros(self.model.coords.horizontal.nodal_shape),
            },
        ), self.component_forcing_class.zeros(nodal_shape=self.model.coords.horizontal.nodal_shape).copy(
            lfluxland = jnp.bool_(True),
        )

    def generate_step_function(self, jitted: bool = True):
        def step_function(state, forcing, t):
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state=state.metadata,
                save_interval=self.save_interval / 86400.0,  # in days
                total_time=self.config.timestep / 86400.0,  # in days
                forcing=forcing,
            )

            phydata = asfloat64(mean_leaf(predictions.physics, axis=0))
            extra = {
                "total_heat_flux" : jnp.sum(phydata.surface_flux.hfluxn, axis=2),
            }
            # phydata is a stacked object, so I take the mean here.
            # Howwever, this action will be done by jcm in the new jcm PR.
            return JCMState(
                prog=asfloat64(mean_leaf(predictions.dynamics, axis=0)),
                phydata=phydata,
                metadata=new_atm_modal_state,
                extra=extra,
            ), predictions

        return jax.jit(step_function) if jitted else step_function

    def validate(self):
        pass

    def predictions_to_xarray(
        self,
        predictions,
    ):
        return predictions.to_xarray()

    def report(self):
        pass
