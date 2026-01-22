"""JCM Wrapper Class"""

from dataclasses import dataclass

import jax_datetime as jdt
from jcm.model import Model as RawJCMModel
from jcm.forcing import ForcingData
from jcm.physics.speedy.physics_data import PhysicsData

import jax
import jax.numpy as jnp
from jax import Array
from jem import constants as constants

from jcm.physics_interface import dynamics_state_to_physics_state
from jcm.physics_interface import PhysicsState

from dinosaur import primitive_equations, primitive_equations_states

from jem.utils.bulk_op import mean_leaf
from jem.base.domain import Domain
from jem.base.variable import VariableMetadata, VariableRegistry

import tree_math
from typing import Any, Dict


def check_before_setattr(target, attribute_name, value, *, raise_exception=True):
    if hasattr(target, attribute_name):
        message = f"Attribute name `{attribute_name:s}` already exists."
        if raise_exception:
            raise Exception(message)
        else:
            print(f"Warning: {message:s}")
    
    setattr(target, attribute_name, value)


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


def make_jem_compatible(model: RawJCMModel, land_model_active: bool, save_interval: float):
    
    timestep = model.dt_si.to_timedelta().total_seconds()
   
    check_before_setattr(model, "timestep", timestep)
    check_before_setattr(model, "component_state_class", JCMState)
    check_before_setattr(model, "component_forcing_class", ForcingData)

    D3_nodal_shape = model.coords.nodal_shape
    D2_nodal_shape = D3_nodal_shape[1:]

    check_before_setattr(model, "state_variable_registry", VariableRegistry([
        VariableMetadata(name="extra.total_heat_flux", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
    ]))

    check_before_setattr(model, "forcing_variable_registry", VariableRegistry([
        VariableMetadata(name="sea_surface_temperature", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
        VariableMetadata(name="sice_am", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
        VariableMetadata(name="snowc_am", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
        VariableMetadata(name="soilw_am", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
        VariableMetadata(name="stl_am", shape=D2_nodal_shape, dimensions=("longitude", "latitude")),
    ]))

    check_before_setattr(model, "domain", Domain.from_grid_specification(
        f"JCM::T{model.coords.horizontal.total_wavenumbers - 2}"
    ))

    def initialize():
        _modal_state = asfloat64(model._prepare_initial_modal_state())
        model._final_modal_state = _modal_state
        return JCMState(
            metadata=_modal_state,
            phydata=asfloat64(
                PhysicsData.zeros(
                    model.coords.horizontal.nodal_shape,
                    model.coords.vertical.layers,
                )
            ),
            prog=dynamics_state_to_physics_state(_modal_state, model.primitive),
            extra={
                "total_heat_flux" : jnp.zeros(model.coords.horizontal.nodal_shape),
            },
        ), ForcingData.zeros(nodal_shape=model.coords.horizontal.nodal_shape).copy(
            lfluxland = jnp.bool_(land_model_active),
        )

    def generate_step_function(jitted: bool = True):
        def step_function(state, forcing, t):
            new_atm_modal_state, predictions = model.run_from_state(
                initial_state=state.metadata,
                save_interval=save_interval / 86400.0,  # in days
                total_time=timestep / 86400.0,  # in days
                forcing=forcing,
            )

            phydata = asfloat64(mean_leaf(predictions.physics, axis=0))
            extra = {
                "total_heat_flux" : - jnp.sum(phydata.surface_flux.hfluxn, axis=2), # upward positive
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

    def predictions_to_xarray(predictions):
        return predictions.to_xarray()

    def get_info():
        return {
            "diffusion" : str(model.diffusion),
        }

    check_before_setattr(model, "initialize", initialize)
    check_before_setattr(model, "predictions_to_xarray", predictions_to_xarray)
    check_before_setattr(model, "generate_step_function", generate_step_function)
    check_before_setattr(model, "get_info", get_info)

    return model
