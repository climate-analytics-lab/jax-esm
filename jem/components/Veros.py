"""JCM adapter to JEM"""

from dataclasses import dataclass
import numpy as np

import jax_datetime as jdt
import jax
import jax.numpy as jnp
from jax import Array

from jem.utils.bulk_op import mean_leaf

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

@tree_math.struct
@dataclass
class JCMState:
    prog: PhysicsState
    phydata: Any
    extra: Dict[str, Array]
    metadata: primitive_equations_states

def make_jem_compatible(
    model: Model,
    coupling_timestep: jdt.Timedelta,
    save_interval: jdt.Timedelta = jdt.to_timedelta(1, "day"),
    land_model_active: bool = True,
) -> Model:
    
    timestep = jdt.to_timedelta(int(model.dt_si.to_timedelta().total_seconds()), "second")
    if timestep * np.floor(coupling_timestep / timestep) != coupling_timestep:
        raise Exception("Coupling timestep should be a multiple of timestep.")
    
    check_before_setattr(model, "component_state_class", JCMState)
    check_before_setattr(model, "component_forcing_class", ForcingData)

    D3_nodal_shape = model.coords.nodal_shape
    D2_nodal_shape = D3_nodal_shape[1:]
    D2_information = (D2_nodal_shape, ("longitude", "latitude"))

    check_before_setattr(model, "state_variable_registry", {
        "extra.total_heat_flux" : D2_information,
    })

    check_before_setattr(model, "forcing_variable_registry", {
        varname : D2_information for varname in [
            "sea_surface_temperature",
            "sice_am",
            "snowc_am",
            "soilw_am",
            "stl_am",
        ]
    })

    #check_before_setattr(model, "grids", Grids.generate_grids_from_grid_specification(
    #    f"JCM::T{model.coords.horizontal.total_wavenumbers - 2}"
    #))

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
        # Notice: since save_interval and total_time are claimed
        #         static parameters, we cannot pass in traceable
        #         object. So use item() to convert from scalar
        #         jax.Array to float.


        def step_function(state, forcing, step): 
            n_state = state.copy()
            acc.step(n_state)
            
            return n_state, stack_objects(dict(
                
            )) 


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
