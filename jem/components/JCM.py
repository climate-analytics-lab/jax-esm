"""JCM adapter to JEM"""

import numpy as np

import jax_datetime as jdt
from jcm.model import Model
from jcm.forcing import ForcingData
from jcm.physics.speedy.physics_data import PhysicsData

import jax
import jax.numpy as jnp
from jem import constants as constants

from jcm.physics_interface import dynamics_state_to_physics_state


from jem.utils.bulk_op import mean_leaf



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
    return jax.tree_util.tree_map(lambda arr: jnp.array(arr).astype(jnp.float64), tree)

def make_jem_compatible(
    model: Model,
    coupling_timestep: jdt.Timedelta,
    save_interval: jdt.Timedelta = jdt.to_timedelta(1, "day"),
    land_model_active: bool = True,
) -> Model:
    
    timestep = jdt.to_timedelta(int(model.dt_si.to_timedelta().total_seconds()), "second")
   
    if timestep * np.floor(coupling_timestep / timestep) != coupling_timestep:
        raise Exception("Coupling timestep should be a multiple of timestep.")
    
    D2_nodal_shape = model.coords.nodal_shape[1:]
    def initialize():
        _modal_state = asfloat64(model._prepare_initial_modal_state())
        model._final_modal_state = _modal_state
        return (
            asfloat64(model._prepare_initial_modal_state()),
            {
                "phydata" : asfloat64(
                    PhysicsData.zeros(
                        model.coords.horizontal.nodal_shape,
                        model.coords.vertical.layers,
                    )
                ),
                "total_heat_flux" : jnp.zeros(D2_nodal_shape),
            },
            ForcingData.zeros(nodal_shape=model.coords.horizontal.nodal_shape).copy(
                lfluxland = jnp.bool_(land_model_active),
            )
        )


    def generate_step_function(jitted: bool = True):
        # Notice: since save_interval and total_time are claimed
        #         static parameters, we cannot pass in traceable
        #         object. So use item() to convert from scalar
        #         jax.Array to float.
        save_interval_day=(save_interval / jdt.to_timedelta(1, "day")).item() 
        total_time_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item()
        def step_function(state, forcing, step):
            new_atm_modal_state, predictions = model.run_from_state(
                initial_state=state,
                save_interval=save_interval_day,  
                total_time=total_time_day,
                forcing=forcing,
            )
            
            # phydata is a stacked object, so I take the mean here.
            # Howwever, this action will be done by jcm in the new jcm PR.
            phydata = asfloat64(mean_leaf(predictions.physics, axis=0))
            total_heat_flux = - jnp.sum(phydata.surface_flux.hfluxn, axis=2) # upward positive
            
            return (
                new_atm_modal_state,
                {
                    "phydata" : phydata,
                    "total_heat_flux" : total_heat_flux,
                },
                predictions
            )

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
