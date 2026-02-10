"""JCM adapter to JEM"""

import numpy as np

from jcm.model import Model
from jcm.forcing import ForcingData
from jcm.forcing import default_forcing
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics_interface import dynamics_state_to_physics_state
from jcm.physics_interface import PhysicsState

import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jem.utils.bulk_op import mean_leaf

def safe_setattr(target, attribute_name, value, *, raise_exception=True):
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
    land_model_active: bool = True,
) -> Model:
    """Adapt the input jcm model to jem framework
    
    This function in-place injects `initialize`, `generate_step_function`, 
    `predictions_to_xarray`, and `get_info` into jcm model object. Also, check
    if jcm's time step `dt_si` can perfectly divide `coupling_timestep`.
    
    """    
     
    timestep = jdt.to_timedelta(int(model.dt_si.to_timedelta().total_seconds()), "second")
   
    if timestep * np.floor(coupling_timestep / timestep) != coupling_timestep:
        raise Exception("Coupling timestep should be a multiple of timestep.")

    D2_nodal_shape = model.coords.nodal_shape[1:]
    def initialize():
        return (
            model._prepare_initial_modal_state(),
            {
                "physics" : asfloat64(
                    PhysicsData.zeros(
                        model.coords.horizontal.nodal_shape,
                        model.coords.vertical.layers,
                    )
                ),
                "total_heat_flux" : jnp.zeros(D2_nodal_shape),
            },
            default_forcing(model.coords.horizontal).copy(
                lfluxland = jnp.bool_(land_model_active),
            )
        )

    def generate_step_function(jitted: bool = True):
        # Notice: since save_interval and total_time are claimed
        #         static parameters, we cannot pass in traceable
        #         object. So use item() to convert from scalar
        #         jax.Array to float.
        save_interval_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item() 
        total_time_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item()
        def step_function(state, forcing, step):
            new_atm_modal_state, predictions = model.run_from_state(
                initial_state=state,
                save_interval=save_interval_day,  
                total_time=total_time_day,
                forcing=forcing,
                output_averages=True,
            )
            physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)
            total_heat_flux = - jnp.sum(physics_no_time_dimension.surface_flux.hfluxn, axis=-1) # upward positive

            # This is a bug in jcm: Time dimension vanishes when save_interval == total_time
            if len(predictions.dynamics.normalized_surface_pressure.shape) == 2:
                nsp = predictions.dynamics.normalized_surface_pressure
                predictions.dynamics.normalized_surface_pressure = jnp.reshape(nsp, (1,) + nsp.shape)
                
            return (
                new_atm_modal_state,
                {
                    "physics" : physics_no_time_dimension,
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

    safe_setattr(model, "initialize", initialize)
    safe_setattr(model, "predictions_to_xarray", predictions_to_xarray)
    safe_setattr(model, "generate_step_function", generate_step_function)
    safe_setattr(model, "get_info", get_info)

    return model
