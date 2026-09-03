"""JCM adapter to JEM"""

from typing import Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import tree_math
from jcm.forcing import default_forcing
from jcm.model import Model


def safe_setattr(target, attribute_name, value, *, raise_exception=True):
    if hasattr(target, attribute_name):
        message = f"Attribute name `{attribute_name:s}` already exists."
        if raise_exception:
            raise AttributeError(message)
        else:
            print(f"Warning: {message:s}")

    setattr(target, attribute_name, value)


@tree_math.struct
class JCMDerived:
    physics: Any  # jcm's own PhysicsCarryState -- opaque passthrough, shape/keys
                  # depend on which physics terms are configured on the Model
    total_heat_flux: jnp.ndarray
    total_freshwater_flux: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        physics,
        total_heat_flux=None,
        total_freshwater_flux=None,
    ):
        return cls(
            physics,
            total_heat_flux if total_heat_flux is not None else jnp.zeros(shape),
            total_freshwater_flux if total_freshwater_flux is not None else jnp.zeros(shape),
        )

def make_jem_compatible(
    model: Model,
    coupling_timestep: jdt.Timedelta,
) -> Model:
    """Adapt the input jcm model to jem framework
    
    This function in-place injects `initialize`, `generate_step_function`, 
    `predictions_to_xarray`, and `get_info` into jcm model object. Also, check
    if jcm's time step `dt_si` can perfectly divide `coupling_timestep`.
    
    """    
   
    # Check if couopling_timestep is a multiple of jcm's native timestep
    timestep = jdt.to_timedelta(int(model.dt_si.to_timedelta().total_seconds()), "second")
    if timestep * np.floor(coupling_timestep / timestep) != coupling_timestep:
        raise ValueError("Coupling timestep should be a multiple of timestep.")

    D2_nodal_shape = model.coords.nodal_shape[1:]
    def initialize():

        state=model._prepare_initial_dycore_state()
        forcing = default_forcing(model.coords.horizontal)
        
        # Predictions shape is still morphing in the development.
        # Use run_from_state to get the shape of predictions. This might
        # cost a few second extra but will be resilience to major code 
        # update in jcm
        save_interval_day = (timestep / jdt.to_timedelta(1, "day")).item() 
        _, _, predictions = model.run_from_state_with_carry(
            initial_state=state,
            save_interval=save_interval_day,  
            total_time=save_interval_day,
            forcing=forcing,
            output_averages=True,
        )
        physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)

        return {
            "state": state,
            "derived": JCMDerived.zeros(D2_nodal_shape, physics_no_time_dimension),
            "forcing": forcing,
        }

    def generate_step_function():
        # Notice: since save_interval and total_time are claimed
        #         static parameters, we cannot pass in traceable
        #         object. So use item() to convert from scalar
        #         jax.Array to float.
        save_interval_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item() 
        total_time_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item()
        def step_function(carry, step):
            state = carry["state"]
            forcing = carry["forcing"]
            new_atm_modal_state, _, predictions = model.run_from_state_with_carry(
                initial_state=state,
                save_interval=save_interval_day,  
                total_time=total_time_day,
                forcing=forcing,
                output_averages=True,
            )
            physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)

            # jcm publishes only the (ix, il) GRID-MEAN surface fluxes: the
            # per-tile (land/sea) breakdown it used to expose is gone, so a
            # component that needs a tile-resolved flux has to compute it in
            # the land/ocean component itself rather than read it here.
            # Sign convention: jcm's `hfluxn` is positive DOWNWARD (into the
            # surface) while every jem surface component takes heat flux
            # positive UPWARD (out of the surface), hence the negation.
            # `evap` is already upward-positive, in g m-2 s-1.
            sf = physics_no_time_dimension["_surface_flux"]
            total_heat_flux = -sf.hfluxn
            evaporation = sf.evap

            total_freshwater_flux = (
                evaporation
                 - physics_no_time_dimension["_convection"].precnv
                 - physics_no_time_dimension["_condensation"].precls
            ) / 1000.0 # The number 1000.0 is the convert factor of mass density flux of freshwater from g/m^2/s to kg/m^2/s

            return (
                {
                    "state": new_atm_modal_state,
                    "derived": JCMDerived(
                        physics_no_time_dimension,
                        total_heat_flux,
                        total_freshwater_flux,
                    ),
                    "forcing": forcing,
                },
                predictions
            )

        return step_function

    def predictions_to_xarray(predictions):
        from jcm.model import ModelPredictions
        # Re-attach coords and physics lost during pytree flatten/unflatten
        full_predictions = ModelPredictions(predictions._predictions, model.coords, model.physics)
        return full_predictions.to_xarray()

    def get_info():
        return {
            "Basic" : str(model),
        }

    safe_setattr(model, "initialize", initialize)
    safe_setattr(model, "predictions_to_xarray", predictions_to_xarray)
    safe_setattr(model, "generate_step_function", generate_step_function)
    safe_setattr(model, "get_info", get_info)

    return model
