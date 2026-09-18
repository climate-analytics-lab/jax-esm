"""JCM adapter to JEM"""

import numpy as np

from jcm.model import Model
from jcm.forcing import default_forcing

import jax
import jax.numpy as jnp
import jax_datetime as jdt


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


# ---------------------------------------------------------------------------
# The atmosphere's clock.
#
# ``primitive_equations.State.sim_time`` is a float32 seconds counter that the
# SIL3 integrator advances by ``dt`` every model step. float32 cannot hold an
# 1800-s step exactly once sim_time exceeds 2**24 s (194 d); from 2**27 s
# (4.25 yr) every step adds 1792 s (year = 366.87 d) and from 2**32 s (136.1 yr)
# every step adds 2048 s (year = 321.0 d). Carrying that counter across coupling
# calls therefore detaches the insolation from the calendar -- see
# pelagos-jcm CHANGELOG.md 2026-09-17.
#
# Invariant of the carry returned by this adapter:
#   * ``carry["state"].sim_time == 0`` at every carry boundary, and
#   * ``carry["date"]`` is the model's calendar time (whole days since 1970-01-01
#     and seconds within the day, both int32, exact) -- i.e. the date at which
#     ``sim_time == 0``.
# Each coupling call re-zeroes sim_time, hands ``carry["date"]`` to
# ``Model.run_from_state(start_date=...)`` as the (traced) date origin, and then
# advances the date by ``coupling_timestep`` in integer arithmetic. ``sim_time``
# never exceeds one coupling interval, where float32 is exact.
#
# The date is a plain dict of int32 leaves (not a ``jdt.Datetime``) so that generic
# tree_maps over the carry (``asfloat64`` above, pmap replication, dtype fix-ups)
# cannot silently turn it into a float. It is deliberately NOT passed through
# ``asfloat64``. Consumers that need the atmosphere's date must read
# ``carry["date"]``; ``carry["state"].sim_time`` is no longer a clock.
# ---------------------------------------------------------------------------
def date_to_carry(date: jdt.Datetime) -> dict:
    """``jdt.Datetime`` -> ``{"days": int32, "seconds": int32}`` carry leaves."""
    return {
        "days": jnp.asarray(date.delta.days, dtype=jnp.int32),
        "seconds": jnp.asarray(date.delta.seconds, dtype=jnp.int32),
    }


def carry_to_date(date_leaves: dict) -> jdt.Datetime:
    """``{"days", "seconds"}`` carry leaves -> ``jdt.Datetime`` (works on tracers)."""
    return jdt.Datetime(
        jdt.Timedelta.from_normalized(date_leaves["days"], date_leaves["seconds"])
    )


def _rezero_sim_time(state):
    """Return ``state`` with ``sim_time`` set to exactly 0 (same dtype/shape)."""
    return state.replace(sim_time=jnp.zeros_like(jnp.asarray(state.sim_time)))

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
        raise Exception("Coupling timestep should be a multiple of timestep.")

    D2_nodal_shape = model.coords.nodal_shape[1:]
    def initialize():

        state=model._prepare_initial_modal_state()
        forcing = default_forcing(model.coords.horizontal)
        
        # Predictions shape is still morphing in the development.
        # Use run_from_state to get the shape of predictions. This might
        # cost a few second extra but will be resilience to major code 
        # update in jcm
        save_interval_day = (timestep / jdt.to_timedelta(1, "day")).item() 
        _, predictions = model.run_from_state(
            initial_state=state,
            save_interval=save_interval_day,  
            total_time=save_interval_day,
            forcing=forcing,
            output_averages=True,
        )
        physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)

        carry = asfloat64(dict(
            state=_rezero_sim_time(state),
            derived={ # Derived
                "physics" : physics_no_time_dimension,
                "total_heat_flux" : jnp.zeros(D2_nodal_shape),
                "total_freshwater_flux" : jnp.zeros(D2_nodal_shape),
            },
            forcing=forcing,
        ))
        # Exact integer date; NOT through asfloat64 (see the clock note above).
        carry["date"] = date_to_carry(model.start_date)
        return carry

    def generate_step_function():
        # Notice: since save_interval and total_time are claimed
        #         static parameters, we cannot pass in traceable
        #         object. So use item() to convert from scalar
        #         jax.Array to float.
        save_interval_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item() 
        total_time_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item()
        def step_function(carry, step):
            forcing = asfloat64(carry["forcing"])
            if "date" not in carry:
                raise KeyError(
                    "JCM carry has no 'date': it was not built by this adapter's "
                    "initialize(), or a caller dropped the key. The carried date is the "
                    "atmosphere's only clock (state.sim_time is re-zeroed every call)."
                )
            date = carry_to_date(carry["date"])
            # Enforce the invariant on the way in (the initial carry may hold any
            # float here), integrate one coupling interval from sim_time = 0 with the
            # carried date as the origin, then re-zero and advance the date exactly.
            state = _rezero_sim_time(carry["state"])
            new_atm_modal_state, predictions = model.run_from_state(
                initial_state=state,
                save_interval=save_interval_day,  
                total_time=total_time_day,
                forcing=forcing,
                output_averages=True,
                start_date=date,
            )
            new_atm_modal_state = _rezero_sim_time(new_atm_modal_state)
            new_date = date + coupling_timestep
            physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)
            total_heat_flux = - jnp.sum(physics_no_time_dimension.surface_flux.hfluxn, axis=2) # convert to upward positive
            evaporation = jnp.sum(physics_no_time_dimension.surface_flux.evap, axis=2) # upward positive

            total_freshwater_flux = (
                evaporation
                 - physics_no_time_dimension.convection.precnv
                 - physics_no_time_dimension.condensation.precls
            ) / 1000.0 # The number 1000.0 is the convert factor of mass density flux of freshwater from g/m^2/s to kg/m^2/s

            new_carry = asfloat64(dict(
                state=new_atm_modal_state,
                derived={
                    "physics" : physics_no_time_dimension,
                    "total_heat_flux" : total_heat_flux,
                    "total_freshwater_flux" : total_freshwater_flux,
                },
                forcing=forcing,
            ))
            new_carry["date"] = date_to_carry(new_date)
            return new_carry, predictions

        return step_function

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
