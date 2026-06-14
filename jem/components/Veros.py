"""Veros adapter to JEM"""
from typing import Annotated, Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr

from jem.utils.bulk_op import stack_objects
import jem.utils.data_structure as data_structure

from veros import runtime_settings
print("Setting veros.runtime_settings...")
setattr(runtime_settings, "backend", "jax")
setattr(runtime_settings, "force_overwrite", True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')
#setattr(runtime_settings, 'device', 'cpu')
from veros.core.operators import numpy as npx, update, at # noqa: E402

def copy_veros_state(state):
    return jax.tree_util.tree_map(lambda x: x, state)
    
def check_before_setattr(target, attribute_name, value, *, raise_exception=True):
    if hasattr(target, attribute_name):
        message = f"Attribute name `{attribute_name:s}` already exists."
        if raise_exception:
            raise Exception(message)
        else:
            print(f"Warning: {message:s}")
    
    setattr(target, attribute_name, value)

@data_structure.typed_and_dimensioned
class AbstractVerosForcing:
    heat_flux: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    freshwater_flux: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    surface_taux: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    surface_tauy: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    surface_air_temperature: Annotated[float, ("longitude", "latitude"), "two_dimensional"]

def make_jem_compatible(
    model: Any,
    coupling_timestep: jdt.Timedelta,
) -> Any:

    timestep = jdt.to_timedelta(int(model.state.settings.dt_tracer), "second")
    if timestep * jnp.floor(coupling_timestep / timestep) != coupling_timestep:
        raise Exception("Coupling timestep should be a multiple of timestep.")
    steps_per_coupling_timestep = int(coupling_timestep / timestep)
    
    nxt = model.state.dimensions["xt"]
    nyt = model.state.dimensions["yt"]
    ghost_cell = 2 # Veros hard-coded ghost cell number
    
    decorator = data_structure.build_dataclass_from_typed_and_dimensioned({"two_dimensional": (nxt, nyt)})
    VerosForcing = decorator(AbstractVerosForcing)
    horizontal_T_shape = (nxt, nyt)
    horizontal_U_shape = (nxt, nyt)
    horizontal_V_shape = (nxt, nyt)
    
    mask_T = jnp.array(model.state.variables.maskT)
    mask_T = mask_T[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell]

    def set_forcing(state):
        print("The original set_forcing in the VerosSetup object is replaced "
              "by this empty set_forcing function. JEM-veros will set the "
              "forcing in the step_function.")
        
    def initialize():
        initial_state = model.state
        initial_derived = {
            'sea_surface_temperature' : jnp.zeros(horizontal_T_shape) + 273.15,
            'sea_surface_u' : jnp.zeros(horizontal_U_shape),
            'sea_surface_v' : jnp.zeros(horizontal_V_shape),
        }
        initial_forcing = VerosForcing.zeros()
        print(f"initial_forcing.surface_taux.shape = {str(initial_forcing.surface_taux.shape)}")
        return dict(state=initial_state, derived=initial_derived, forcing=initial_forcing)
    
    def generate_step_function():

        def step_function(carry, step):
            
            state = carry["state"]
            forcing = carry["forcing"]
            cp_0 = 3991.86795711963
            salinity_ref = 35.0 # PSU
            vs = state.variables
            settings = state.settings
             
            with vs.unlock():
                
                if not settings.enable_tempsalt_sources:
                    print("Warning: settings.enable_tempsalt_sources = False. This is for the `temp_source` variable, "
                          "which is very similar to forc_temp_surface."
                    )
                vs.surface_taux = update(vs.surface_taux, at[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell], forcing["surface_taux"])
                vs.surface_tauy = update(vs.surface_tauy, at[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell], forcing["surface_tauy"])
                # The following computation is learned from
                # `veros/setups/global_1deg/global_1deg.py`
                if settings.enable_tke:
                    print("Veros: settings.enable_tke is set true")
                    surface_stress_squared = (
                        (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                        + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
                    )
                    # `sqrt` has an infinite derivative at 0, so AD through `sqrt(x)`
                    # blows up to NaN as x -> 0, even though the primal value (0)
                    # stays finite. Floor the *squared* magnitude -- sqrt's argument
                    # -- at min_stress_magnitude**2 so sqrt and its derivative stay
                    # bounded; this naturally caps the resulting magnitude from below
                    # at min_stress_magnitude.
                    min_stress_magnitude = 1e-3
                    surface_stress_magnitude = npx.sqrt(
                        npx.maximum(surface_stress_squared, min_stress_magnitude ** 2)
                    )
                    vs.forc_tke_surface = update(
                        vs.forc_tke_surface,
                        at[1:-1, 1:-1],
                        surface_stress_magnitude ** 1.5,
                    )

                # W/m^2 K kg/J m^3/kg = K m/s
                vs.forc_temp_surface = update(
                    vs.forc_temp_surface,
                    at[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell],
                    - forcing["heat_flux"] * vs.maskT[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1] / cp_0 / settings.rho_0
                )

                # freshwater_flux is upward positive. Therefore, positive freshwater_flux should increase salinity
                vs.forc_salt_surface = update(
                    vs.forc_salt_surface,
                    at[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell],
                     forcing["freshwater_flux"] * vs.maskT[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1] / settings.rho_0 * salinity_ref
                )

            def _sub_step_function(_, state):
                model.step(state)
                return state

            state = jax.lax.fori_loop(0, steps_per_coupling_timestep, _sub_step_function, state)
            # `fori_loop` reconstructs the carry into fresh VerosState/VerosVariables
            # instances (via tree_unflatten), so the `vs` bound above is now stale;
            # rebind it to the evolved state before reading diagnostics from it.
            vs = state.variables

            sea_surface_temperature = vs.temp[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1, vs.tau] + 273.15
            sea_surface_temperature = jnp.where( sea_surface_temperature < 100, 288.15, sea_surface_temperature )
            sea_surface_salinity = vs.salt[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1, vs.tau]
            
            temp = vs.temp[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, :, vs.tau]
            salt = vs.salt[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, :, vs.tau]
            u = vs.u[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, :, vs.tau]
            v = vs.v[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, :, vs.tau]

            sea_surface_u = vs.u[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1, vs.tau]
            sea_surface_v = vs.v[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1, vs.tau]

            return dict(
                state=state,
                derived={
                    "sea_surface_temperature" : sea_surface_temperature,
                    "sea_surface_u" : sea_surface_u,
                    "sea_surface_v" : sea_surface_v,
                },
                forcing=forcing,
            ), stack_objects([dict(
                    sea_surface_temperature = sea_surface_temperature,
                    sea_surface_salinity = sea_surface_salinity,
                    sea_surface_u = sea_surface_u,
                    sea_surface_v = sea_surface_v,
                    temp = temp,
                    salt = salt,
                    u = u,
                    v = v,
                    surface_air_temperature = forcing.surface_air_temperature,
                    surface_taux = forcing.surface_taux,
                    surface_tauy = forcing.surface_tauy,
                    heat_flux = forcing.heat_flux,
                    freshwater_flux = forcing.freshwater_flux,
            )])


        return step_function

    def predictions_to_xarray(predictions):
        return xr.Dataset(
            data_vars=dict(
                temp = (["time", "longitude", "latitude", "depth"], predictions["temp"]),
                salt = (["time", "longitude", "latitude", "depth"], predictions["salt"]),
                u = (["time", "longitude", "latitude", "depth"], predictions["u"]),
                v = (["time", "longitude", "latitude", "depth"], predictions["v"]),
                sea_surface_temperature = (["time", "longitude", "latitude"], predictions["sea_surface_temperature"]),
                sea_surface_u = (["time", "longitude", "latitude"], predictions["sea_surface_u"]),
                sea_surface_v = (["time", "longitude", "latitude"], predictions["sea_surface_v"]),
                sea_surface_salinity = (["time", "longitude", "latitude"], predictions["sea_surface_salinity"]),
                surface_air_temperature = (["time", "longitude", "latitude"], predictions["surface_air_temperature"]),
                surface_taux = (["time", "longitude", "latitude"], predictions["surface_taux"]),
                surface_tauy = (["time", "longitude", "latitude"], predictions["surface_tauy"]),
                heat_flux = (["time", "longitude", "latitude"], predictions["heat_flux"]),
                freshwater_flux = (["time", "longitude", "latitude"], predictions["freshwater_flux"]),
                mask_T = (["longitude", "latitude", "depth"], mask_T),
                mask_surface_T = (["longitude", "latitude"], mask_T[:, :, -1]),
            ),
        )

    def get_info():
        return {
            key: str(value) for key, value in model.state.settings.items()
        }

    check_before_setattr(model, "initialize", initialize)
    check_before_setattr(model, "predictions_to_xarray", predictions_to_xarray)
    check_before_setattr(model, "generate_step_function", generate_step_function)
    check_before_setattr(model, "get_info", get_info)
    check_before_setattr(model, "set_forcing", set_forcing, raise_exception=False)
    
    return model
