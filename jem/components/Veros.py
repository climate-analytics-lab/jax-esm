"""Veros adapter to JEM"""
from typing import Annotated, Any, Dict, Optional, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
import jax_datetime as jdt
import xarray as xr
import tree_math

from jem.utils.bulk_op import mean_leaf, stack_objects
import jem.utils.data_structure as data_structure

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
    total_heat_flux: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    wind_x: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    wind_y: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    surface_air_temperature: Annotated[float, ("longitude", "latitude"), "two_dimensional"]

@tree_math.struct
@dataclass
class VerosState:
    raw_state: Any
    sea_surface_temperature: Any

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
    
    decorator = data_structure.build_dataclass_from_typed_and_dimensioned({"two_dimensional": (nxt, nyt)})
    VerosForcing = decorator(AbstractVerosForcing)
    D2_shape = (nxt, nyt)
    D2_information = (D2_shape, ("longitude", "latitude"))
    check_before_setattr(model, "state_variable_registry", {
        varname : D2_information for varname in [
            "sea_surface_temperature",
        ]
    })

    check_before_setattr(model, "forcing_variable_registry", {
        varname : D2_information for varname in [
            "wind_x",
            "wind_y",
            "surface_air_temperature",
            "total_heat_flux",
        ]
    })

    def initialize():
        initial_state = VerosState(raw_state = model.state, sea_surface_temperature = jnp.zeros(D2_shape) + 288.15)
        initial_forcing = VerosForcing.zeros()
        return initial_state, initial_forcing

    def generate_step_function(jitted: bool = True):


        def step_function(state, forcing, step):
            
            drag_coefficient = 1e-3 # dimensionless
            air_density = 1.22 # kg / m^3
            gc = 2 # hard-coded ghost cell number
            vs = state.raw_state.variables
            
            wind_velocity = jnp.sqrt(forcing.wind_x**2 + forcing.wind_y**2)
            
            with vs.unlock():
                vs.surface_taux = vs.surface_taux.at[gc:-gc, gc:-gc].set( drag_coefficient * air_density * wind_velocity * forcing.wind_x)
                vs.surface_tauy = vs.surface_tauy.at[gc:-gc, gc:-gc].set( drag_coefficient * air_density * wind_velocity * forcing.wind_y)
            for _ in range(steps_per_coupling_timestep):
                model.step(state.raw_state)
            
            state.sea_surface_temperature = vs.temp[gc:-gc, gc:-gc, -1, vs.tau] + 273.15
            state.sea_surface_temperature = jnp.where( state.sea_surface_temperature < 100, 288.15, state.sea_surface_temperature )
            return state, stack_objects([dict(
                sea_surface_temperature = state.sea_surface_temperature,
                surface_air_temperature = forcing.surface_air_temperature,
                wind_x = forcing.wind_x,
                wind_y = forcing.wind_y,
                total_heat_flux = forcing.total_heat_flux,
            )])


        return jax.jit(step_function) if jitted else step_function

    def predictions_to_xarray(predictions):
        return xr.Dataset(
            data_vars=dict(
                sea_surface_temperature = (["time", "longitude", "latitude"], predictions["sea_surface_temperature"]),
                surface_air_temperature = (["time", "longitude", "latitude"], predictions["surface_air_temperature"]),
                wind_x = (["time", "longitude", "latitude"], predictions["wind_x"]),
                wind_y = (["time", "longitude", "latitude"], predictions["wind_y"]),
                total_heat_flux = (["time", "longitude", "latitude"], predictions["total_heat_flux"]),
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

    return model
