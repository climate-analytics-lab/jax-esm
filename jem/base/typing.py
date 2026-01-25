from typing import TypeVar, Any, Callable, List, Dict
import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr
from dataclasses import dataclass

Array = jnp.ndarray
ArrayOrArrayTuple = Array | tuple[Array, ...]
Numeric = float | int | Array

ComponentName = str

Pytree = Any
Workflow = Pytree
State = Pytree
Forcing = Pytree
SimulationTime = float

History = TypeVar('History')
JittableFlag = bool

InitializeFunction = Callable[[], tuple[State, Forcing]]

StepFunction = Callable[[State, Forcing, SimulationTime], tuple[State, History]]
StepFunctionGenerator = Callable[[JittableFlag], StepFunction]

HistoryToXarray = Callable[History, xr.Dataset]

GetInfoFunction = Callable[[], Dict]


ForcingMapperFunction = Callable[ [ Dict[str, Forcing], Dict[str, State] ], Dict[str, Forcing] ]
RegridderFunction = Callable[[Array], Array]

@dataclass
class JEMComponent:
    # mandatory
    component_state_class : State
    component_forcing_class : Forcing
    state_variable_registry : Any
    forcing_variable_registry : Any
    initialize : InitializeFunction
    generate_step_function : StepFunctionGenerator
    predictions_to_xarray : HistoryToXarray
    get_info : GetInfoFunction

    # jem-internal
    raw_component: Any
    name: str
 

@dataclass
class JEMForcingMapper:
    # mandatory
    involved_component_names: List[str]
    get_info : GetInfoFunction
    map_forcings: ForcingMapperFunction
    
    # jem-internal
    raw_forcing_mapper: Any
    name: str

 

