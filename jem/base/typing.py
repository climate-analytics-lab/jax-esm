from typing import TypeVar, Any, Callable, List, Dict, Optional
import jax.numpy as jnp
import xarray as xr
from dataclasses import dataclass
from typeguard import typechecked

Array = jnp.ndarray
ArrayOrArrayTuple = Array | tuple[Array, ...] # type: ignore
Numeric = float | int | Array

ComponentName = str

Pytree = Any
Workflow = Pytree
State = Pytree
Derived = Pytree
Forcing = Pytree
SimulationTime = float

History = TypeVar('History')

InitializeFunction = Callable[[], tuple[State, Derived, Forcing]]

StepFunction = Callable[[State, Forcing, SimulationTime], tuple[State, History]]
StepFunctionGenerator = Callable[[], StepFunction]
TrajectoryFunction = Callable[[Pytree], tuple[Pytree, History]]

HistoryToXarray = Callable[[History], xr.Dataset]

GetInfoFunction = Callable[[], Dict]

ForcingMapperFunction = Callable[ [ Dict[str, State], Dict[str, Derived], Dict[str, Forcing] ], Dict[str, Forcing] ]


VariableName = str
VariableShape = tuple[int, ...]
VariableDimension = tuple[str, ...]
VariableMetadata = tuple[ VariableShape, Optional[VariableDimension]]
VariableRegistry = Dict[VariableName, VariableMetadata]

@typechecked
@dataclass
class JEMComponent:
    # mandatory
    initialize : InitializeFunction
    generate_step_function : StepFunctionGenerator
    predictions_to_xarray : HistoryToXarray
    get_info : GetInfoFunction

    # jem-internal
    raw_component: Any
    name: str
 
@typechecked
@dataclass
class JEMForcingMapper:
    # mandatory
    involved_component_names: List[str]
    get_info : GetInfoFunction
    map_forcings: ForcingMapperFunction
    
    # jem-internal
    raw_forcing_mapper: Any
    name: str

 
JEMForcingMapperType = JEMForcingMapper | Any
JEMComponentType = JEMComponent | Any
