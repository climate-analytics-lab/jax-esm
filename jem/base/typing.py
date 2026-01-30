from typing import TypeVar, Any, Callable, List, Dict, Optional
import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr
from dataclasses import dataclass
from typeguard import typechecked

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
TrajectoryFunction = Callable[[Pytree], tuple[Pytree, History]]

HistoryToXarray = Callable[[History], xr.Dataset]

GetInfoFunction = Callable[[], Dict]


ForcingMapperFunction = Callable[ [ Dict[str, Forcing], Dict[str, State] ], Dict[str, Forcing] ]
RegridderFunction = Callable[[Array], Array]

VariableName = str
VariableShape = tuple[int, ...]
VariableDimension = tuple[str, ...]
VariableMetadata = tuple[ VariableShape, Optional[VariableDimension]]
VariableRegistry = Dict[VariableName, VariableMetadata]

@typechecked
@dataclass
class JEMComponent:
    # mandatory
    component_state_class : State
    component_forcing_class : Forcing
    state_variable_registry : VariableRegistry
    forcing_variable_registry : VariableRegistry
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
