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
ComponentCarry = Pytree
CoupledCarry = Dict[ComponentName, ComponentCarry]
SimulationTime = float

History = TypeVar('History')

InitializeFunction = Callable[[], ComponentCarry]
StepFunction = Callable[[ComponentCarry, SimulationTime], tuple[ComponentCarry, History]]
StepFunctionGenerator = Callable[[], StepFunction]
ForcingMapperFunction = Callable[ [ CoupledCarry ], CoupledCarry ]
TrajectoryFunction = Callable[[CoupledCarry], tuple[CoupledCarry, History]]

HistoryToXarray = Callable[[History], xr.Dataset]

GetInfoFunction = Callable[[], Dict]





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
