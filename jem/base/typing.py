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
MapperFunction = Callable[ [ CoupledCarry ], CoupledCarry ]
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
class JEMMapper:
    # jem-internal
    raw_mapper: Any
    name: str

    # mandatory
    involved_component_names: List[str]
    get_info : GetInfoFunction
    __call__: MapperFunction

    # =====================================================
    # JEM developer explaination for defining __call__ here
    # =====================================================
    # Since dataclass will treat the __call__ above as an instance field, the
    # resulting JEMMapper will not be callable. Therefore, we have to define
    # a __call__ here to get around
    def __call__(self, carry):
        # Fetching the __call__ in the instance field and execute it.
        # This is the same as `self.raw_mapper(carry)`
        return self.__call__(carry)
 
JEMMapperType = JEMMapper | Any
JEMComponentType = JEMComponent | Any
