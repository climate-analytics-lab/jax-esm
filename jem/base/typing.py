from typing import Any, Callable, List, Dict, Optional
import xarray as xr
from dataclasses import dataclass
from typeguard import typechecked

Pytree = Any

ComponentName = str
Workflow = Pytree  # Any nested sequence of ComponentName
ComponentCarry = Pytree
CoupledCarry = Dict[ComponentName, ComponentCarry]
SimulationTime = float
Predictions = Pytree

InitializeFunction = Callable[[], ComponentCarry]
StepFunction = Callable[[ComponentCarry, SimulationTime], tuple[ComponentCarry, Predictions]]
StepFunctionGenerator = Callable[[], StepFunction]
MapperFunction = Callable[[ CoupledCarry ], CoupledCarry ]
TrajectoryFunction = Callable[[CoupledCarry], tuple[CoupledCarry, Predictions]]
PredictionsToXarrayFunction = Callable[[Predictions], xr.Dataset]
GetInfoFunction = Callable[[], Dict]

VariableName = str
VariableShape = tuple[int, ...]
VariableDimension = tuple[str, ...]
VariableMetadata = tuple[VariableShape, Optional[VariableDimension]]
VariableRegistry = Dict[VariableName, VariableMetadata]

@typechecked
@dataclass
class JEMComponent:
    # mandatory
    initialize: InitializeFunction
    generate_step_function: StepFunctionGenerator

    # jem-internal
    raw_component: Any
    name: str
 
    predictions_to_xarray: Optional[PredictionsToXarrayFunction] = None
    get_info: Optional[GetInfoFunction] = None



