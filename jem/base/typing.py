from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import xarray as xr
from typeguard import typechecked

Pytree = Any

ComponentName = str
Workflow = Pytree  # Any nested sequence of ComponentName
ComponentCarry = Pytree
CoupledCarry = dict[ComponentName, ComponentCarry]
SimulationTime = float
Predictions = Pytree

InitializeFunction = Callable[[], ComponentCarry]
StepFunction = Callable[[ComponentCarry, SimulationTime], tuple[ComponentCarry, Predictions]]
StepFunctionGenerator = Callable[[], StepFunction]
MapperFunction = Callable[[ CoupledCarry ], CoupledCarry ]
TrajectoryFunction = Callable[[CoupledCarry], tuple[CoupledCarry, Predictions]]
PredictionsToXarrayFunction = Callable[[Predictions], xr.Dataset]
GetInfoFunction = Callable[[], dict]

VariableName = str
VariableShape = tuple[int, ...]
VariableDimension = tuple[str, ...]
VariableMetadata = tuple[VariableShape, VariableDimension | None]
VariableRegistry = dict[VariableName, VariableMetadata]

@typechecked
@dataclass
class JEMComponent:
    # mandatory
    initialize: InitializeFunction
    generate_step_function: StepFunctionGenerator

    # optional 
    predictions_to_xarray: PredictionsToXarrayFunction
    get_info: GetInfoFunction


    # jem-internal
    raw_component: Any
    name: str


