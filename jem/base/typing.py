from typing import TypeVar, Any, Callable, List, Dict
import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr

Array = jnp.ndarray
ArrayOrArrayTuple = Array | tuple[Array, ...]
Numeric = float | int | Array

Pytree = Any

CalendarTimeStr = str
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


