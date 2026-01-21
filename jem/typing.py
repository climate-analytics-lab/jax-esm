from typing import Any


PyTree = Any

Array = np.ndarray | jnp.ndarray
ArrayOrArrayTuple = Array | tuple[Array, ...]
Numeric = float | int | Array
Unit = scales.Unit
Quantity = scales.Quantity
PRNGKeyArray = Any

# TODO(dkochkov): Rename PyTreeXXX to XXX once old types are pruned.
PyTreeState = TypeVar('PyTreeState')
Pytree = Any
PyTreeMemory = Pytree
PyTreeDiagnostics = Pytree

AuxFeatures = dict[str, Any]
DataState = dict[str, Any]
ForcingData = dict[str, Any]

State = TypeVar('State')
StateFn = Callable[[State], State]
InverseFn = Callable[[State, jnp.ndarray], State]
StepFn = Callable[[State, State], State]
FilterFn = Callable[[State, State, State], tuple[State, State]]

ScanFn = Callable[..., Any]
PytreeFn = Callable[[Pytree], Pytree]
PyTreeTermsFn = Callable[[PyTreeState], PyTreeState]
PyTreeInverseFn = Callable[[PyTreeState, Numeric], PyTreeState]
TimeStepFn = Callable[[PyTreeState], PyTreeState]
PyTreeFilterFn = Callable[[PyTreeState], PyTreeState]
PyTreeStepFilterFn = Callable[[PyTreeState, PyTreeState], PyTreeState]
PyTreeStepFilterModule = Callable[..., PyTreeStepFilterFn]

Forcing = Pytree
ForcingFn = Callable[[ForcingData, float], Forcing]
ForcingModule = Callable[..., ForcingFn]

PostProcessFn = Callable[..., Any]
Params = Mapping[str, Mapping[str, Array]] | None

StepFn = Callable[[PyTreeState, Forcing | None], PyTreeState]
StepModule = Callable[..., StepFn]
CorrectorFn = Callable[
    [PyTreeState, PyTreeState | None, Forcing | None], PyTreeState
]
CorrectorModule = Callable[..., CorrectorFn]
ParameterizationFn = Callable[
    [
        PyTreeState,
        PyTreeMemory | None,
        PyTreeDiagnostics | None,
        RandomnessState | None,
        Forcing | None,
    ],
    PyTreeState,
]
ParameterizationModule = Callable[..., ParameterizationFn]
TrajectoryFn = Callable[..., tuple[Any, Any]]
TransformFn = Callable[[Pytree], Pytree]
TransformModule = Callable[..., TransformFn]

GatingFactory = Callable[..., Callable[[Array, Array], Array]]
TowerFactory = Callable[..., Callable[..., Any]]
LayerFactory = Callable[..., Callable[..., Any]]

EmbeddingFn = Callable[
    [
        Pytree,
        PyTreeMemory | None,
        PyTreeDiagnostics | None,
        RandomnessState | None,
        Forcing | None,
    ],
    Pytree,
]
EmbeddingModule = Callable[..., EmbeddingFn]
