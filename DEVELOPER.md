# DEVELOPER.md — JAX-ESM (JEM)

A technical reference for developers and AI agents working on this codebase.

---

## Table of Contents

1. [Project Purpose](#project-purpose)
2. [Package Layout](#package-layout)
3. [Core Concepts](#core-concepts)
4. [The JEM Interface Contract](#the-jem-interface-contract)
5. [Coupler Internals](#coupler-internals)
6. [Built-in Components](#built-in-components)
7. [Mapping Layer](#mapping-layer)
8. [Data Structure System](#data-structure-system)
9. [Type Aliases (`jem.base.typing`)](#type-aliases-jembasetyping)
10. [Utilities](#utilities)
11. [Testing](#testing)
12. [Adding a New Component](#adding-a-new-component)
13. [Conventions & Constraints](#conventions--constraints)

---

## Project Purpose

JAX-ESM (imported as `jem`) is a **differentiable coupler** for Earth System Models. It connects
independent climate components — atmosphere, ocean, land — into a single JAX-compatible simulation
loop that can be JIT-compiled, run on GPUs, and differentiated end-to-end.

The central design decision: components are kept black-box. JEM only requires a small duck-typed
interface from each component; it does not impose inheritance.

---

## Package Layout

```
jax-esm/
├── jem/                          # installable package (import as `jem`)
│   ├── __init__.py               # exports Coupler, typing
│   ├── constants.py              # physical constants
│   ├── base/
│   │   ├── coupler.py            # Coupler class — the engine
│   │   ├── interface.py          # duck-type resolver (resolve_interface)
│   │   ├── typing.py             # all type aliases and JEMComponent dataclass
│   │   └── exceptions.py        # ValidationError and friends
│   ├── components/
│   │   ├── __init__.py           # exports JCM, SlabOceanModel, SlabLandModel, SlabAtmosphereModel,
│   │   │                         # SlabSeaiceModel; lazily exports Veros (optional dependency)
│   │   ├── jcm_component.py      # adapter for jcm (JAX Climate Model atmosphere)
│   │   ├── veros_component.py    # adapter for Veros ocean model
│   │   └── slab/
│   │       ├── base.py           # SlabModelBase (shared grid/mask/timestep logic)
│   │       ├── slab_ocean_model/ # SlabOceanModel
│   │       ├── slab_land_model/  # SlabLandModel
│   │       ├── slab_atmosphere_model/ # SlabAtmosphereModel
│   │       └── slab_seaice_model/     # SlabSeaiceModel
│   ├── mapping/
│   │   ├── __init__.py           # exports BasicMapper, IdentityRegridder, Grid, GridSpecification
│   │   ├── mapper.py             # BasicMapper — variable exchange between components
│   │   ├── regridder.py          # BasicRegridder, IdentityRegridder, BilinearRegridder
│   │   ├── grid.py               # Grid and GridSpecification dataclasses
│   │   └── builtin_grid_generator.py
│   └── utils/
│       ├── bulk_op.py            # stack_objects, concat_objects, mean_leaf, unwrap_leading_dims
│       ├── cycles.py             # periodic/cyclic index helpers
│       ├── data_structure.py     # @typed_and_dimensioned decorator + dataclass builder
│       ├── datetime_tools.py
│       ├── domain_grid_tools.py
│       ├── idealized_distribution.py
│       └── tree_tools.py
├── tests/
│   ├── unit/
│   │   ├── test_coupler.py       # Coupler unit tests (full coverage)
│   │   └── test_interface.py     # resolve_interface tests
│   └── notebooks/
│       └── test_notebooks.py     # notebook smoke tests
├── notebooks/
│   ├── 01_basic/                 # aquaplanet examples (JCM + SlabOceanModel)
│   ├── 02_experimental/          # earth-like topography, JCM–Veros coupling
│   └── 03_non_geoscience/        # SpringSystem demo (non-climate use case)
├── docs/                         # Sphinx documentation
└── pyproject.toml
```

---

## Core Concepts

### Carry

Every component owns a **carry**: a pytree (usually a plain `dict`) that holds all mutable
state between timesteps. A typical carry looks like:

```python
{
    "state":   <component_state_pytree>,
    "forcing": <component_forcing_pytree>,
    "derived": <optional_derived_quantities>,
}
```

The `Coupler` holds a `CoupledCarry`, which is just `Dict[component_name, carry]`.

### Workflow

A `Workflow` is an ordered list (or nested JAX pytree of strings) that drives one coupling
timestep. Each element is either a **component name** (runs that component's step function)
or a **mapper name** (runs the mapper, which rewrites the coupled carry). Example:

```python
workflow = ["interaction_atm_ocn", "atm", "ocn"]
```

Execution is sequential within a step, but the outer loop (`jax.lax.scan`) is vectorized.

### Scan Loop

`Coupler.run()` calls `jax.lax.scan` over `jnp.arange(iterations)`. Each scan step calls
the compiled `step_function(carry, step_index)`. For debugging, pass `jitted=False` to use
a Python `for` loop (`adhoc_scan` in `coupler.py`) instead.

---

## The JEM Interface Contract

`Coupler.add_component()` calls `resolve_interface()` (in `jem/base/interface.py`) to
extract the four callable/member bindings from any raw object into a `JEMComponent` wrapper.

**Required methods on a raw component:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `initialize()` | `() -> ComponentCarry` | Return the initial pytree carry for this component |
| `generate_step_function()` | `() -> StepFunction` | Return a step function (may be JIT-compiled externally) |

`StepFunction` signature: `(ComponentCarry, SimulationTime) -> (ComponentCarry, Predictions)`

**Optional methods** (set to `None` if absent):

| Method | Signature | Purpose |
|--------|-----------|---------|
| `predictions_to_xarray(predictions)` | `(Predictions) -> xr.Dataset` | Convert trajectory output to Dataset |
| `get_info()` | `() -> Dict` | Return metadata about the component |

**Custom method name mapping:** A component class can define a `__JEM_CUSTOMIZED_MAPPING__`
dict to remap its method names to the expected JEM names. This allows adapting third-party
objects without subclassing.

---

## Coupler Internals

### `Coupler` (`jem/base/coupler.py`)

Key attributes:
- `components: Dict[str, JEMComponent]` — registered components (stored as wrapped `JEMComponent`)
- `mappers: Dict[str, MapperFunction]` — registered mappers (callables)
- `trajectory_holder` — caches the last compiled trajectory function for reuse

Key methods:

| Method | What it does |
|--------|--------------|
| `initialize()` | Calls `component.initialize()` for all components; returns `CoupledCarry` |
| `generate_step_function(workflow, jitted)` | Builds and (optionally) JIT-compiles a single-timestep function |
| `generate_trajectory_function(workflow, iterations, jitted)` | Wraps step function in `jax.lax.scan` (or `adhoc_scan`) |
| `run(workflow, iterations, ...)` | High-level: initialize → trajectory → return `(initial_carry, final_carry, predictions)` |
| `predictions_to_xarray(predictions)` | Delegates to each component's `predictions_to_xarray` |
| `get_info()` | Aggregates info from all components and mappers |

### `adhoc_scan` (`coupler.py`)

A Python-loop drop-in replacement for `jax.lax.scan`, used when `jitted=False`.
Outputs are stacked via `stack_objects` so the return structure matches `lax.scan`.

### Name uniqueness invariant

Component names and mapper names share the same namespace. `_verify_name_uniqueness()`
is called before every `generate_step_function` call to catch conflicts early.

---

## Built-in Components

### JCM (`jem/components/jcm_component.py`)

Wraps a `jcm.model.Model` (spectral atmosphere from `jax-gcm`). Does not subclass;
instead, `make_jem_compatible(model, coupling_timestep)` monkey-patches the four JEM
methods directly onto the `jcm.Model` instance.

Carry structure:
```python
{
    "state":   <jcm modal state>,
    "forcing": <jcm forcing dataclass>,
    "derived": {
        "physics":                <physics output from last step>,
        "total_heat_flux":        Array[lat, lon],    # upward positive, W/m²
        "total_freshwater_flux":  Array[lat, lon],    # upward positive, kg/m²/s
    }
}
```

The JCM adapter calls `model.run_from_state()` each step, passing `coupling_timestep` as
both `save_interval` and `total_time`. It uses `asfloat64()` on all arrays to work around
a JAX dtype inconsistency bug in JCM where some arrays are initialized as `int32`.

### SlabOceanModel (`jem/components/slab/slab_ocean_model/`)

Simple mixed-layer ocean. Integrates SST using an **Euler backward scheme**:

```
SST_new = time_factor * (SST_anom + cd_factor * (-F_net))
```

where `time_factor = 1 / (1 + dt/tau)` accounts for climatological relaxation.

Three `forcing_method` options: `"None"`, `"Qflux"`, `"relaxation"`.

After the Euler backward update, `sea_surface_temperature` is clamped at the seawater freezing
point (following CESM's slab-ocean/CICE convention), and the heat that clamp removes (or, when
the mixed layer sits above freezing, the surplus available to melt ice) is reported as a single
signed diagnostic, `ice_frazil_melt_energy` — CESM's `frzmlt`. It has no ice physics of its own;
the diagnostic is meant to be consumed by a sea-ice component (`SlabSeaiceModel`) via the coupler.

Carry structure:
```python
{
    "state":   OceanState(sim_time, sea_surface_temperature, mixed_layer_depth),
    "forcing": OceanForcing(total_heat_flux, q_flux),
    "derived": {
        "ice_frazil_melt_energy": Array[lat, lon],  # J/m², signed; + forms ice, - can melt ice
    },
}
```

### SlabLandModel (`jem/components/slab/slab_land_model/`)

One-layer land surface temperature with climatological relaxation.
Same Euler backward scheme as `SlabOceanModel`.

### SlabAtmosphereModel (`jem/components/slab/slab_atmosphere_model/`)

Idealized slab atmosphere. Used for testing and non-geoscience examples.

### SlabSeaiceModel (`jem/components/slab/slab_seaice_model/`)

Basal-only sea-ice thickness model: grows and melts purely from `SlabOceanModel`'s
`ice_frazil_melt_energy` forcing (no dynamics, no lateral processes). Exposes a smooth
thickness-to-fraction closure so an atmosphere model can use ice fraction as a boundary
condition.

Carry structure:
```python
{
    "state":   SeaiceState(sim_time, ice_thickness, ice_surface_temperature),
    "forcing": SeaiceForcing(ice_frazil_melt_energy),  # from SlabOceanModel, via a mapper
    "derived": {
        "ice_fraction": Array[lat, lon],  # smooth closure of thickness -> fraction
    },
}
```

### Veros (`jem/components/veros_component.py`)

Adapter for Veros (a full 3D ocean GCM). Requires the `veros-jittable` fork.
Uses monkey-patching (same pattern as JCM). **Not JIT-compilable** — the inner loop
calls `model.step(state)` in Python.

---

## Mapping Layer

### `BasicMapper` (`jem/mapping/mapper.py`)

Implements `MapperFunction` (callable with signature `(CoupledCarry) -> CoupledCarry`).

Stores two dictionaries:
- `mappings`: `{(src_component, tgt_component): {src_var_name: tgt_var_name}}`
- `regridders`: `{(src_comp, tgt_comp, src_var, tgt_var): callable}`

Variable names support **dotted paths** (e.g. `"derived.total_heat_flux"`) that are
resolved by `strget`/`strset` helper functions, which traverse dicts, sequences, and
object attributes uniformly.

To register a mapping:

```python
mapper.add_mapping(
    source=("atm", "derived.total_heat_flux"),
    target=("ocn", "forcing.total_heat_flux"),
    regridder=None,  # defaults to identity
)
```

### Regridders (`jem/mapping/regridder.py`)

- `BasicRegridder` — abstract base; validates input/output shapes
- `IdentityRegridder` — pass-through (no interpolation)
- `BilinearRegridder` — bilinear interpolation via `scipy` (placeholder; not fully JAX-compatible)

### Grid (`jem/mapping/grid.py`)

`Grid` wraps a `coordax.Coordinate` with optional weights, binary mask (`bmask`: 1=land, 0=ocean),
and fractional mask (`fmask`).

`GridSpecification` holds a `grid_universe` (e.g. `"JCM"`) and `grid_family` (e.g. `"T31"`)
parsed from strings like `"JCM::T31"`.

---

## Data Structure System

`jem/utils/data_structure.py` provides two decorators for defining JAX-compatible,
type-annotated state/forcing classes:

### `@typed_and_dimensioned`

Applied to a class with `Annotated[type, dim_names, shape_name]` fields.
Records field metadata in `cls._fields` and registers the class as a JAX pytree node
(via `register_pytree_node_class`). The class becomes a `dataclass`.

```python
@typed_and_dimensioned
class OceanState:
    sim_time: Annotated[float, (), "zero_dimensional"]
    sea_surface_temperature: Annotated[float, ("lon", "lat"), "two_dimensional"]
```

### `build_dataclass_from_typed_and_dimensioned(shape_dict)`

Finalizes a schema class by binding concrete shapes to shape names.
Returns a new class with `zeros()`, `ones()`, `copy(replace_dict)`, and
`typed_and_dimensioned_info()` class methods.

```python
decorator = build_dataclass_from_typed_and_dimensioned({"two_dimensional": (96, 48)})
OceanState = decorator(OceanState)
state = OceanState.zeros()
state = state.copy({"sea_surface_temperature": my_sst_array})
```

This pattern is used by all slab components and the Veros adapter.

---

## Type Aliases (`jem/base/typing`)

```python
Pytree            = Any
ComponentName     = str
ComponentCarry    = Pytree
CoupledCarry      = Dict[ComponentName, ComponentCarry]
SimulationTime    = float

InitializeFunction      = Callable[[], ComponentCarry]
StepFunction            = Callable[[ComponentCarry, SimulationTime], tuple[ComponentCarry, Predictions]]
StepFunctionGenerator   = Callable[[], StepFunction]
MapperFunction          = Callable[[CoupledCarry], CoupledCarry]
TrajectoryFunction      = Callable[[CoupledCarry], tuple[CoupledCarry, Predictions]]
PredictionsToXarrayFunction = Callable[[Predictions], xr.Dataset]
GetInfoFunction         = Callable[[], Dict]
```

`JEMComponent` is a `@typechecked @dataclass` that holds the resolved callables
(`initialize`, `generate_step_function`, `predictions_to_xarray`, `get_info`),
plus `raw_component` (the original object) and `name`.

---

## Utilities

### `jem/utils/bulk_op.py`

| Function | Description |
|----------|-------------|
| `stack_objects(objs)` | `jax.tree_util.tree_map(jnp.stack, *objs)` — transposes a list of pytrees into a pytree of stacked arrays |
| `concat_objects(objs, axis)` | Same but with `jnp.concatenate` |
| `unwrap_leading_dims(obj, first_n_dim)` | Reshapes arrays to merge leading `n` dims into one. Used after `lax.scan` to flatten `[outer, inner, ...]` → `[outer*inner, ...]` |
| `mean_leaf(tree, axis)` | `tree_map(jnp.mean(..., axis=axis), tree)` |

### `jem/utils/data_structure.py`

See [Data Structure System](#data-structure-system) above.

### `jem/utils/cycles.py`

Helpers for computing periodic (cyclic) indices into climatology arrays,
used by `SlabOceanModel` and `SlabLandModel` for monthly Q-flux/SST-clim lookup.

---

## Testing

Run all tests:
```bash
pytest
```

Run only unit tests:
```bash
pytest tests/unit/
```

Run notebook smoke tests:
```bash
pytest tests/notebooks/
```

### Unit test conventions (`tests/unit/test_coupler.py`)

- Tests import `jem` directly (no stubbing in current version; earlier stubs are commented out).
- Component test doubles are created with `_make_raw_component(name)` — a class with
  the four required JEM methods hardcoded.
- `_build_coupler(comp_names, mappers)` is the standard factory for building a `Coupler` in tests.
- Tests cover: `adhoc_scan`, `generate_scan_function`, construction, add/remove, `initialize`,
  name-uniqueness, workflow validation, `generate_step_function`, `generate_trajectory_function`,
  `predictions_to_xarray`, `get_info`, and a full smoke-test cycle.

---

## Adding a New Component

1. **Create the class** in `jem/components/` (or adapt an external model).

2. **Implement the required interface** (duck-typed; no base class required):
   ```python
   class MyComponent:
       def initialize(self) -> dict:
           return {"state": ..., "forcing": ...}

       def generate_step_function(self):
           def step(carry, step_index):
               # advance carry by one timestep
               return new_carry, predictions
           return step
   ```

3. **Optionally add** `predictions_to_xarray(predictions) -> xr.Dataset` and
   `get_info() -> dict`.

4. **Export** from `jem/components/__init__.py`.

5. **Register with the Coupler**:
   ```python
   coupler = Coupler(components={"mycomp": MyComponent(...)})
   ```

6. **Write tests** following the pattern in `tests/unit/test_coupler.py`.

---

## Conventions & Constraints

- **Python ≥ 3.11** required (uses `X | Y` union syntax in annotations).
- **PEP 8** style; formatter is `ruff` (config in `pyproject.toml`).
- **No commits without permission** — do not run `git commit` unless explicitly asked.
- **Required packages**: `jax`, `xarray`, `argparse`, `netCDF4`.
- **CLI scripts** must use `argparse` and be invoked via customizable bash scripts.
- **Sign conventions**: heat flux positive upward; freshwater flux positive upward (evaporation);
  `ice_frazil_melt_energy` positive means the ocean mixed layer went sub-freezing (forms new ice),
  negative means surplus heat is available to melt existing ice (CESM's `frzmlt` convention).
- **Land mask**: `bmask == 1` means land; `bmask == 0` means ocean.
- **Grid spec strings**: `"<universe>::<family>"` format, e.g. `"JCM::T31"`, `"Veros::1deg"`.
- **No JAX tracing through Python conditionals** inside step functions — use `jnp.where`.
- **`jitted=False`** mode uses `adhoc_scan` (Python loop) for debugging; it is not
  equivalent to `jitted=True` in performance or tracing behavior.
- **Predictions shape**: after `lax.scan`, `predictions` arrays have leading shape
  `(iterations,)`. After `unwrap_leading_dims`, they are `(iterations * substeps,)`.
