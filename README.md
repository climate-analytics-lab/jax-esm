# JAX-ESM: A JAX-based Earth System Model Coupler

JAX-ESM is a JAX-based coupling framework for Earth system components, specifically designed for coupling JCM (JAX Climate Model) with ocean and flux models. It provides efficient time integration using `jax.lax.scan` and supports component-specific sub-stepping for numerical stability.

## Features

- **JAX-Native**: Fully JIT-compilable, GPU-ready, and differentiable
- **Dynamic State Creation**: Factory functions for creating custom component states with arithmetic operations
- **Efficient Time Integration**: Uses `jax.lax.scan` for vectorized time stepping with optional debug mode
- **Component Sub-stepping**: Each component can use internal sub-steps for numerical stability
- **Direct Component Coupling**: Components can directly access each other's state for tight integration
- **xarray Integration**: Built-in conversion to xarray Datasets for analysis

## Installation 

```
# Install published jem
pip install jem

# Using locally cloned jem
git clone -b [version_tag] https://github.com/climate-analytics-lab/jax-esm
cd jax-esm
pip install -e "."
```

## Quick Start

Here is an example to run an aquaplanet simulation.

```
import jcm
import jax_datetime as jdt

from jem import Coupler
from jem.components import JCM, SlabOceanModel
from jem.mapping import BasicMapper

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
one_second = jdt.to_timedelta(1, "second")

mapper = BasicMapper()
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("ocn", "forcing.total_heat_flux"),
)
mapper.add_mapping(
    source = ("ocn", "state.sea_surface_temperature"),
    target = ("atm", "forcing.sea_surface_temperature"),
)

atm_model = jcm.model.Model(
    start_date=start_datetime,
)

atm_model = JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
    land_model_active=False,
)

model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=SlabOceanModel(
            start_datetime=start_datetime,
            timestep=coupling_timestep / one_second,
        ),
    ),
    mappers=dict(mapper=mapper),
)

simulation_interval = jdt.to_timedelta(30, "day")
initial_state, final_state, predictions = model.run(
    workflow=["mapper", "atm", "ocn"],
    iterations = int(simulation_interval / coupling_timestep),
)

output_dict = model.predictions_to_xarray(predictions)
print(output_dict["atm"]) # xarray.Dataset
print(output_dict["ocn"])
```

## Architecture

### Component Interface

Each component must inherit from `CoupledComponent` and implement:

- **`__init__(config)`**: Initialize with ComponentConfig, need to define `component_state_class` and `component_forcing_class`.
- **`initialize()`**: Return initial component state and forcing
- **`generate_step_function()`**: Return a JIT-compiled step function
  - Signature: `step_function(state, forcing, time) -> (new_component_state, predictions)`

`component_class_class` and `component_forcing_class` are JAX pytrees with arithmetic operations via `tree_math.struct`.

### Component Coupling

The current implementation uses direct coupling:
- Coupler creates a `CoupledState` with fields for each component
- Coupler passes the state of each component and forcing to the corresponding component's `step_function`.

### Time Integration

- Uses `jax.lax.scan` for efficient time stepping
- Components advance in parallel each coupling timestep
- Each component can use internal sub-stepping (`substeps` parameter)
- Debug mode available with `jax_scan=False` (uses Python loop)

## Examples
- `jupytext_notebooks/jem_JCM_SlabOceanModel_SlabLandModel.py`: JAX-GCM coupled with slab ocean and slab land models with Earth landscape.

## Integration with JAX-GCM (JCM)
JAX-ESM is specifically designed for coupling JCM (JAX Climate Model) with ocean models:

### Included Components

1. **JCM (Atmosphere)**
   - Location: `jem/components/JCM/`
   - Wraps JCM atmosphere model from jax-gcm
   - Handles conversion between Dinosaur dynamics states and physics states
   - Supports internal sub-stepping

2. **SlabOceanModel**
   - Location: `jem/components/SlabOceanModel/`
   - Mixed-layer ocean with climatological relaxation
   - Anomaly-based SST evolution using Euler backward scheme

3. **SlabLandModel**
   - Location: `jem/components/SlabLandModel/`
   - One layer land with climatological relaxation
   - Anomaly-based land surface temperature evolution using Euler backward scheme

## Current Limitations
- **JCM Dependency**: Component tests require JCM (jax-gcm) to be installed

See `CODE_REVIEW.md` for detailed analysis and improvement recommendations.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality (see `tests/` for examples)
4. Ensure tests pass: `pytest`
5. Follow existing code style
6. Submit a pull request

## Development Status

- **Version**: 0.1.0 (Alpha)
- **Status**: Prototype coupling framework
- **Production Ready**: Core functionality stable
- **API Stability**: Subject to change

