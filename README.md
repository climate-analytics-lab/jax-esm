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

Please install with `jax-gcm` (hereafter `jcm`) functionality.
```bash

pip install -e ".[jcm]"

# Need to clone jax-gcm manually for now
git clone https://github.com/climate-analytics-lab/jax-gcm.git
export PYTHONPATH=`pwd`/jax-gcm:$PYTHONPATH
```

For development, please run additionally
```bash
pip install -e ".[dev]"
```

## Quick Start

You can run a Jax-gcm coupled run with
```
python tests/integration/test_jesm_JCM_SlabOceanModel_SlabLandModel.py
```
The results will be placed in the directory `output`. 
An example of jax-gcm coupled to slab ocean and land models is as follows:

```python

# File tests/integration/test_jesm_JCM_SlabOceanModel_SlabLandModel.py

from jax_esm.tool_scripts.generate_jcm_forcing_and_topography_files import (
    generate_jcm_forcing_and_topography_files,
)
from jax_esm.components import JCM, SlabLandModel, SlabOceanModel
from jax_esm.coupling.factory.simple_coupling import couple_atm_ocn_lnd as couple
import jcm
import jax_datetime as jdt
from pathlib import Path

resolution = 31
grid_specification = f"JCM::T{resolution:d}"

coupling_timestep = 86400.0
start_datetime = jdt.to_datetime("2000-01-01")
simulation_interval = jdt.to_timedelta(10, "day")
output_dir = Path("output/JCM_SOM_SLM").resolve()

external_files = generate_jcm_forcing_and_topography_files(resolution=resolution)
print("Output dir: ", str(output_dir))
output_dir.mkdir(exist_ok=True, parents=True)

# Creating components
components = dict(
    atm=JCM(
        model=jcm.model.Model(start_date=start_datetime),
        coupling_timestep=coupling_timestep,
    ),
    ocn=SlabOceanModel(
        grid_specification=grid_specification,
        timestep=coupling_timestep,
        start_datetime=start_datetime,
        save_interval=coupling_timestep,
        relaxation_time=60 * 86400.0,
        mask_file=external_files["terrain"],
        SST_clim_file=external_files["forcing"],
    ),
    lnd=SlabLandModel(
        grid_specification=grid_specification,
        timestep=3600 * 6,
        start_datetime=start_datetime,
        save_interval=coupling_timestep,
        relaxation_time=60 * 86400.0,
        topography_file=external_files["terrain"],
        mask_file=external_files["terrain"],
        land_clim_file=external_files["forcing"],
    ),
)

# Creating model
model = couple(**components)

# Obtain initial condition
initial_coupled_state = model.initialize()

# Run coupled model
print("Running model...")
state_holder, predictions = model.run(
    initial_coupled_state=initial_coupled_state,
    start_time=0,
    end_time=simulation_interval / jdt.to_timedelta(1, "second"),
    jitted=True,
    show_progress=True,
    tqdm_kwargs=dict(desc="Simulation"),
)
# Convert output into xarray
output_dict = model.predictions_to_xarray(predictions)

for component_name, ds in output_dict.items():
    output_file = output_dir / f"{component_name:s}.nc"
    print("Output file: ", str(output_file))
    ds.to_netcdf(output_file, engine="netcdf4")

```

## Architecture

### Component Interface

Each component must inherit from `Component` and implement:

- **`__init__(config)`**: Initialize with ComponentConfig, need to define `component_state_class` and `component_forcing_class`.
- **`initialize()`**: Return initial component state
- **`generate_step_function()`**: Return a JIT-compiled step function
  - Signature: `step_function(state, forcing, time) -> (new_component_state, predictions)`

`component_class_class` and `component_forcing_class` are JAX pytrees with arithmetic operations via `tree_math.struct`.

### Component Coupling

The current implementation uses direct coupling:
- Coupler creates a `CoupledState` with fields for each component
- Coupler pass the state of each component and forcing to the corresponding component's `step_function`.

### Time Integration

- Uses `jax.lax.scan` for efficient time stepping
- Components advance in parallel each coupling timestep
- Each component can use internal sub-stepping (`substeps` parameter)
- Debug mode available with `jax_scan=False` (uses Python loop)

## Examples

See the `tests/integration/` directories:
- `tests/integration/test_jesm_JCM_SlabOceanModel.py`: JAX-GCM coupled with slab ocean on an aquaplanet.
- `tests/integration/test_jesm_JCM_SlabOceanModel_SlabLandModel.py`: JAX-GCM coupled with slab ocean and slab land models with Earth landscape.
- `tests/integration/test_jesm_SlabAtmosphereModel_SlabOceanModel.py`: Coupled slab atmosphere-ocean model on an aquaplanet.
- `tests/integration/test_jesm_SlabAtmosphereModel_SlabOceanModel_SlabLandModel.py`: Coupled slab atmosphere-ocean-land model with Earth landscape.

## Integration with JAX-GCM (JCM)

JAX-ESM is specifically designed for coupling JCM (JAX Climate Model) with ocean models:

### Included Components

1. **JCM (Atmosphere)**
   - Location: `jax_esm/components/JCM/`
   - Wraps JCM atmosphere model from jax-gcm
   - Handles conversion between Dinosaur dynamics states and physics states
   - Supports internal sub-stepping

2. **SlabOceanModel**
   - Location: `jax_esm/components/SlabOceanModel/`
   - Mixed-layer ocean with climatological relaxation
   - Anomaly-based SST evolution using Euler backward scheme

3. **SlabLandModel**
   - Location: `jax_esm/components/SlabLandModel/`
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

