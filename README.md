# JAX-ESM: A JAX-based Earth System Model Coupler

JAX-ESM is a JAX-based coupling framework for Earth system components, specifically designed for coupling JCM (JAX Climate Model) with ocean and flux models. It provides efficient time integration using `jax.lax.scan` and supports component-specific sub-stepping for numerical stability.

## Features

- **JAX-Native**: Fully JIT-compilable, GPU-ready, and differentiable
- **Dynamic State Creation**: Factory functions for creating custom component states with arithmetic operations
- **Efficient Time Integration**: Uses `jax.lax.scan` for vectorized time stepping with optional debug mode
- **Component Sub-stepping**: Each component can use internal sub-steps for numerical stability
- **Direct Component Coupling**: Components can directly access each other's state for tight integration
- **xarray Integration**: Built-in conversion to xarray Datasets for analysis

## Lightning Start: Installation + Run
Copy and paste in your terminal to set up a fresh Conda environment and run example code.
```
conda create -y -n jem_fresh python=3.13
conda activate jem_fresh

# Remember to replace `your_credential`.
git clone https://[your_credential]@github.com/climate-analytics-lab/jax-esm.git
cd jax-esm

pip3 install -e ".[jcm,plot]"

export PYTHONPATH=`pwd`

# You can run the following file directly, or with Notebook+Jupytext.
python3 jupytext_notebooks/jem_JCM_SlabOceanModel_SlabLandModel.py
```

## Installation 

Please install `jax-gcm` (hereafter `jcm`) and `matplotlib` with
```bash
pip install -e ".[jcm,plot]"
```

For development, please run additionally
```bash
pip install -e ".[dev]"
```

## Quick Start

You can run a Jax-gcm coupled run with
```
python3 jupytext_notebooks/jem_JCM_SlabOceanModel_SlabLandModel.py
```
whose essential code is below
```python
# Same as jupytext_notebooks/jem_JCM_SlabOceanModel_SlabLandModel.py 
# but only the essential part
import jcm
from jcm.geometry import Geometry
import jax_datetime as jdt

from jem.tool_scripts.generate_jcm_forcing_and_topography_files import (
    generate_jcm_forcing_and_topography_files,
)
from jem.components import JCM, SlabLandModel, SlabOceanModel
from jem.mapping import IdentityRegridder
from jem.mapping import BasicForcingMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_interval = jdt.to_timedelta(10, "day")
output_dir = Path("output/JCM_SOM_SLM").resolve()

external_files = generate_jcm_forcing_and_topography_files()
output_dir.mkdir(exist_ok=True, parents=True)
geometry = Geometry.from_file(external_files["terrain"])
one_second = jdt.to_timedelta(1, "second")

# Creating components
atm_model = jcm.model.Model(
    start_date=start_datetime,
    geometry=geometry
)

JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
    save_interval=jdt.to_timedelta(12, "hour"),
)

components = dict(
    atm=atm_model,
    ocn=SlabOceanModel(
        start_datetime=start_datetime,
        mask_file=external_files["terrain"],
        SST_clim_file=external_files["forcing"],
    ),
    lnd=SlabLandModel(
        start_datetime=start_datetime,
        topography_file=external_files["terrain"],
        mask_file=external_files["terrain"],
        land_clim_file=external_files["forcing"],
    ),
)

# Creating regridders and mapping
identity_regridder = IdentityRegridder()
forcing_mapper = BasicForcingMapper(components=components)
forcing_mapper.add_forcing_mapping(
    source = ("atm", "extra.total_heat_flux"),
    target = ("ocn", "flux.total_heat_flux"),
    regridder = identity_regridder,
)
forcing_mapper.add_forcing_mapping(
    source = ("ocn", "prog.sea_surface_temperature"),
    target = ("atm", "sea_surface_temperature"),
    regridder = identity_regridder,
)
forcing_mapper.add_forcing_mapping(
    source = ("atm", "extra.total_heat_flux"),
    target = ("lnd", "flux.total_heat_flux"),
    regridder = identity_regridder,
)
forcing_mapper.add_forcing_mapping(
    source = ("lnd", "prog.land_surface_temperature"),
    target = ("atm", "stl_am"),
    regridder = identity_regridder,
)

# Make coupled model
model = Coupler(
    components=components,
    forcing_mappers=dict(fm=forcing_mapper),
)

initial_coupled_state_forcing = model.initialize()
trajectory_function = model.generate_trajectory_function(
    workflow=["fm", "atm", "ocn", "lnd"],
    iterations = int(simulation_interval / coupling_timestep),
)

# Run coupled model
state_holder, predictions = trajectory_function(initial_coupled_state_forcing)

# Write output to netcdf files
for component_name, ds in output_dict.items():
    output_file = output_dir / f"{component_name:s}.nc"
    print("Output file: ", str(output_file))
    ds.to_netcdf(output_file, engine="netcdf4")

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

