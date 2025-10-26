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

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python

import jcm
topo_file = (Path(jcm.__file__).parent / "data/bc/t30/clim/boundaries_daily.nc").resolve()

from jax_esm.coupling.couplers.CoupledJCMSlabOceanModel import CoupledJCMSlabOceanModel
from datetime import datetime

# Creating model
model = CoupledJCMSlabOceanModel(
    topo_file = topo_file,
    horizontal_resolution = 31,
    JCM_layers = 8,
    coupling_timestep = 86400.0,
    JCM_substeps = 24,
    SlabOceanModel_substeps = 1,
    start_datetime = datetime(year=2001, month=1, day=1),
)

# Obtain initial condition
initial_state = model.initialize()

# Run coupled model 
final_state, predictions = model.run(
    init_cplstate = initial_state,
    start_time = 0.0,
    end_time = 86400.0 * 5,  # 5 days
    jax_scan = True,
)

# Convert output into xarray
ds_output = model.predictions_to_xarray(predictions)
     
```

## Architecture

### Component Interface

Each component must inherit from `Component` and implement:

- **`__init__(config)`**: Initialize with ComponentConfig, create state class
- **`initialize()`**: Return initial component state
- **`gen_step_fn()`**: Return a JIT-compiled step function
  - Signature: `step_fn(coupled_state, t) -> (new_component_state, predictions)`

### State Management

States are created dynamically using factory functions:

```python
# Create field groups
ProgClass = create_field_group_class(
    cls_name="MyProg",
    fields=[
        ("temperature", float, (64, 128)),
        ("sim_time", float, ()),
    ],
)

# Create component state
StateClass = create_component_state_class(
    prog_cls=ProgClass,
    phydata_cls=PhydataClass,
)
```

Component states have two main fields:
- **`prog`**: Prognostic variables (temperature, SST, time, etc.)
- **`phydata`**: Physical/diagnostic data (fluxes, derived quantities)

States are JAX pytrees with arithmetic operations via `tree_math.struct`.

### Component Coupling

The current implementation uses **direct coupling**:
- Coupler creates a `CoupledState` with fields for each component
- Step functions receive the full coupled state: `step_fn(cpl, t)`
- Components directly access each other: `cpl.ocn.prog.T`, `cpl.atm.phydata.flux`
- Currently hardcoded to 3 components: `atm`, `flx`, `ocn`

### Time Integration

- Uses `jax.lax.scan` for efficient time stepping
- Components advance in parallel each coupling timestep
- Each component can use internal sub-stepping (`substeps` parameter)
- Debug mode available with `jax_scan=False` (uses Python loop)

## Examples

See the `examples/` and `notebooks/` directories:
- `examples/jax_gcm_slab_ocean.py`: JAX-GCM coupled with slab ocean and flux model
- `notebooks/01TestCoupledSpeedy.ipynb`: Working coupled JCM-FluxModel-SlabOcean example
- `examples/jax_gcm_example.py`: **OUTDATED** - uses old API
- `examples/simple_slab_ocean_test.py`: **OUTDATED** - uses old API

**Note**: Use the notebook or `jax_gcm_slab_ocean.py` as reference for current API.

## Integration with JAX-GCM (JCM)

JAX-ESM is specifically designed for coupling JCM (JAX Climate Model) with ocean models:

### Included Components

1. **JCM (Atmosphere)**
   - Location: `jax_esm/components/JCM/`
   - Wraps JCM atmosphere model from jax-gcm
   - Handles conversion between Dinosaur dynamics states and physics states
   - Supports internal sub-stepping

2. **FluxModel**
   - Location: `jax_esm/components/FluxModel/`
   - Computes total heat flux from atmosphere to ocean
   - Formula: `heatflx = -atm.phydata.surface_flux.hfluxn.sum(axis=-1)`
   - Sign convention: positive upward

3. **SlabOceanModel**
   - Location: `jax_esm/components/SlabOceanModel/`
   - Mixed-layer ocean with climatological relaxation
   - Anomaly-based SST evolution using Euler backward scheme
   - Physics:
     ```python
     cd = rho_ocean * cp_ocean * mixed_layer_depth  # J/K/m²
     T_anomaly_new = time_factor * (T_anomaly + heat_flux * cd_factor)
     SST_new = T_anomaly_new + SST_climatology + climatology_trend * dt
     ```

### Example Coupled System

```python
from jax_esm.components.JCM.JCM import JCM
from jax_esm.components.FluxModel.FluxModel import FluxModel
from jax_esm.components.SlabOceanModel.SlabOceanModel import SlabOceanModel

# Configure components
atm = JCM(atm_config)
flx = FluxModel(flux_config)
ocn = SlabOceanModel(ocean_config)

# Create coupler
coupler = Coupler(
    components={"atm": atm, "flx": flx, "ocn": ocn},
    config=CouplerConfig(timestep=3600.0),
)

# Run coupled simulation
initial_state = coupler.initialize()
final_state, predictions = coupler.run(
    init_cplstate=initial_state,
    start_time=0.0,
    end_time=86400.0 * 30,  # 30 days
    timestep=3600.0,
)
```

## Testing

```bash
# Run all tests
pytest

# Run core coupler tests (no JCM required)
pytest tests/test_coupler.py -v

# Run with coverage
pytest --cov=jax_esm --cov-report=html

# Run specific test
pytest tests/test_integration.py::TestIntegration::test_coupled_initialization -v
```

Test suite includes:
- **Unit tests**: SlabOceanModel, FluxModel, Coupler
- **Integration tests**: Full coupled system validation
- **CI/CD**: GitHub Actions with multi-platform testing

## Current Limitations

- **3-Component System**: Coupler is currently hardcoded to `atm`, `flx`, `ocn` keys
- **Direct Coupling**: Components directly access each other's state (tight coupling)
- **No FluxExchanger**: Flux translation layer exists but is unused
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

