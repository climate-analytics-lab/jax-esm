# JAX-ESM: A JAX-based Earth System Model Coupler

JAX-ESM is a flexible, efficient coupling framework for Earth system components written in JAX. It provides a clean interface for translating boundary conditions between atmospheric, oceanic, and land surface models, with support for different time steps and efficient integration using `jax.lax.scan`.

## Features

- **Modular Design**: Clean component interface for easy integration of new models
- **Flexible Time Integration**: Support for different time steps across components with automatic subcycling
- **Efficient JAX Implementation**: Leverages `jax.lax.scan` for fast, vectorized time integration
- **Boundary Condition Translation**: Automatic translation of fluxes and boundary conditions between components
- **Conservation-Ready**: Framework supports flux conservation checks
- **Grid Interpolation**: Utilities for coupling components with different resolutions

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
import jax
from jax_esm import Component, ComponentConfig, Coupler

# Define your components (atmosphere, ocean, land, etc.)
class MyAtmosphere(Component):
    # Implementation...
    pass

class MyOcean(Component):
    # Implementation...
    pass

# Configure components
atmosphere = MyAtmosphere(ComponentConfig(
    name="atmosphere",
    timestep=1800.0,  # 30 minutes
    grid={"type": "latlon", "nlat": 64, "nlon": 128},
    params={}
))

ocean = MyOcean(ComponentConfig(
    name="ocean", 
    timestep=3600.0,  # 1 hour
    grid={"type": "latlon", "nlat": 64, "nlon": 128},
    params={}
))

# Create coupler
coupler = Coupler(
    components={"atmosphere": atmosphere, "ocean": ocean},
    coupling_timestep=3600.0,  # 1 hour
)

# Initialize and run
rng_key = jax.random.PRNGKey(42)
initial_states = coupler.initialize(rng_key)
final_states, history = coupler.run(
    initial_states=initial_states,
    start_time=0.0,
    end_time=86400.0,  # 1 day
)
```

## Architecture

### Components

Each Earth system component must implement the `Component` interface:

- `initialize()`: Set up initial state
- `step()`: Advance state by one timestep and compute output fluxes
- `compute_tendencies()`: Calculate tendencies for prognostic variables
- `get_boundary_fields()`: Extract fields needed by other components
- `get_required_fluxes()`: Declare required input fluxes
- `get_provided_fluxes()`: Declare provided output fluxes

### State Management

Component states are organized into:
- **Prognostic**: Variables that evolve with time (temperature, winds, etc.)
- **Diagnostic**: Computed quantities (precipitation, cloud fraction, etc.)
- **Boundary**: Surface properties and boundary conditions
- **Forcing**: External forcing fields
- **Metadata**: Time, coordinates, and other information

### Flux Exchange

The `FluxExchanger` handles:
- Mapping flux names between components
- Applying transformations (e.g., wind speed to wind stress)
- Accumulating fluxes from multiple sources
- Conservation checks (optional)

### Time Integration

The `TimeIntegrator` provides:
- Automatic subcycling for components with different timesteps
- Efficient integration using `jax.lax.scan`
- Support for saving states at specified frequencies

## Examples

See the `examples/` directory for:
- `jax_gcm_example.py`: Basic coupling example with simplified atmosphere and ocean
- `jax_gcm_slab_ocean.py`: Coupling JAX-GCM with the slab ocean model
- `simple_slab_ocean_test.py`: Testing the slab ocean model component

## Integration with JAX-GCM

JAX-ESM is designed to work seamlessly with JAX-GCM. The framework includes:

### Slab Ocean Model

The `SlabOceanModel` component implements a simple mixed-layer ocean following the formula:

```python
@jit
def run_slabocean_model(
    sst, hfluxn, time_factor, cd_factor, 
    sst_clim_1, sst_clim_2, hfluxn_clim
):
    sst_anom = sst - sst_clim_1
    hfluxn_anom = hfluxn - hfluxn_clim    
    new_sst_anom = time_factor * (sst_anom + hfluxn_anom * cd_factor)
    new_sst = new_sst_anom + sst_clim_2
    return new_sst
```

Key features:
- Configurable mixed layer depth and relaxation timescale
- Anomaly-based evolution with climatological relaxation
- Physical heat capacity calculation: `hfluxn = rho_sw * cp_sw * mld * 0.1 / 3600.0`

### JAX-GCM Wrapper

The `JaxGcmWrapper` component adapts JAX-GCM for coupling:

```python
from jax_esm import ComponentConfig, SlabOceanModel, Coupler
from jax_esm.components import JaxGcmWrapper

# Create components
atmosphere = JaxGcmWrapper(atm_config, gcm_model=your_gcm)
ocean = SlabOceanModel(ocean_config)

# Couple them
coupler = Coupler(
    components={"atmosphere": atmosphere, "ocean": ocean},
    coupling_timestep=3600.0,  # 1 hour
)
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

Apache License - see LICENSE file for details.