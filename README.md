# JAX-ESM: A JAX-based Earth System Model Coupler

JAX-ESM is a JAX-based coupling framework for Earth system components, specifically designed for coupling JCM (JAX Climate Model) with ocean and flux models. It provides efficient time integration using `jax.lax.scan` and supports component-specific sub-stepping for numerical stability.

## Features

- **JAX-Native**: Fully JIT-compilable, GPU-ready, and differentiable
- **Duck-typed Components**: Any object with `initialize()` and `generate_step_function()` can be coupled; no base class to inherit from
- **Efficient Time Integration**: Uses `jax.lax.scan` for vectorized time stepping with optional debug mode
- **Component Sub-stepping**: Each component can use internal sub-steps for numerical stability
- **Direct Component Coupling**: Components can directly access each other's state for tight integration
- **xarray Integration**: Built-in conversion to xarray Datasets for analysis

## Installation 

```
# JAX-GCM (jcm) >= 2.1 is not on PyPI yet: install its dev branch from source FIRST
git clone https://github.com/climate-analytics-lab/jax-gcm
cd jax-gcm
git switch dev
pip install -e "."
cd ..

# Install JEM
git clone https://github.com/climate-analytics-lab/jax-esm
cd jax-esm
pip install -e "."
cd ..

# Optional: the jittable Veros fork, only needed for the JCM-Veros examples
git clone https://github.com/meteorologytoday/veros-jittable.git
cd veros-jittable
pip install -e "."
```

## Quick Start

Here is a complete, runnable aquaplanet simulation coupling the JCM atmosphere
to JEM's slab ocean. It takes a couple of minutes on a laptop CPU.

```python
from pathlib import Path

import jax_datetime as jdt
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords

from jem import Coupler
from jem.components import JCM, SlabOceanModel
from jem.components.slab.grid import generate_slab_grid

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
one_second = jdt.to_timedelta(1, "second")


# A mapper is any function CoupledCarry -> CoupledCarry: it is the only place
# where components exchange information.
def atm_ocn_mapper(coupled_carry):
    atm = coupled_carry["atm"]
    ocn = coupled_carry["ocn"]
    ocn["forcing"].total_heat_flux = atm["derived"].total_heat_flux
    atm["forcing"].sea_surface_temperature = ocn["state"].sea_surface_temperature
    return coupled_carry


# The JCM atmosphere: a plain jcm.model.Model, adapted in place.
atm_model = JCM.make_jem_compatible(
    jcm.model.Model(coords=get_speedy_coords(), start_date=start_datetime),
    coupling_timestep=coupling_timestep,
)

# Aquaplanet: no mask file, so the fractional mask is all zero (no land).
grid = generate_slab_grid("JCM::T31")

model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=SlabOceanModel(
            grid=grid,
            start_datetime=start_datetime,
            timestep=coupling_timestep / one_second,
        ),
    ),
    mappers=dict(atm_ocn_mapper=atm_ocn_mapper),
)

# The workflow is the coupling scheme: exchange, then step each component.
simulation_interval = jdt.to_timedelta(10, "day")
initial_carry, final_carry, predictions = model.run(
    workflow=["atm_ocn_mapper", "atm", "ocn"],
    iterations=int(simulation_interval / coupling_timestep),
)

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
for component_name, ds in model.predictions_to_xarray(predictions).items():
    ds.to_netcdf(output_dir / f"{component_name:s}.nc", engine="netcdf4")
```

Longer versions of this run, including the sea-ice component and the plotting
code that produced the animation below, are in
`examples/01_basic/01_aquaplanet.ipynb`.

![Surface specific humidity](gallery/JCM_SOM_demo.gif)

## Documentation

For more details, build it locally with:

```
cd jax-esm/docs
pip install -r requirements.txt
make html
```

Then open `docs/build/html/index.html` in your browser.

## Architecture

### Component Interface

Each component needs to provide two methods:

- **`initialize()`**: Return initial component carry value, a pytree.
- **`generate_step_function()`**: Return the step function (the coupler JIT-compiles it)
  - Signature: `step_function(component_carry, step) -> (new_component_carry, predictions)`

### Component Coupling

The current implementation uses direct coupling:
- Coupler creates a dictionary of component carry values.
- Coupler passes the carry value of each component to the corresponding component's `step_function`.
- To exchange variables like fluxes, pass mapper functions to Coupler. A mapper function receives
  the dictionary of component carry values and returns a new one. 

### Time Integration

- Uses `jax.lax.scan` for efficient time stepping
- Within a coupling timestep the workflow runs sequentially, in the order given
- Debug mode available with `jitted=False` (uses a Python loop)

See `docs/source/design/architecture.md` for the carry layout and the full
interface contract.

## Examples
- `examples/01_basic`: aquaplanet setups coupling JCM to the slab models.
- `examples/02_experimental`: features under development, such as earth-like
  topography and JCM-Veros coupling.
- `examples/03_non_geoscience`: a spring system, showing that the coupler is
  not specific to climate components.

## Integration with JAX-GCM (JCM)
JAX-ESM is specifically designed for coupling JCM (JAX Climate Model) with ocean, land, and sea-ice models.

### Included Components

1. **JCM (Atmosphere)**
   - Location: `jem/components/jcm_component.py`
   - Wraps JCM atmosphere model from jax-gcm
   - Handles conversion between Dinosaur dynamics states and physics states
   - Supports internal sub-stepping

2. **Veros (Ocean, full 3D)**
   - Location: `jem/components/veros_component.py`
   - Wraps the [jittable Veros](https://github.com/meteorologytoday/veros-jittable) ocean GCM
   - Optional dependency; lazily imported so `jem.components` works without `veros` installed

3. **SlabOceanModel**
   - Location: `jem/components/slab/slab_ocean_model/`
   - Mixed-layer ocean with climatological relaxation
   - Anomaly-based SST evolution using Euler backward scheme
   - Reports `ice_frazil_melt_energy`, a freeze/melt heat diagnostic for coupling to `SlabSeaiceModel`

4. **SlabLandModel**
   - Location: `jem/components/slab/slab_land_model/`
   - One layer land with climatological relaxation
   - Anomaly-based land surface temperature evolution using Euler backward scheme

5. **SlabAtmosphereModel**
   - Location: `jem/components/slab/slab_atmosphere_model/`
   - Idealized slab atmosphere, used for testing and non-geoscience examples

6. **SlabSeaiceModel**
   - Location: `jem/components/slab/slab_seaice_model/`
   - Basal-only sea-ice thickness model driven by `SlabOceanModel`'s freeze/melt potential
   - Exposes a smooth thickness-to-fraction closure for an atmosphere model's ice-fraction boundary condition

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality (see `tests/` for examples)
4. Ensure the gates pass:
   ```bash
   ruff check .
   JAX_PLATFORMS=cpu pytest tests -q -m "not slow"
   JAX_PLATFORMS=cpu mypy jem/ --ignore-missing-imports
   ```
5. Follow the conventions in `CLAUDE.md`
6. Submit a pull request

## Development Status

- **Version**: 0.1.0 (Alpha)
- **Status**: Prototype coupling framework
- **API Stability**: Subject to change without deprecation until 1.0

## Miscellaneous

The regridding files are generated from repo [EarthSystemGrid.py](https://github.com/meteorologytoday/EarthSystemGrids.py). 

## License

MIT — see [LICENSE](LICENSE).
