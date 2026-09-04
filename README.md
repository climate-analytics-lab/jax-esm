# JAX-ESM: A JAX-based Earth System Model Coupler

JAX-ESM is a JAX-based coupling framework for Earth system components, specifically designed for coupling JCM (JAX Climate Model) with ocean, land and sea-ice models. It provides efficient time integration using `jax.lax.scan` and supports component-specific sub-stepping for numerical stability.

## Features

- **JAX-Native**: Fully JIT-compilable, GPU-ready, and differentiable
- **A small component contract**: any object with a `name`, an `initialize()` and a
  `step(carry, time)` is a component (`jem.base.component.Component`, a
  `typing.Protocol`); there is no base class to inherit from, so an external model
  is adapted by a thin wrapper class
- **One clock**: the `Coupler` owns the coupling timestep, start date and calendar and
  hands every component the same `CouplingTime`, so components cannot disagree about
  the date and the seasonal cycle survives chunked runs and restarts
- **Efficient Time Integration**: `Coupler.generate_trajectory_function()` returns a
  pure `carry -> (carry, diagnostics)` function built on `jax.lax.scan`
- **Differentiable parameters**: component parameters are `flax.struct` dataclasses that
  travel in the carry, so `jax.grad` through a coupled run reaches them
- **xarray Integration**: `Coupler.to_xarray()` labels every component's output on the
  same time axis and grid coordinates, so the datasets merge

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
from jem.components import JCMComponent, SlabOceanModel
from jem.components.slab import SlabGrid

start_date = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")


# An exchanger is the only place where components exchange information. It is
# traced with everything else, so it must not write into the carries it is
# handed: it builds new ones and returns the mapping to continue with.
def atm_ocn_exchange(components, time):
    del time  # this exchange does not depend on the date
    atm, ocn = components["atm"], components["ocn"]
    ocn = dict(
        ocn,
        forcing=ocn["forcing"].replace(
            total_heat_flux=atm["derived"].total_heat_flux,
        ),
    )
    atm = dict(
        atm,
        forcing=atm["forcing"].replace(
            sea_surface_temperature=ocn["state"].sea_surface_temperature,
        ),
    )
    return dict(components, atm=atm, ocn=ocn)


# The JCM atmosphere: a plain jcm.model.Model, wrapped as a component.
atm_model = jcm.model.Model(coords=get_speedy_coords(), start_date=start_date)
atm = JCMComponent(atm_model)

# Aquaplanet: the slab grid is built from the atmosphere's own horizontal grid,
# and with no fractional mask every cell is ocean.
grid = SlabGrid.from_coords(atm_model.coords.horizontal)

coupler = Coupler(
    {"atm": atm, "ocn": SlabOceanModel(grid)},
    {"atm_ocn_exchange": atm_ocn_exchange},
    coupling_timestep=coupling_timestep,
    start_date=start_date,
)
print(repr(coupler))

# The default workflow is every exchanger followed by every component, so the
# fields are exchanged first and both components then step on the same state.
simulation_interval = jdt.to_timedelta(10, "day")
run = coupler.generate_trajectory_function(
    int(simulation_interval / coupling_timestep)
)
final_carry, diagnostics = run(coupler.initialize())

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
for component_name, ds in coupler.to_xarray(diagnostics).items():
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

### The component contract

A component is any object satisfying `jem.base.component.Component`:

- **`name: str`** — its name in the coupler's workflow, carry and output.
- **`initialize() -> carry`** — build the initial carry, a pytree. It must not
  integrate the model.
- **`step(carry, time) -> (new_carry, diagnostics)`** — advance one coupling
  timestep. `time` is a `CouplingTime` (the coupler's clock); the returned carry
  must have exactly the structure, shapes and dtypes of the one it received, or
  `lax.scan` rejects it. `diagnostics` is the per-step output pytree, which the
  coupler stacks along a leading time axis.

Three capabilities are optional and detected with `isinstance`:
`SupportsXarray` (`to_xarray(diagnostics, time)`), `SupportsCheckpoint`
(`save_state`/`load_state`) and `SupportsBind` (`bind(coupling_timestep=...,
start_date=..., calendar=...)`, called once by the coupler at registration for
components with an internal timestep, such as JCM and Veros).

### Component coupling

Components never call each other. An **exchanger** —
`Callable[[dict[str, Carry], CouplingTime], dict[str, Carry]]` — receives the
mapping of every component's carry and returns the mapping to continue with. It
is traced along with everything else, so it must be pure: build new structs
(`.replace(...)` / `dataclasses.replace`) rather than assigning into the carries
it was handed, and never change their pytree structure.

### Time integration

- `Coupler.step_function()` returns one coupled step;
  `Coupler.generate_trajectory_function(iterations, remat=..., jit=...)` drives it
  with `jax.lax.scan`.
- The clock lives in the carry (`CoupledCarry.step`), not in the scan index, so
  calling a trajectory function twice continues the run instead of restarting it.
- Within a coupling timestep the `workflow` runs sequentially in the order given;
  by default that is every exchanger followed by every component.

See `docs/source/design/architecture.md` for the carry layout and the full
contract.

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
   - Location: `jem/components/jcm/`
   - `JCMComponent` wraps a `jcm.model.Model` from jax-gcm; the model itself is
     left untouched
   - Threads JCM's cross-step physics carry through the coupled run, and sub-steps
     internally at JCM's own timestep
   - Publishes the surface exchange (`jem/components/jcm/exchange_fields.py`) in
     JEM's conventions: heat flux positive upward, water fluxes in kg m-2 s-1

2. **Veros (Ocean, full 3D)**
   - Location: `jem/components/veros_component.py`
   - `VerosComponent` wraps the [jittable Veros](https://github.com/meteorologytoday/veros-jittable) ocean GCM
   - Optional dependency; lazily imported so `jem.components` works without `veros` installed

3. **SlabOceanModel**
   - Location: `jem/components/slab/slab_ocean_model/`
   - Mixed-layer ocean with optional Q-flux or relaxation to climatology
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

Each slab model takes its tunables as a `flax.struct` parameter dataclass
(`SlabOceanParameters`, ...) whose numeric fields are pytree leaves, and carries
them in `carry["params"]`, so a gradient of a coupled run with respect to a
physical parameter needs no special casing.

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

- **Version**: single-sourced from `jem.__version__`
- **Status**: Alpha. The next release is 1.0.0a0, the "core API contract"
  described at the top of [CHANGELOG.md](CHANGELOG.md).
- **API Stability**: subject to change without deprecation until 1.0; every
  removal or rename is recorded in [CHANGELOG.md](CHANGELOG.md)

## Miscellaneous

The regridding files are generated from repo [EarthSystemGrid.py](https://github.com/meteorologytoday/EarthSystemGrids.py). 

## License

MIT — see [LICENSE](LICENSE).
