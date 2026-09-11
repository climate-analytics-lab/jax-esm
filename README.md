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
  travel in the carry, so `jax.grad` through a coupled run reaches them — the ones
  `step` reads by replacing the leaf in `carry["params"]`, and the ones that are
  initial conditions by passing them to `initialize`
- **xarray Integration**: `Coupler.to_xarray()` labels every component's output on the
  same time axis and grid coordinates, so the datasets merge
- **One run loop**: `jem.run_chunked()` integrates in chunks, writes a file per
  component per chunk, checkpoints, and stops on an unhealthy state — and every
  run default lives on it
- **One command line**: `python -m jem.main` composes a coupled model from Hydra
  config groups, JAX-GCM's own groups included, re-rooted under `atmosphere`

## Installation 

JAX-ESM is developed and tested against **one** jax-gcm revision, recorded as
`JCM_SUPPORTED_REV` in [`jem/components/jcm/contract.py`](jem/components/jcm/contract.py)
together with every jax-gcm name JAX-ESM calls. The `jcm>=2.1.0b0` floor in
`pyproject.toml` is the loosest statement of the same thing — jax-gcm bumps its
version only at release, so the pin cannot be expressed as a version. Check out
that revision if a coupled run fails with an `AttributeError` inside `jcm`:
`pytest tests/unit/test_jcm_contract.py` reports exactly which name moved.

```
# JAX-GCM (jcm) >= 2.1 is not on PyPI yet: install its dev branch from source FIRST
git clone https://github.com/climate-analytics-lab/jax-gcm
cd jax-gcm
git switch dev                # then `git checkout <JCM_SUPPORTED_REV>` to pin it
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
import jax_datetime as jdt
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords

from jem import Coupler, default_exchangers, run_chunked
from jem.components import JCMComponent, SlabOceanModel
from jem.components.slab import SlabGrid

start_date = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")

# The JCM atmosphere: a plain jcm.model.Model, wrapped as a component.
atm_model = jcm.model.Model(coords=get_speedy_coords(), start_date=start_date)
atm = JCMComponent(atm_model)

# Aquaplanet: the slab grid is built from the atmosphere's own horizontal grid,
# and with no fractional mask every cell is ocean.
grid = SlabGrid.from_coords(atm_model.coords.horizontal)

# An exchanger is the only place where components exchange information.
# `default_exchangers` is the standard wiring written down once — here, the
# atmosphere's surface heat flux drives the ocean and the ocean's SST comes
# back as the atmosphere's boundary condition — filtered to whichever of the
# standard components (`atm`, `ocn`, `lnd`, `seaice`) are present.
components = {"atm": atm, "ocn": SlabOceanModel(grid)}
coupler = Coupler(
    components,
    default_exchangers(components),
    coupling_timestep=coupling_timestep,
    start_date=start_date,
)
print(repr(coupler))

# One run loop for every coupled run: integrate a chunk, write one file per
# component, check the atmosphere is still healthy, repeat. Every run default
# lives on `run_chunked` itself.
result = run_chunked(
    coupler, total_time="10 days", chunk="5 days", output_dir="output"
)
print(result.steps_completed, "coupled steps;", len(result.paths), "files")
```

An exchange the standard table cannot express — one that regrids, computes a
flux, converts units or blends two fields — is written as a plain function
instead; `docs/source/tutorial.rst` works one through. Longer versions of this
run, including the sea-ice component and the plotting code that produced the
animation below, are in `examples/01_basic/01_aquaplanet.ipynb`.

![Surface specific humidity](gallery/JCM_SOM_demo.gif)

## Running from the command line

The same run as one command. `python -m jem.main` (or the `jem` console script)
composes the model from Hydra config groups: JAX-ESM's own groups at the top
level, and **jax-gcm's own groups re-rooted under `atmosphere`**, so anything
that works in `python -m jcm.main` works here with the group's package spelled
out.

```bash
python -m jem.main +configuration=aquaplanet-slab coupled_run=smoke
python -m jem.main --help       # every group, option and override spelling
python -m jem.main +configuration=earth-slab --cfg job   # compose, print, don't run
```

| To do this | Write this |
| --- | --- |
| Run a named coupled configuration | `+configuration=earth-slab` |
| Compose a whole jax-gcm bundle as the atmosphere | `+configuration@atmosphere=speedy-t31` |
| Change one atmosphere group | `physics@atmosphere.physics=held_suarez grid@atmosphere.grid=held_suarez_t31_l8` |
| Set one atmosphere key | `atmosphere.run.time_step=7` |
| Choose a surface component | `ocean=slab_relax ocean.sst_clim_file=${jcm_data:bc/t30/clim/forcing.nc}` |
| Drop one | `land=none` |
| Set a component parameter | `+ocean.params.relaxation_time=1e6` |
| Override a physical constant, for every component | `+atmosphere.constants.grav=9.7` |
| Choose the run settings | `coupled_run=smoke`, or `coupled_run.total_time="90 days"` |

Two things worth knowing:

- **Spell the `@atmosphere`.** `+configuration=speedy-t31` without it composes
  that jax-gcm bundle at the *root*, where its `physics`, `terrain` and `run`
  keys are nobody's and nothing reads them. The atmosphere's groups always
  carry their package: `<group>@atmosphere.<group>=<option>`.
- **The coupled run's own settings are `coupled_run`, not `run`.**
  `atmosphere.run` is the atmosphere's run config, and a `run` group here would
  shadow jax-gcm's. `coupled_run/default.yaml` is the complete schema, so every
  key is overridable without a `+`.
- **The exit status means what a scheduler thinks it means.** `0` when the run
  reached the time it was asked for, `1` when the health gate stopped it early
  (the output and checkpoint written so far are kept, and the reason is
  logged). So `python -m jem.main ... && <post-processing>` runs the
  post-processing only on a run that finished.

The YAML is wiring only — `_target_`, required input files, and the non-default
choices that define a named configuration. Every physics default lives on the
Python class that owns it, and every *run* default on `jem.driver.run_chunked`.

## Long runs: checkpoints and monthly means

`run_chunked` writes a restart after every chunk when it is given a path, and
resuming is the same call:

```python
result = run_chunked(
    coupler,
    total_time="10 years",
    chunk="30 days",             # a health check, a file and a restart per month
    output_dir="output",
    output_averages=True,        # one record per chunk: the monthly mean
    checkpoint_path="checkpoint",
)
```

Run it again with the same `checkpoint_path` and it continues from the coupled
step the checkpoint holds — `python -m jem.main ... coupled_run=longrun` is the
command-line form. The checkpoint is one directory, rewritten atomically each
chunk; a save interrupted half way through is detected and skipped rather than
resumed from.

For a reduction that must not cost memory proportional to the run, accumulate
it *inside* the scan instead of writing every step out:

```python
from jem.accumulate import monthly_mean

monthly = monthly_mean(coupler)
trajectory = coupler.generate_trajectory_function(365, accumulate=monthly)
carry, accumulator = trajectory(coupler.initialize())
means = monthly.finalize(accumulator)      # one (12, ...) record per variable
```

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

- `Coupler.generate_step_function()` returns one coupled step;
  `Coupler.generate_trajectory_function(iterations, remat=..., jit=...)` drives it
  with `jax.lax.scan`.
- The clock lives in the carry (`CoupledCarry.step`), not in the scan index, so
  calling a trajectory function twice continues the run instead of restarting it.
- Within a coupling timestep the `workflow` runs sequentially in the order given;
  by default that is every exchanger followed by every component. It may be
  written nested, and a name may appear more than once — an element listed *n*
  times runs *n* times per coupled step on a clock *n* times faster, which is
  how one part of a model runs a fast loop inside a slower coupling:

  ```python
  workflow=[["atm_lnd_exchange", "atm", "lnd"] * 24, "atm_ocn_exchange", "ocn"]
  ```
- A `Coupler` is itself a component, so a coupled model can be a component of a
  slower one — the same fast loop written as a model in its own right:

  ```python
  fast = Coupler({"atm": atm, "lnd": lnd}, {"atm_lnd_exchange": ...},
                 coupling_timestep=jdt.to_timedelta(1, "hour"), start_date=start_date)
  model = Coupler({"atm_lnd": fast, "ocn": ocn}, {"atm_ocn_exchange": ...},
                  coupling_timestep=jdt.to_timedelta(1, "day"), start_date=start_date)
  ```

  The outer step must be a whole multiple of the inner one; `jem.nested_carry` /
  `jem.with_nested_carry` are how an outer exchanger reaches an inner component,
  and the inner datasets come out of `to_xarray` under their own names, on the
  inner clock. The two forms produce identical runs — see
  `docs/source/design/architecture.md`.
- `model.save_state(carry, directory)` / `model.load_state(directory)`
  checkpoint the whole coupled model. The coupler derives each component's
  writer from the component itself, so a driver never lists them; a component
  whose carry is not a plain pytree (`VerosComponent`, through Veros' HDF5
  restart file) writes its own subdirectory, and because a `Coupler` is one of
  those components, a nested model checkpoints by recursion.

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

How a parameter is varied depends on when the model reads it:

```python
# A PROCESS parameter is read by step() out of the carry, every step, so it is
# varied by replacing that leaf:
carry = ocn.initialize()
carry["params"] = carry["params"].replace(relaxation_time=tau)

# An INITIAL-CONDITION parameter (initial_sst, initial_ice_thickness, every
# SlabAtmosphereParameters field) is read once, by initialize(), and never
# again -- replacing it in an existing carry does nothing, so it is given to
# initialize instead:
carry = ocn.initialize(ocn.params.replace(initial_sst=sst0))

# In a coupled model, through the coupler, which routes by component name:
coupled_carry = coupled.initialize({"ocn": ocn.params.replace(initial_sst=sst0)})
```

Both are differentiable: `jax.grad` of a trajectory reaches a process parameter
through the carry and an initial condition through `initialize`. Each
`*Parameters` docstring says which of its fields are initial conditions;
`docs/source/design/architecture.md` has the full pattern.

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
- **Status**: Alpha. The next release is 1.0.0b0, "the driver and configuration
  layer", described at the top of [CHANGELOG.md](CHANGELOG.md).
- **API Stability**: subject to change without deprecation until 1.0; every
  removal or rename is recorded in [CHANGELOG.md](CHANGELOG.md)

## Miscellaneous

The regridding files are generated from repo [EarthSystemGrid.py](https://github.com/meteorologytoday/EarthSystemGrids.py). 

## License

MIT — see [LICENSE](LICENSE).
