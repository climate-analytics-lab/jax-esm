# JAX-ESM: A JAX-based Earth System Model Coupler

JAX-ESM is a JAX-based coupling framework for Earth system components, specifically designed for coupling JCM (JAX Climate Model) with ocean, land and sea-ice models. It provides efficient time integration using `jax.lax.scan` and supports component-specific sub-stepping for numerical stability.

## Features

- **JAX-Native**: Fully JIT-compilable, GPU-ready, and differentiable
- **A small component contract**: any object with a `name`, an `initialize()` and a
  `step(carry, time)` is a component (`jem.base.component.Component`, a
  `typing.Protocol`); there is no base class to inherit from, so an external model
  is adapted by a thin wrapper class
- **One clock**: the `Coupler` carries a single `jax_datetime.Datetime`, advanced by
  the coupling timestep every step, and hands every component the same
  `CouplingTime` built from it, so components cannot disagree about the date and
  the seasonal cycle survives chunked runs and restarts. There is no calendar
  option — the clock is `jax_datetime`'s proleptic Gregorian, full stop
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
together with every jax-gcm name JAX-ESM calls. The `jcm>=3.0.0rc1` floor in
`pyproject.toml` is the loosest statement of the same thing — jax-gcm bumps its
version only at release, so the pin cannot be expressed as a version. Check out
that revision if a coupled run fails with an `AttributeError` inside `jcm`:
`pytest tests/unit/test_jcm_contract.py` reports exactly which name moved.

```
# JAX-GCM (jcm) >= 3.0 is not on PyPI yet: install its dev branch from source FIRST
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

# Optional: the jittable Veros fork, only needed for the Veros configurations
# (+configuration=veros-double-drake / veros-earth)
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
atm_model = jcm.model.Model(coords=get_speedy_coords(), start_time=start_date)
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
# lives on `run_chunked` itself. Each file is named after the coupled step its
# chunk starts at, so this writes `atm-00000000.nc` and `atm-00000005.nc`
# (and the ocean's two) into `output/`.
result = run_chunked(
    coupler, total_time="10 days", chunk="5 days", output_dir="output"
)
print(result.steps_completed, "coupled steps;", len(result.paths), "files")
```

An exchange the standard table cannot express — one that regrids, computes a
flux, converts units or blends two fields — is written as a plain function
instead; `docs/source/adding_a_component.rst` works one through, and
`docs/source/python_api.md` is this same block with the full construction
around it. Longer versions of this run, including the sea-ice component and
the plotting code that produced the animation below, are in
`examples/01_basic/01_aquaplanet.ipynb` -- `examples/README.md` lists every
example and the command or notebook that runs it.

![Surface specific humidity](gallery/JCM_SOM_demo.gif)

## Running from the command line

The same run as one command. `python -m jem.main` (or the `jem` console script)
composes the model from Hydra config groups: JAX-ESM's own groups at the top
level, and **jax-gcm's own groups re-rooted under `atmosphere`**, so anything
that works in `python -m jcm.main` works here with the group's package spelled
out.

```bash
python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run
python -m jem.main --help       # every group, option and override spelling
python -m jem.main +configuration=earth-slab --cfg job   # compose, print, don't run
```

| To do this | Write this |
| --- | --- |
| Run a named coupled configuration | `+configuration=earth-slab` |
| Compose a whole jax-gcm bundle as the atmosphere | `+configuration@atmosphere=speedy-t31` |
| Change one atmosphere group | `physics@atmosphere.physics=held_suarez grid@atmosphere.grid=held_suarez_t31_l8` |
| Set one atmosphere key | `atmosphere.run.time_step=7` |
| Choose a surface component | `ocean=slab_relax ocean.sst_clim_file='${jcm_data:bc/t30/clim/forcing.nc}'` |
| Drop one | `land=none` |
| Set a component parameter | `+ocean.params.relaxation_time=1e6` |
| Override a physical constant, for every component | `+atmosphere.constants.grav=9.7` |
| Choose the run settings | `coupled_run=short_run`, or `coupled_run.total_time="90 days"` |

Things worth knowing:

- **Spell the `@atmosphere`.** `+configuration=speedy-t31` without it composes
  that jax-gcm bundle at the *root*, where its `physics`, `terrain` and `run`
  keys are nobody's and nothing reads them. The atmosphere's groups always
  carry their package: `<group>@atmosphere.<group>=<option>`.
- **Single-quote a `${...}` resolver**, as `ocean.sst_clim_file='${jcm_data:...}'`
  does above: it is a resolver Hydra expands when the config is composed, and
  an unquoted one is expanded by the shell first — to nothing — so the
  override arrives empty.
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

## Long runs: checkpoints and chunk means

`run_chunked` writes a restart after every chunk — checkpointing is **on by
default** — and resuming is the same call:

```python
result = run_chunked(
    coupler,
    total_time="6 years",        # 2190 days: a whole number of 30-day chunks
    chunk="30 days",             # a health check, a file and a restart per chunk
    output_dir="output",
    output_averages=True,        # one record per chunk: its 30-day-window mean
    # checkpoint_path="checkpoint" is the default, relative to output_dir
)
```

Run it again with the same `output_dir` (or the same `checkpoint_path`) and it
continues from the coupled step the checkpoint holds —
`python -m jem.main ... coupled_run=long_run` is the command-line form. A
*relative* `checkpoint_path` resolves against `output_dir`, so every run gets
its own restart directory — Hydra makes a fresh output directory per run — and
resuming is deliberately the same action that would otherwise overwrite a run's
output. An absolute path is used as given, and `checkpoint_path=None`
(`coupled_run.checkpoint_path=null`) turns checkpointing off. The checkpoint is one directory, rewritten atomically each
chunk; a save interrupted half way through is detected and skipped rather than
resumed from.

Because resuming is the same command as starting, the run **says where its
starting state came from**, in one INFO line before anything is compiled —
`coupler.initialize()`, the `initial_carry` argument, or a named checkpoint,
always with the coupled step. When a `checkpoint_path` holds no complete
checkpoint the line before says so in as many words: every component is
starting from its initial state rather than from a restart. That is a WARNING
when the directory is the wreckage of an interrupted save, and INFO when there
is simply nothing there — which, with checkpointing on by default, is what
every first run sees. `load_carry` names
each component's own source in turn. Loading is all-or-nothing: every leaf comes
from the checkpoint, and a component the checkpoint does not hold is an error,
never a silent fresh initialization.

The gate runs before the checkpoint, and a chunk it rejects is not
checkpointed: there is one restart directory and it is overwritten in place, so
a stopped run leaves it holding the last chunk that passed rather than the
state that failed.

`checkpoint_interval` (`coupled_run.checkpoint_interval`, null by default)
saves less often than every chunk, for a run whose chunks are short for one of
the other reasons a chunk exists — a health check every few days, an output
file per day. It is a whole number of chunks, counted in coupled steps from the
start of the *run*, so a resumed run checkpoints where an uninterrupted one
would; a completed run always checkpoints its last chunk, and a run the health
gate stops always checkpoints the last chunk that passed, so neither loses work
to it. A run that is *killed* falls back to the last interval boundary and
re-integrates the chunks after it on the resume, **rewriting** their output
files — safe, because each file is named after the coupled step its chunk
starts at, so the second pass writes the same names from the same state.

That last part holds only while the resume keeps the same `chunk` and runs at
least as far as the killed pass got. A resume that *rechunks* starts its files
at different steps, so it would write beside the killed run's leftovers rather
than over them and leave two passes' records for the same simulated time in one
directory; a resume that stops earlier leaves that pass's later files stranded
beyond its own end. So a resumed run checks, before anything is compiled, that
every output file at or after the step it resumed from is one it really writes
over — on one of its own chunk boundaries **and** before the step it stops at —
and says at INFO how many it will rewrite — or, for a chunk it keeps no record
of, remove. Anything else and it **refuses**, with a `ValueError` naming the
files it would leave behind, grouped by which of the two they are (an overlap,
or past the end of this run), plus the step, the chunk and the ways out (resume
with the chunk those files were written under and, for those past the end, a
`total_time` that reaches them; remove them; or write into another
`output_dir`). It never deletes a file it is not going to write: which of the
two passes to keep is the user's call, not the driver's. The one thing it does
remove is a name it is itself responsible for and keeps no record for — a chunk
the `subsample` stride lands on none of, whose output from this pass is
nothing. Files from before the restart point, and files this coupler would
never have written, are not in question — and the check is skipped entirely for
a run that writes no files: an accumulated run, or a call whose checkpoint has
already reached `total_time` and so has nothing left to integrate.

`output_averages` and `subsample` reduce the *files* only. The health check is
given each chunk exactly as it was integrated — every record — because it
judges a chunk by its last record and its extremes, and a chunk mean (which
skips NaNs) or a stride that drops the last record would report an atmosphere
that blew up at the end of the month as healthy.

`subsample=n` keeps every *n*-th **coupled step of the run**, counting from
its start, with every record that step produced (a component the workflow runs
several times per coupled step keeps all of them, or none). So the files hold
the same records however the run was chunked and wherever it was resumed, and
`chunk` stays free to be chosen for memory and restart granularity alone. A
chunk that contains no step on the stride writes no file at all (and removes
one an earlier pass left at that name), so the output is one file per
component per chunk except for the chunks that kept nothing.

For a reduction that must not cost memory proportional to the run, accumulate
it *inside* the scan instead of writing every step out:

```python
from jem.accumulate import monthly_mean, windowed_mean

monthly = monthly_mean(coupler)                            # 12 calendar months
pentads = windowed_mean(coupler, "5 days", n_windows=73)   # or any fixed window
trajectory = coupler.generate_trajectory_function(365, accumulate=monthly)
carry, accumulator = trajectory(coupler.initialize())
means = monthly.finalize(accumulator)      # one (12, ...) record per variable
```

Each record is binned by the real Gregorian calendar month of its own interval
midpoint (`jcm.date.gregorian_ymd_from_days`), the same instant it is labelled
with in the written output, so `monthly.finalize(...)` and
`to_xarray(...).groupby("time.month").mean()` of the same run are the same
numbers exactly — a leap February holds 29 records, not 28, with no separate
model-calendar-vs-label reconciliation to make: there is one calendar, the
proleptic Gregorian `jax_datetime` uses throughout.

`run_chunked(..., accumulate=monthly, health_check=None)` does the same from
the driver, threading the accumulator across the chunks and returning it on
`RunResult.accumulator`. An accumulated run has no per-step diagnostics, so it
writes no files, cannot run the health gate (which is refused rather than
skipped — losing the gate has to be a decision), and does not checkpoint the
accumulator: the checkpoint is the model's restart state, the accumulator is an
analysis product, and a resumed run therefore accumulates only what it
integrates.

A component the workflow runs *n* times per coupled step keeps that axis —
`(12, n, ...)`, the monthly mean of each sub-step slot — and each of its
records is binned by its own midpoint, not the coupled step's. A nested
coupler's inner steps are treated the same way. Fold that axis away with
`fold_records`, which weights each slot by its own count (a straight mean
over the slots is right only where every slot holds the same number of
records, which is what a month boundary breaks):

```python
from jem.accumulate import fold_records

sums, counts = accumulator                    # counts["atm"]: (12, n)
means = monthly.finalize(accumulator)         # means["atm"]:  (12, n, ...)
per_month = fold_records(means["atm"], counts["atm"])          # (12, ...)
```

**Twelve bins or one per month of the run.** `monthly_mean(coupler)` bins into
the twelve calendar months, so a ten-year run composites its ten Januaries —
a climatology. Give it a size and it bins into the months the run passes
through instead, in order, each with a bin of its own:

```python
months = monthly_mean(coupler, total_time="3650 days")   # or n_months=121
means = months.finalize(accumulator)   # ~121 bins: Jul 2001, Aug 2001, …
```

These are real calendar months whatever day the run starts on — read directly
off each record's own date, with no month-length table and no start-of-year
phase to compute — so they never drift, whatever the coupling timestep. A run
longer than the accumulator wraps modulo `n_months`, compositing whole
calendar months (never a part of one); size it with `total_time` (which
counts the distinct months the run's records actually touch) to avoid the
wrap entirely.

`windowed_mean(coupler, window, n_windows=...)` is a different, simpler
reduction: `n_windows` windows of a fixed length — the 5-day and 7-day means a
sub-seasonal forecast is scored on — measured in whole records from the run's
own start, with no reference to any calendar, sized either by `n_windows` or
by `total_time`. A run longer than the accumulator wraps, so window *w*
composites every *w*-th window. `window` may also be a **sequence** of
lengths, which the windows cycle through (daily leads for a forecast's first
week, then pentads). A window is never a calendar month, whatever its length
and however it is phased — calendar months come from `monthly_mean`, which
reads them off the real date instead of a length pattern.

The accumulator is an ordinary pytree in the scan carry, so **a binned mean is
differentiable**: `jax.grad` of a loss on `monthly.finalize(accumulator)`
reaches a component parameter through the reduction exactly as it does through
the trajectory, which is what calibrating against monthly observations needs.
See the worked example in `docs/source/design/architecture.md`.

## Documentation

For more details, build it locally with:

```
cd jax-esm/docs
pip install -r requirements.txt
make html
```

Then open `docs/build/html/index.html` in your browser. The two starting
points are `docs/source/getting_started.rst` (install and the command line)
and `docs/source/python_api.md` (the complete direct-Python construction).

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
(`save_carry`/`load_carry`) and `SupportsBind` (`bind(coupling_timestep=...,
start_date=...)`, called once by the coupler at registration for components
with an internal timestep, such as JCM and Veros).

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
- The clock lives in the carry (`CoupledCarry.time`, a carried
  `jax_datetime.Datetime` advanced by the coupling timestep every step — plus
  `CoupledCarry.step`, a plain counter), not in the scan index, so calling a
  trajectory function twice continues the run instead of restarting it.
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
- `model.save_carry(carry, directory)` / `model.load_carry(directory)`
  checkpoint the whole coupled model. The coupler derives each component's
  writer from the component itself, so a driver never lists them; a component
  whose carry is not a plain pytree (`VerosComponent`, through Veros' HDF5
  restart file) writes its own subdirectory, and because a `Coupler` is one of
  those components, a nested model checkpoints by recursion.

See `docs/source/design/architecture.md` for the carry layout and the full
contract.

## Examples

Every example is either one `python -m jem.main +configuration=...` command
or a notebook doing one thing the command line cannot (plotting, building a
carry by hand); `examples/README.md` lists every one of them with the command
or notebook that runs it. `examples/03_non_geoscience` couples a spring system
rather than an atmosphere and an ocean, showing that the coupler is not
specific to climate components.

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
  layer".
- **API Stability**: subject to change without deprecation until 1.0; git
  history is the record of every removal or rename.

## Miscellaneous

The regridding files are generated from repo [EarthSystemGrid.py](https://github.com/meteorologytoday/EarthSystemGrids.py). 

## License

MIT — see [LICENSE](LICENSE).
