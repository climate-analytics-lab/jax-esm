# JAX-ESM API and usability review (pre-release, 2026-09)

Scope: the whole `jem` package on `main` (c7b5bda, after #103 "drop-mapping"),
its examples, docs, tests and CI, reviewed against the JCM `dev` branch
(226172e4) that it couples to. The emphasis is usability without giving up
flexibility or differentiability, and convergence with JCM's syntax,
libraries and configuration patterns. The two in-flight PRs (#106, #107) were
read as well because they show where the example code is heading.

The short version: the coupling *idea* is sound and already differentiable end
to end, but almost everything a user touches (building a model, running it,
writing output, restarting) lives in copy-pasted example code rather than in
the package. The fix is structural, not cosmetic: give JAX-ESM the same
`main.py` + `runners.py` + `config/` spine that JCM has, and make the
components and the JCM adapter consume JCM's public objects (coords, terrain,
forcing, constants, parameters) instead of re-deriving them.

---

## 1. What is good and should be kept

- **The coupling model is simple and correct for JAX.** A per-component
  *carry* pytree, a per-component `step(carry, i) -> (carry, diagnostics)`,
  an ordered *workflow* of component and mapper names, one `lax.scan` over
  coupling steps. Mappers are plain functions of the coupled carry. This is
  the right primitive: it composes, it jits, and notebook 03 shows a
  `jax.jvp` through JCM + slab ocean working today.
- **Sub-stepping is handled where it belongs.** JCM integrates its own
  `dt` inside one coupling step (`run_from_state_with_carry`), Veros uses a
  `fori_loop`; the coupler never needs to know.
- **The slab physics is well documented.** The `frzmlt` freeze/melt
  potential in `SlabOceanModel`, the smooth `1 - exp(-h/h0)` ice-fraction
  closure in `SlabSeaiceModel`, the `sqrt` gradient guards in the Veros
  adapter and PR #106: these are exactly the kind of "why" comments JCM's
  `CLAUDE.md` asks for.
- **`ESMFRegridder` is solid.** Sparse weights applied with gather /
  scatter-add, jit-compiled, `vmap`-able, Fortran-order conventions written
  down once and honoured by `SlabGrid` and the SCRIP reader. The SCRIP and
  UGRID readers validate their inputs and fail loudly.
- **The Veros work is genuinely hard and mostly done right** (ghost cells,
  TKE surface forcing, rotation of stress into the rotated grid, checkpoint
  via Veros's own restart writer).

Nothing below argues for throwing any of that away.

## 2. The central problem: the package is missing its driver layer

JCM's user-facing spine is `jcm/main.py` (Hydra entry point),
`jcm/runners.py` (config → objects, chunked run loop, health gate,
checkpoint rotation, output writer) and `jcm/config/<group>/*.yaml`. A
production run is one command, and every named configuration is a YAML file
with comments explaining why each setting is what it is.

JAX-ESM has none of this. Every example re-implements the same driver:

| Driver responsibility | Where it currently lives |
|---|---|
| Build JCM + wrap it + build the slab grid + build components + wire the mapper + `Coupler(...)` | copied into every one of the 5 JCM notebooks, `03_long_aquaplanet.py`, the Veros `main.py`, and twice more in PR #106 (`model_setup.py` ×2) |
| Batch loop over `simulation_interval`, NaN health check, per-component `to_netcdf`, "model exploded" exit | `03_long_aquaplanet.py`, Veros `main.py`, PR #106 `main.py` ×2 (the PR adds re-run-on-explosion and restart-from-checkpoint to two more copies) |
| Output post-processing (`isel(time=::5)`, `reduce(np.mean)`, `xr.merge` of selected variables) | 4 notebooks + 3 scripts, each slightly different |
| CLI arguments | `argparse` in each `main.py`, plus a `run.sh` per example that hard-codes the values |
| Plotting / animation | ~60 lines duplicated in 4 notebooks |

Measured on the five JCM-coupled notebooks (code lines that also appear
verbatim in at least two other notebooks):

| notebook | code lines | duplicated |
|---|---|---|
| 01_basic/01_aquaplanet | 136 | 128 |
| 01_basic/02_aquaplanet_customized_initial_condition | 164 | 135 |
| 01_basic/03 ... using_gradient | 96 | 47 |
| 01_basic/04_jcm_slabs_mixed_grid_aqua_planet | 157 | 125 |
| 02_experimental/01_earth | 145 | 105 |

The genuinely unique content of each example is a handful of lines: which
components, which terrain/forcing file, which grid file, one SST perturbation
(02), one `jax.jvp` call (03), four regridder lines (04). Those are exactly
the things a config file and a couple of CLI overrides should express.

The pattern is not an accident: JAX-ESM's own `CLAUDE.md` mandates it
("Use `argparse` for handling command-line arguments", "Python scripts must
be executed via customizable bash scripts"), which is the opposite of JCM's
"No bespoke run scripts — new configurations go through Hydra". Any agent or
contributor following the repo instructions will keep producing
`main.py` + `run.sh` pairs. Rewrite `CLAUDE.md` to JCM's rules first (Hydra
config groups, `runners.py`, design docs under `docs/source/design/`, the
lint/test gate), otherwise every later fix regresses.

PR #106 makes this worse rather than better. It moves `notebooks/` to
`examples/`, adds a second Veros example whose `model_setup.py` (419 lines)
and `veros_case_setup.py` (368 lines) are copies of the first with ~30 lines
changed, and adds a `run.sh` per example that is invoked by the test-suite.
The `build_model()` function in `model_setup.py` is the right instinct, but
it belongs in `jem/runners.py`, parameterised by config, not in each
example directory.

### 2.1 Recommended target (mirrors JCM exactly)

```
jem/
  main.py          @hydra.main(config_path="config", config_name="config") -> runners.run(cfg)
  runners.py       build_atmosphere(cfg) / build_ocean(cfg) / ... / build_coupler(cfg) / run(cfg)
                   chunked loop, health gate, checkpoint rotation, save_predictions — ONE copy
  config/
    config.yaml    defaults: [atmosphere: jcm, ocean: slab, land: none, seaice: none,
                              coupling: daily, run: default, output: netcdf]
    atmosphere/jcm.yaml           # thin: selects jcm config groups (see 2.2)
    ocean/{slab,slab_qflux,slab_relax,veros}.yaml
    land/{none,slab_speedy}.yaml
    seaice/{none,slab}.yaml
    coupling/daily.yaml           # coupling_timestep, workflow, exchanges (see 4.3)
    regrid/{same_grid,esmf}.yaml  # weight files for a2o / o2a
    run/{default,smoke,longrun}.yaml  # total_time, chunk_days, checkpoint_path, health gate
```

Then the examples collapse to:

```bash
python -m jem.main                                   # notebook 01 (aquaplanet, JCM + slab ocean + slab sea ice)
python -m jem.main seaice=none                       # notebook 03's model
python -m jem.main terrain=from_file land=slab_speedy ocean=slab_relax \
       terrain.file=... ocean.sst_clim_file=...      # notebook 01_earth
python -m jem.main ocean.grid=scrip:DisplacedPoleGrid regrid=esmf   # notebook 04
python -m jem.main ocean=veros ocean.setup=examples/veros/double_drake.py run=longrun
```

Notebook 02 (custom SST) and notebook 03 (`jvp`) keep a short Python cell
each: `coupler = runners.build_coupler(cfg)`, edit the carry, run. Advanced
cases (Veros) keep a bespoke `veros_case_setup.py`, because a Veros
`VerosSetup` subclass *is* unique content, but nothing else.

### 2.2 Reuse JCM's config groups instead of re-describing the atmosphere

JCM already ships its YAML as package data (`config/**/*.yaml` in
`pyproject.toml`), so JAX-ESM can compose it directly:

```yaml
# jem/config/config.yaml
hydra:
  searchpath:
    - pkg://jcm.config
defaults:
  - physics@atmosphere.physics: speedy
  - grid@atmosphere.grid: speedy_t31_l8
  - run@atmosphere.run: default
  - terrain@atmosphere.terrain: aquaplanet
  - forcing@atmosphere.forcing: default
  - diffusion@atmosphere.diffusion: default
  - ocean: slab
  - ...
```

and `runners.build_atmosphere(cfg)` is then a one-liner around
`jcm.runners.build_model(cfg.atmosphere)` and `jcm.runners.build_forcing(...)`.
Two consequences worth spelling out:

1. The JCM config hardening that is in progress flows into JAX-ESM
   automatically. Every knob JCM exposes (`physics=echam`, `init=from_state`,
   `+constants.grav=`, hf:// data paths, `run.start_date`) becomes available
   to a coupled run for free, with the same spelling.
2. JAX-ESM stops carrying its own copies of "build a T31 grid", "load a
   terrain file", "generate forcing files by shelling out to
   `interpolate.py`" (`jem/tool_scripts/generate_jcm_forcing_and_topography_files.py`
   runs a subprocess against a private file inside the jcm package; that
   whole module goes away).

One caveat to plan for: `cfg.atmosphere.run.total_time` must equal the
coupling timestep (JCM is integrated one coupling step at a time), so the
runner should set it, not the user. Make that explicit in the YAML comment.

## 3. Coupler API (`jem/base/coupler.py`)

Mostly fine as a primitive; the problems are in the ergonomics around it.

1. **The coupler does not own time.** `step_function(carry, step)` passes an
   integer step index; every component keeps its own `sim_time` in its state
   and its own `start_datetime`, `timestep` and `calendar` constructor
   arguments. The coupling timestep is therefore specified four times in
   every example (`coupling_timestep` to `make_jem_compatible`,
   `timestep=coupling_timestep / one_second` to each slab, `iterations`
   to `run`), in two different types (`jdt.Timedelta` vs float seconds).
   Recommendation: `Coupler(components, exchanges, coupling_timestep:
   jdt.Timedelta, start_date, calendar)`; the coupler computes the
   `DateData`/sim-time for each step (JCM already has
   `Model._date_from_sim_time`; the equivalent belongs in the coupler) and
   passes it to `step`. Components stop carrying clocks.
2. **`run()` is mistyped and overloaded.** Its annotation says it returns a
   `TrajectoryFunction`; it returns `(initial_carry, final_carry,
   predictions)`. `reuse_last_available_trajectory=True` silently reuses the
   previously compiled trajectory even if `workflow` or `iterations` changed,
   which is a foot-gun for exactly the batch loops that use it. `checkpoint=`
   means `jax.checkpoint` (rematerialisation), which collides with the
   restart-checkpoint concept the examples and `jem.utils.checkpoints`
   use. `jitted=False` switches both jit *and* the scan implementation.
   Recommendation: `generate_trajectory_function(...)` stays the
   compile-once API; `run` becomes the thin chunked driver in `runners.py`;
   rename `checkpoint` to `remat`; split `jitted` into `jit: bool` and
   `python_loop: bool` (or drop the Python loop and rely on
   `jax.disable_jit()`).
3. **Predictions convention forces boilerplate into every component.** The
   coupler stacks per-step outputs and then `unwrap_leading_dims(...,
   first_n_dim=2)` twice, so every component must return arrays with a
   leading time axis. That is why every slab step ends with
   `stack_objects([result])` (a length-1 fake time axis) and why the JCM
   adapter returns a `ModelPredictions` whose `_predictions` attribute is
   later reached into to rebuild it. Recommendation: a component returns a
   plain diagnostics pytree per coupling step; the coupler adds the time
   axis. Components that sub-step and want finer output can return
   `(n_sub, ...)` arrays behind a declared flag.
4. **Mutation.** Mappers mutate `tree_math.struct` fields in place
   (`ocn["forcing"].total_heat_flux = ...`), the step function mutates the
   input dict (`carry[name] = ...`), and notebook 03 assigns into a global
   `initial_coupled_carry` from inside a jitted function (a tracer leak
   waiting to happen for anyone who copies the pattern). Under `lax.scan`
   this is harmless; under `jitted=False` the caller's `initial_carry` *is*
   the final carry after `run()`. JCM's convention is immutable structs +
   `.replace()`. Adopt it in the coupler and in the documented mapper style.
5. **Console noise.** `print` is used 69 times in `jem/`; `logging` zero
   times. `generate_step_function(verbose=True)` prints the flattened
   workflow at every trace; `resolve_interface(verbose=True)` prints every
   member it checks. JCM routes everything through
   `logging.getLogger("jcm")` with a `log_level` config knob. Do the same.
6. **Dead or vestigial code.** `adhoc_scan` times each step and discards
   the timing; `_validate_components` is a no-op; `show_progress` is accepted
   by `generate_step_function` and ignored; `trajectory_holder` is declared
   as `tracjectory_holder`; `Workflow = Pytree` lets any pytree through and
   is then flattened with `jax.tree.flatten`, so a nested workflow has no
   meaning beyond "flatten me".

## 4. Component contract (`jem/base/interface.py`, `typing.py`)

The duck-typed `resolve_interface()` (115 lines, reflection over
`typing.Callable` signatures, a `__JEM_CUSTOMIZED_MAPPING__` renaming hook,
`typeguard` on the resulting dataclass) is more machinery than the contract
needs, and it pushes the adapters into monkey-patching:
`JCM.make_jem_compatible(model)` does `setattr(model, "initialize", ...)`
on a live `jcm.model.Model`, and the Veros adapter does the same plus
overwrites `set_forcing`. That works, but:

- the adapter cannot hold state of its own (the forcing, a cached
  physics carry, the coupling timestep) except via closures;
- `Model` already has `run`, `resume`, `bootstrap_state`... adding
  `initialize` to it invites confusion about which "initialize" is meant;
- neither `mypy` nor a reader can see what the component interface is.

Recommendation: a ~20-line `typing.Protocol`

```python
class Component(Protocol):
    def initialize(self) -> Carry: ...
    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]: ...
    def to_xarray(self, diagnostics) -> xr.Dataset: ...            # optional
    def save_state(self, carry, path) / load_state(self, path): ... # optional (Veros)
```

with `JCMComponent(model, forcing=None, coupling_timestep=...)` and
`VerosComponent(setup, ...)` as ordinary wrapper classes. Keep
`make_jem_compatible` for one release as a deprecated shim if anything
external depends on it. Delete `resolve_interface`, `JEMComponent`,
`typeguard` and the 300 lines of tests that exercise the reflection.

### 4.1 The JCM adapter is fragile in ways that will bite at release

`jem/components/jcm_component.py`:

- **It indexes a diagnostics layout that JCM `dev` has already changed.**
  `hfluxn[:, :, 0/1/2]` and `evap[:, :, 2]` assume the `(ix, il, 3)`
  land/sea/mean channel layout. JCM's release notes ("SPEEDY surface fluxes
  are published as flat 2D maps", #645/#328/#390) collapse these to
  `(ix, il)` grid means and note explicitly that a coupled surface model
  "now needs [per-tile fluxes] from its own land/ocean components". JAX-ESM
  CI is green only because it checks out `jax-gcm` `main` (v2.0.1); the
  next JCM release breaks it. (Verified below in §9 against `dev`.)
- **It reaches into private/physics-package-specific keys.**
  `model._prepare_initial_dycore_state()` (private),
  `predictions.physics["_surface_flux"]`, `["_convection"].precnv`,
  `["_condensation"].precls` (SPEEDY term names; an ECHAM physics package
  has none of these), `ModelPredictions(predictions._predictions, ...)`.
  So "coupled JCM" currently means "coupled SPEEDY"; `physics=echam`, the
  v2 focus of JCM, cannot be coupled.
- **It discards the cross-step physics carry.** It calls
  `run_from_state_with_carry` but drops the returned `final_physics_state`
  and never passes `initial_physics_state`, so the radiation sub-cycle
  cache, prior-step TKE etc. are rebuilt from zero at every coupling step.
  JCM's own `resume()` exists precisely to avoid this. Thread the carry
  through the JEM carry (`carry["physics"]`).
- **`initialize()` runs a full one-step integration** (compile + execute)
  just to discover the diagnostics pytree shape. JCM now has
  `Model.bootstrap_state()` and `physics.get_empty_data(coords)` for this.
- **The `asfloat64` cast** exists "to work around a JAX dtype inconsistency
  bug in JCM where some arrays are initialized as `int32`". With x64
  disabled it is actually a float32 cast of *every* leaf, including the
  physics dict. Find the offending JCM leaf and fix it there.
- **Forcing cannot be supplied.** `make_jem_compatible` always uses
  `default_forcing(...)`. Notebook `01_earth` loads the realistic
  `forcing.nc` with `ForcingData.from_file` and then cannot hand it to the
  atmosphere, so the "Earth-like" run uses aquaplanet albedo, ozone and
  sea-ice defaults with only SST/stl/snowc/soilw overwritten by the mapper.

Two of these are things to ask of JCM as part of its hardening (see §8):
a public, physics-package-independent *surface exchange* contract, and a
public initial-state / empty-diagnostics API. The rest is on the JAX-ESM
side.

### 4.2 Slab components (`jem/components/slab/`)

Physics is fine; the API and the purity are not.

- **Parameters are not differentiable and the step function is not
  pure.** `relaxation_time`, `mixed_layer_depth_min/max`, `tdland`,
  `depth_soil`, ... are Python floats on `self`; `initialize()` then
  computes `self.cd_factor`, `self.time_factor`, `self.rhcapl`,
  `self.cdland`, `self.SST_clim`, `self.fmask_l` and stores them on the
  instance, and `generate_step_function()` closes over them. So (a) calling
  `generate_step_function()` before `initialize()` silently closes over
  `None`; (b) none of the slab parameters can be a `jax.grad` target,
  which defeats the point of a differentiable coupler; (c) `initialize()`
  mutates configuration (`self.relaxation_time = jnp.inf`). JCM's pattern
  is a `flax.struct.dataclass` `Parameters` with numeric tunables as pytree
  leaves and static choices as `pytree_node=False`. Adopt it:
  `SlabOceanParameters(relaxation_time=..., mld_min=..., mld_max=...)`,
  computed factors derived inside `step` (they are a handful of flops).
- **The grid is a second source of truth.** `generate_slab_grid("JCM::T31")`
  rebuilds a dinosaur grid from a string DSL, independently of the
  atmosphere's `coords`, and `load_jcm_fractional_mask` re-reads the
  terrain file that `TerrainData.fmask` already holds. PR #106 then has to
  squeeze/transpose an ERA5 mask by hand because the loader is too
  literal. Replace with `SlabGrid.from_coords(coords.horizontal,
  fractional_mask=terrain.fmask)` plus the existing `from_scrip` /
  `from_ugrid`.
- **Bug: idealised land climatology varies with longitude, not latitude.**
  `SlabLandModel._idealized_land_temperature(shape)` unpacks
  `nlat, nlon = shape`, but `grid.shape` is `(n_lon, n_lat)`; with T31 that
  is `linspace(-90, 90, 96)` along the longitude axis. Shapes line up by
  accident so nothing raises. Any run without `land_clim_file` is wrong.
- **Bug/gap: `Q_flux_file` is validated but never loaded.** With
  `forcing_method="Qflux"` the model checks the file exists and then
  integrates with `OceanForcing.q_flux = zeros` unless the caller pokes the
  array into the carry by hand.
- **Land `alb0` is hard-coded to 0.2** (`TODO: Could load from topography
  file`), so the land-ice heat capacity branch never runs; the "Strong
  relaxation because land model seems to have a bug" comment in notebook
  `01_earth` (`tdland = 86400`) is probably this plus the climatology bug.
- **Physical constants are duplicated** in `jem/constants.py` under new
  names (`atmosphere_specific_heat_capacity_under_constant_pressure = 1004`
  vs JCM `cpd = 1004.64`, `g0` vs `grav`, `freezing_point_K` vs `tmelt`).
  JCM's `jcm.constants` is designed as the single source of truth, with
  `set_constants()` overrides for other planets. Import it; keep only
  genuinely ocean/land/ice constants that JCM lacks, and give those the
  same `PhysicalConstants` treatment.
- **Output metadata diverges from JCM.** Slab datasets use dims
  `("longitude", "latitude")` and 2-D coords `latitude2D/longitude2D`; JCM
  writes `lon/lat` with CF attributes via `jcm.cf_metadata`. A user cannot
  `xr.merge` the atmosphere and ocean outputs of the same run. Use the JCM
  names on JCM-grid slabs and CF `coordinates=` attributes on curvilinear
  ones.
- `SlabOceanModel(mask_value=1.0)` is used in both Veros examples to turn
  the ocean model into a polar "fake land". That hack should be a
  `SlabLandModel` configuration (or a `SlabSurface` with a mask), not an
  inverted mask on the ocean.

### 4.3 Exchanges between components

`BasicMapper` (declarative source → target → regridder mappings, dotted
paths) was deleted in #103, but it is what the README, `quick_start.rst`
and `tutorial.rst` still teach, and it is what a YAML-configurable coupling
needs. The free-function mapper should stay as the escape hatch, but the
common case should be data:

```yaml
# jem/config/coupling/daily.yaml
coupling_timestep: 1 day
workflow: [exchange, atm, ocn, seaice]
exchanges:
  - {src: atm.derived.net_heat_flux,        dst: ocn.forcing.net_heat_flux,   regrid: a2o_conserve}
  - {src: ocn.state.sea_surface_temperature, dst: atm.forcing.sea_surface_temperature, regrid: o2a_bilinear}
  - {src: ocn.derived.ice_frazil_melt_energy, dst: seaice.forcing.ice_frazil_melt_energy}
  - {src: seaice.derived.ice_fraction,      dst: atm.forcing.sice_am,         regrid: o2a_conserve}
```

Flux *computations* that currently live inside mappers (the bulk wind-stress
formula and the "swamp" sea-ice mask in the Veros examples) should be
library functions (`jem.fluxes.bulk_wind_stress`) or components, so they
are tested once and reused.

## 5. Output, restart and the run loop (`jem/utils/checkpoints.py`, examples)

- **No output writer in the package.** JCM has `save_predictions` and
  `run_chunked` with `check_health`, `.prev` rotation and periodic archives.
  JAX-ESM examples each hand-roll `to_netcdf` per component per batch with
  ad-hoc subsampling/averaging, and PR #106 adds "re-run N times on
  explosion" and "resume from the last `batch_*` directory" to two copies
  of `main.py`. One `runners.run_chunked(cfg)` replaces all of it.
- **Checkpoints are pickles** of numpy pytrees (`{name}_carry.pkl`), with
  the Veros special case (`from veros.restart import ...`) living inside
  the generic module. JCM uses `flax.serialization` msgpack with per-leaf
  shape validation and a named error. Reuse that; let components with
  exotic state (Veros) implement optional `save_state/load_state`.
- The examples' health gate is `isnan(specific_humidity)`. JCM's
  `jcm.diagnostics.check_health` is the tested version.

## 6. Packaging, dependencies, CI, docs

Packaging (`pyproject.toml`):

- Hard runtime dependencies include `sphinx`, `shibuya` (a Sphinx theme),
  `matplotlib`, `cartopy`, `dataclasses-json` (unused anywhere),
  `typeguard` (only the reflection layer), `jax_tqdm`/`tqdm` (progress
  bar), `coordax` (used by two utility functions that nothing calls). Move
  plotting/docs to extras (`[plot]`, `[docs]`), delete the unused ones,
  add `[veros]`.
- `jcm>=1.1.1` is the pin; the code needs 2.x APIs
  (`run_from_state_with_carry`, `TerrainData.from_file`). Pin to the JCM
  release the adapter is validated against and bump deliberately.
- License is MIT; JCM is Apache-2.0. Not a bug, but worth a deliberate
  decision before a public release (the README/pyproject author fields
  point at the same lab).
- `[tool.black]` (target py38) and `[tool.isort]` sections configure
  formatters the project does not run; `requires-python>=3.11`.

CI (`.github/workflows/tests.yml`):

- The lint job runs on Python 3.10 (below `requires-python`), unpinned
  `ruff`, and `ruff check jem/` only. With the ruff version JCM pins
  (0.15.17) `ruff check jem/` currently reports 2 errors (E402 in the
  Veros adapter) and `ruff check .` 59 (notebooks, tests). Mirror JCM:
  pin ruff, `ruff check .`, per-file ignores for `*.ipynb`.
- The test job is a 2 OS × 3 Python matrix that executes the notebooks,
  i.e. two 180-day T31 coupled integrations plus a 60-day one, per job, on
  CPU, with a 600 s timeout each. The codecov upload is gated on
  `python-version == '3.10'`, which is not in the matrix. Recommendation:
  unit tests with a coverage gate on push (as JCM: fast at 90%), and
  examples run at `run=smoke` (a few days) as the slow PR gate.
- `pip install -e ".[dev]"` installs `jcm` from PyPI *and* the workflow
  puts a checkout of `jax-gcm` on `PYTHONPATH`; which one is imported
  depends on path order. Install the checkout with `pip install -e`.

Tests:

- Unit tests cover only `Coupler` and `resolve_interface`. There are no
  tests for any slab model (which is how the two bugs above survived), the
  grid readers, `ESMFRegridder` (conservation of a constant field is a
  one-line test), checkpoints, or the JCM adapter. No `check_vjp`/`jvp`
  gradient tests anywhere, in a project whose selling point is
  differentiability.
- `test_coupler.py` carries ~100 lines of commented-out `sys.modules` stub
  injection and a test named `test_dead_code_return_info_is_unreachable`.

Docs:

- `README.md` and `docs/source/quick_start.rst` Quick Start import
  `jem.mapping.BasicMapper` (removed) and construct `SlabOceanModel`
  without the now-required `grid`. They do not run.
- `docs/source/tutorial.rst` uses `model._prepare_initial_modal_state()`
  and `predictions.physics.surface_flux` (neither exists) and a
  `land_model_active` kwarg that `make_jem_compatible` does not take.
- `DEVELOPER.md` documents `jem/mapping/`, `jem/utils/data_structure.py`,
  `jem/base/exceptions.py`, `JCM.py`, `Veros.py` — none exist — and the
  README's "Included Components" section lists `jem/components/JCM/`.
- `docs/source/index.rst` still contains the sphinx-quickstart placeholder
  text; `developers.rst` tells contributors to install extras
  `[jcm,plot]` that are not defined; `docs/requirements.txt` pins
  `sphinx-rtd-theme` while `conf.py` uses `shibuya`. PR #107 fixes the
  build, not the content.
- `jem/utils/esmf_regrid.py` ends with a 90-line `example_usage()` /
  `__main__` block of printed marketing copy.

Recommendation: adopt JCM's documentation layout (`README` + `getting_started`
+ `docs/source/design/*.md` with `myst_parser`), regenerate the quick start
from a *tested* example, and delete `DEVELOPER.md` in favour of the design
docs (this file is placed under `docs/source/design/` to start that).

## 7. Smaller points, by file

- `jem/base/coupler.py`: `get_info` returns
  `{"message": "get_info not provided."}` for components without it;
  fine, but `Coupler.get_info` is only ever consumed by
  `tree_tools.print_tree`. Replace both with `__repr__` (JCM's `Model`
  has one).
- `jem/components/veros_component.py`: sets `veros.runtime_settings` as
  a side effect of import, prints on import, hard-codes `cp_0`,
  `salinity_ref`, ghost-cell count, and clamps SST with
  `jnp.where(sst < 100, 288.15, sst)`. Experimental, but these belong in
  the component's parameters and in a documented "why".
- `jem/utils/bulk_op.py`: `concat_objects`, `mean_leaf` unused.
  `jem/utils/cycles.py`: `evaluate_periodic`, `vmap_evaluate_periodic`
  unused (they are the only `coordax` users).
  `jem/utils/datetime_tools.py`, `jem/utils/domain_grid_tools.py`: unused.
  `jem/utils/idealized_distribution.py`: one 6-line function, copied
  again into `03_long_aquaplanet.py`.
- `jem/components/slab/grid.py::generate_slab_grid_from_ugrid`: unused
  now that the data ships as SCRIP; keep one reader.
- `jem/components/slab/slab_ocean_model/__init__.py` exports
  `OceanForcing`, `OceanState` in `__all__` without importing them.
- `notebooks/.gitignore` and `examples/.gitignore` (PR #106) are empty
  files.

## 8. What to ask of JCM during its config hardening

These are small on the JCM side and remove most of the adapter's fragility:

1. **A public surface-exchange contract, independent of physics package.**
   Something like `predictions.surface_exchange` (or a
   `SurfaceExchange` struct on the physics diagnostics) carrying: net
   downward heat flux into the surface, evaporation, total precipitation,
   `u0/v0` (or the stress directly), per-tile where the package has tiles.
   SPEEDY and ECHAM both compute these; only the key names differ today.
   This is the same request the JCM release note anticipates when it says
   a coupled surface model "now needs them from its own land/ocean
   components".
2. **Public equivalents of `_prepare_initial_dycore_state()` and the
   empty-diagnostics template** (`Model.initial_state()`,
   `Model.empty_diagnostics()` or documented use of `bootstrap_state` +
   `physics.get_empty_data`).
3. **A documented "one coupling step" entry point**:
   `run_from_state_with_carry` already is it; document that
   `save_interval == total_time` yields a length-1 time axis, and provide
   `ModelPredictions.with_context(coords, physics)` so callers do not
   reconstruct it from `_predictions`.
4. **Consistent float dtype in `default_forcing`/`ForcingData.zeros`** so
   the `asfloat64` workaround can go.
5. **`jcm.config` reachable via Hydra `searchpath`** (already true because
   the YAML is package data; just document it as supported).

## 9. Verification performed for this review

- Read every module in `jem/`, all notebooks and scripts, tests, CI, docs,
  and the diffs of PRs #106 and #107; cross-read `jcm/main.py`,
  `jcm/runners.py`, `jcm/model.py`, `jcm/checkpoint.py`, `jcm/config/`,
  `jcm/constants.py`, the SPEEDY surface-flux structs on `main` (v2.0.1)
  and `dev`, and JCM's release notes.
- `ruff check jem/` (ruff 0.15.17): 2 errors; `ruff check .`: 59.
- Duplication figures in §2 were computed from the notebook JSON.
- The JCM-`dev` incompatibility and the unit-test status are recorded in
  §9.1 (filled in from a live run in this environment).

### 9.1 Live checks

Clean venv, `jax-gcm dev` (2.1.0b0) and `jax-esm[dev]` installed editable,
CPU only:

| check | result |
|---|---|
| `pytest tests/unit` | 75 passed |
| `mypy jem/ --ignore-missing-imports` (as CI) | clean, 31 files |
| `ruff check jem/` (ruff 0.15.17) / `ruff check .` | 2 errors / 59 errors |
| JCM adapter, aquaplanet JCM + slab ocean, 2 coupling steps | **fails**: `IndexError: Too many indices: array is 2-dimensional, but 3 were indexed` — `hfluxn` and `evap` are `(96, 48)` on `dev` |
| `SlabLandModel._idealized_land_temperature` at T31 | **wrong axis**: std of the field along longitude at fixed latitude = 5.27 K (should be 0); along latitude at fixed longitude = 6e-5 K (should be the whole profile) |
| `asfloat64` | emits a JAX `UserWarning` ("float64 … will be truncated to float32") on every adapter call with x64 off |

## 10. Suggested order of work

1. **Release blockers, small (days).** Replace `CLAUDE.md` with JCM's
   conventions; fix README/quick-start/tutorial to
   the current API; pin `jcm` and update the adapter to the 2-D surface
   flux layout; fix the land idealised-climatology bug and `Q_flux_file`
   loading; trim dependencies; pin ruff and get `ruff check .` clean;
   decide the license.
2. **Driver layer (one PR, the big usability win).** `jem/main.py`,
   `jem/runners.py`, `jem/config/` composing JCM's groups; chunked run
   loop with health gate, checkpoint and output writer; `logging`. Rewrite
   the examples as overrides / 10-line notebooks. Land PR #106's Veros
   science on top of this rather than as two more drivers.
3. **Component contract.** `Component` protocol, wrapper classes,
   coupler-owned time and coupling timestep, declarative exchanges,
   `Parameters` dataclasses for the slabs, `SlabGrid.from_coords`,
   `jcm.constants`, CF-consistent output names. Add slab/regridder/adapter
   unit tests and one `check_vjp` per component.
4. **JCM-side asks (§8)**, coordinated with the config hardening, then
   remove the private-key access and thread the physics carry.
