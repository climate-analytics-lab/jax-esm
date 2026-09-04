# Changelog

All notable changes to JAX-ESM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and JAX-ESM aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from v1.0.0
onwards. Before v1.0.0 the public API may change in any release; every such
change is listed here.

## [Unreleased] — 1.0.0a0, "the core API contract"

Phase 1 of the [API hardening plan][plan]. It replaces the duck-typed component
interface with an explicit protocol, moves the clock out of the components and
into the coupler, makes component parameters differentiable, and settles the
output conventions. **Every entry below is a breaking change unless it says
otherwise**; the code that has to change is named in each one.

[plan]: https://github.com/climate-analytics-lab/jax-esm/blob/claude/jax-esm-api-review-jv7j7u/docs/source/design/api_hardening_plan.md

### Added

- **`jem.base.component`** — the whole contract, in one module:
  - `Component`, a runtime-checkable `typing.Protocol` requiring `name`,
    `initialize() -> carry` and `step(carry, time) -> (carry, diagnostics)`.
    There is still no base class to inherit from, but a component is now
    *checked*: `Coupler.add_component` raises `TypeError` naming the missing
    member.
  - The optional capabilities `SupportsXarray` (`to_xarray(diagnostics,
    time)`), `SupportsCheckpoint` (`save_state`/`load_state`) and
    `SupportsBind` (`bind(*, coupling_timestep, start_date, calendar)`), each
    detected with `isinstance` at the one place that uses it.
  - `CoupledCarry`, the scanned state of the coupled model: `components`
    (one carry per component) plus `step`, the authoritative coupled step
    counter.
  - `CouplingTime`, what every `step` receives: `step`, `sim_time`, and the
    static `dt`, `year_offset_seconds` and `days_per_year`, with a
    `year_fraction` property and `end_of_step()`.
  - `TimeAxis`, the description of a run's output records that every component
    labels its dataset from.
  - `Exchanger`, the type of the functions that move information between
    components.
- `Coupler.step_function()` — the pure one-step function, previously only
  available inside `Coupler.run`.
- `Coupler.coupling_time(step)`, `Coupler.time_axis(first_step, n)` and the
  `coupling_timestep` / `start_date` / `calendar` / `dt_seconds` /
  `year_offset_seconds` / `days_per_year` properties: the clock, readable.
- `Coupler.__repr__`, which names the components, exchangers, workflow and
  clock. It replaces `get_info()` + `tree_tools.print_tree` in the examples.
- `jem.components.jcm` — the JCM adapter as a package:
  - `JCMComponent(model, *, forcing=None)`, name `"atm"`, a wrapper *object*.
  - `jem.components.jcm.exchange_fields`, the single place JCM's
    package-specific diagnostics are translated into JEM's conventions:
    `SurfaceExchange`, the `speedy()` reader, an `echam()` reader that raises
    `NotImplementedError` naming jax-gcm#754, and `detect()`.
- `jem.components.veros_component.VerosComponent(model)`, name `"ocn"`, the
  same treatment for the Veros ocean.
- A `flax.struct` parameters dataclass per slab model, whose numeric fields are
  pytree leaves and which travels in `carry["params"]`, so `jax.grad` of a
  coupled run with respect to a physical parameter needs no special casing:
  - `SlabOceanParameters`: `relaxation_time=60*86400.0`,
    `mixed_layer_depth_min=40.0`, `mixed_layer_depth_max=60.0`,
    `initial_sst=288.15`, static `forcing_method="none"` and
    `ocean_mask_value=0.0`.
  - `SlabLandParameters`: `depth_soil=1.0`, `depth_lice=5.0`,
    `soil_volumetric_heat_capacity=2.50e6`,
    `land_ice_volumetric_heat_capacity=1.93e6`, `tdland=40*86400.0`,
    `flandmin=1/3`, `land_threshold=0.1`, `snow_depth_to_cover_scale=60.0`,
    `land_ice_albedo_threshold=0.4`, `surface_albedo=0.2`.
  - `SlabSeaiceParameters`: `initial_ice_thickness=0.0`,
    `min_ice_thickness=1e-3`, `ice_fraction_thickness_scale=0.5`, static
    `ocean_mask_value=0.0`.
  - `SlabAtmosphereParameters`: `initial_temperature_base=273.15`,
    `initial_temperature_amplitude=17.0`, `initial_zonal_wind=10.0`,
    `initial_meridional_wind=0.0`.
  Each has a `.default()` classmethod, and each model's `params=` argument
  defaults to it.
- `SlabGrid.from_coords(horizontal, fractional_mask=None, threshold=0.5)`,
  which builds a slab grid from the dinosaur horizontal grid the atmosphere is
  discretized on, and `SlabGrid.from_scrip(...)`.
- `jem.constants.SurfaceConstants` and `jem.constants.set_constants(...)`,
  mirroring `jcm.constants`: a frozen dataclass, a live singleton and a module
  `__getattr__`, so `jem.constants.ocean_density` honours an override made
  after import.
- `TimeAxis.datetimes()` and `TimeAxis.attrs`, the single definition of the
  output time labels (JCM's arithmetic, not just JCM's answer);
  `jem.utils.time.time_coordinate(time_axis)` is now the slab-side call site
  that unpacks them for xarray.
- `jem.base.component.seconds_since_new_year(start_date)` and
  `start_year_fraction(start_date, calendar)`, the shared arithmetic behind
  both `CouplingTime.year_fraction` and the date a slab model samples its
  climatology at in `initialize()`, so the two cannot disagree; and
  `NANOSECONDS_PER_DAY`, the factor JCM's time labels go through.
- `SlabModelBase.bind(...)` (the slab models are `SupportsBind` too) and the
  `SlabModelBase.start_year_fraction` property it sets. `initialize()` samples
  a monthly climatology but receives no clock — the clock lives in the carry,
  which does not exist yet — so the coupler's start date reaches the model the
  same way it reaches JCM and Veros. A model that was never registered with a
  coupler reads 1 January, which is what a bare `model.initialize()` in a test
  or a notebook gets.
- Complete exports: `jem` adds the `Carry` and `Diagnostics` aliases a user
  needs to annotate their own component, and `jem.components` adds the four
  `Slab*Parameters`, `SlabGrid`, `SlabModelBase` and `load_monthly_climatology`
  — a documented run can now be assembled from the packages' own exports.
  `jem` still does not re-export the components: importing the JCM wrapper
  pulls in the whole atmosphere, which should not be the cost of `import jem`.
- `docs/source/design/architecture.md` rewritten against the new API, and a
  `tests/examples` note in `docs/source/developers.rst`.

### Changed

- **"Mapper" is now "exchanger".** `Coupler(components, exchangers=...)`,
  `add_exchanger`, `remove_exchanger`, `Coupler.exchangers`. The rename was
  decided in the #108 review: "mapper" reads as a regridding operation,
  whereas one of these functions may regrid, compute a flux, convert units or
  simply copy a field.
- **An exchanger's signature is `(components, time) -> components`**, not
  `coupled_carry -> coupled_carry`. It receives the mapping of component
  carries (a fresh dict) plus the clock, and must build new carries rather
  than assign into the ones it was handed — the coupler hands out the carries
  of a `lax.scan`. The coupler now compares the pytree structure of the
  carries after every workflow element and raises `RuntimeError` naming the
  element that changed it.
- **`Coupler.__init__` takes the coupled model's clock**:
  `Coupler(components, exchangers=None, *, coupling_timestep, start_date,
  calendar="365_day", workflow=None)`. `coupling_timestep` and `start_date`
  are required. `workflow` moved here from `run()`/`generate_*` and defaults
  to every exchanger (in insertion order) followed by every component, so the
  usual coupling scheme need not be spelled out.
- **`Coupler.initialize()` returns a `CoupledCarry`**, not a plain
  `dict[str, carry]`. The per-component carries are under
  `.components`; rebuild one with `carry.replace(components=...)`.
- **`jem.utils.checkpoints.save_coupled_carry`/`load_coupled_carry` take and
  return a `CoupledCarry`**, not a plain `dict[str, carry]`, and the
  checkpoint directory gains one file, `coupled_step.pkl`, holding the coupled
  step counter. A caller that passed `final_carry.components` passes
  `final_carry`; a caller that rebuilt the carry around the loaded dict uses
  the returned `CoupledCarry` directly. The mapping-only helpers are still
  available as `save_component_carries` / `load_component_carries`, which is
  what the Veros restart writer uses for the picklable half of its carry.
- **`Coupler.generate_trajectory_function(iterations, *, remat=False,
  jit=True)`** takes neither `workflow` nor `checkpoint`/`show_progress`/
  `tqdm_kwargs`, and returns `carry -> (final_carry, diagnostics)`.
  Because the step counter lives in the carry rather than in the `lax.scan`
  index, calling it again on the carry it returned *continues* the run; this
  is what makes a chunked or restarted run keep the right date and season.
- **`coupler.components[name]` is the component object itself.** There is no
  `JEMComponent` wrapper and no `.raw_component` — code that reached through
  the wrapper now uses the object directly (`component.model` for
  `JCMComponent` / `VerosComponent`).
- **Components implement `step(carry, time)`**, not
  `generate_step_function() -> step(carry, step_index)`, and **`to_xarray(
  diagnostics, time)`**, not `predictions_to_xarray(predictions)`.
  `step` receives a `CouplingTime`, not a bare index or a float.
- **Slab constructors lost the clock and gained parameters.** They are now
  - `SlabOceanModel(grid, params=SlabOceanParameters(), *, name="ocn",
    sst_clim_file=None, q_flux_file=None)`
  - `SlabLandModel(grid, params=SlabLandParameters(), *, name="lnd",
    land_clim_file=None, surface_albedo=None)`
  - `SlabSeaiceModel(grid, params=SlabSeaiceParameters(), *, name="ice")`
  - `SlabAtmosphereModel(grid, params=SlabAtmosphereParameters(), *,
    name="atm")`

  `start_datetime`, `timestep` and `calendar` are gone from all four (the
  coupler owns them); every physical tunable moved into the parameters object;
  `SST_clim_file`/`Q_flux_file` are now `sst_clim_file`/`q_flux_file`;
  `mask_value` is now `params.ocean_mask_value`;
  `initialization_sea_surface_temperature` is now `params.initial_sst` and
  `initialization_ice_thickness` is `params.initial_ice_thickness`. The
  component's name is a constructor argument rather than the class name.
- **`sim_time` is gone from every state struct** (`OceanState`,
  `LandState`, `SeaiceState`, `AtmosphereState`): a component holds no clock.
  Every slab carry gains a `"params"` entry alongside `state`/`forcing`/
  `derived`.
- **`forcing_method` values are lowercase** `"none" | "qflux" | "relaxation"`
  (`jem.components.slab.slab_ocean_model.params.FORCING_METHODS`). `None` and
  `"None"` are no longer accepted; an unknown value raises `ValueError`.
- **`SlabOceanModel` refuses `forcing_method="relaxation"` without an
  `sst_clim_file`.** The previous behaviour — silently setting the relaxation
  timescale to infinity and then dereferencing a climatology that was never
  loaded — could not work.
- **Science-visible default change**: without an SST climatology the ocean's
  idealized initial profile is now built on `params.initial_sst = 288.15 K`,
  giving 288–298 K instead of the previous 273–283 K. The constructor has
  always accepted this value but never used it — the base of the profile was
  hard-wired to the freezing point — so wiring it up also moves the default
  aquaplanet start to a sensible one.
- **`SlabLandModel` loads its climatologies by name and requires exactly 12
  monthly records.** The branch that accepted daily data is gone; a file with
  a different record count raises `ValueError` naming the file and the check.
  Fields the file does not carry still fall back to idealized ones.
- **The JCM carry gained `"physics"`**: JCM's cross-step physics carry
  (sub-cycled radiation, prior-step TKE, term-to-term tendencies) is threaded
  through `run_from_state_with_carry` instead of being dropped and rebuilt
  once per coupling interval, which was a silent, systematic error in every
  coupled run. It holds integer and boolean leaves, so it must never be cast
  wholesale to a float dtype.
- **`JCMDerived` gained `evaporation`, `precipitation`, `u0` and `v0`**
  alongside `total_heat_flux`, `total_freshwater_flux` and the opaque
  `physics` passthrough, so an exchanger no longer has to dig the surface wind
  and water fluxes out of the physics dict itself.
- **`JCMComponent.initialize()` does not integrate.** The previous adapter ran
  a whole throwaway coupling step to learn the structure of the diagnostics it
  would later store, which cost a step per run and started the atmosphere one
  coupling interval ahead of the coupler's clock.
- **Output conventions**, so that `xr.merge` of two components' datasets from
  one run aligns instead of producing an outer join:
  - slab datasets use dims `("time", "lon", "lat")` with 1-D degree
    coordinates carrying JCM's own names and values (a curvilinear grid keeps
    dims `("time", "x", "y")` with 2-D auxiliary `lat`/`lon` and a CF
    `coordinates` attribute), replacing the old `latitude2D`/`longitude2D`
    auxiliary coordinates;
  - the `time` coordinate is an absolute `datetime64[ns]` axis, not
    "hours since <start>";
  - a record is labelled with the **end** of the interval it covers, which is
    JCM's convention;
  - state and derived variables keep their plain names and every variable that
    came from a component's *forcing* is written with a `forcing_` prefix
    (`jem.components.slab.base.FORCING_VARIABLE_PREFIX` and the
    `forcing_variable(name)` helper that applies it). Two components
    legitimately hold the same physical field — one produced it, the other
    received it — and without the prefix `xr.merge` of their datasets collides
    on the shared name. The renames are: `SlabAtmosphereModel` and
    `SlabLandModel` write `forcing_total_heat_flux` (was `total_heat_flux`),
    `SlabOceanModel` in Q-flux mode writes `forcing_q_flux` (was `q_flux`), and
    `SlabSeaiceModel` writes `forcing_ice_frazil_melt_energy` (was
    `ice_frazil_melt_energy`). Everything else keeps its name, including the
    ocean's own `total_heat_flux` (a derived quantity — the effective heat flux
    applied to the mixed layer) and `ice_frazil_melt_energy`.
- `Coupler.to_xarray(diagnostics, *, first_step=0)` replaces
  `predictions_to_xarray(predictions)`. Pass `first_step` — the `step` of the
  carry the chunk started from — when writing a chunked run, or every chunk is
  labelled with the first chunk's dates.
- `jem.constants` now holds only what `jcm.constants` does not define, in a
  `SurfaceConstants` singleton. The duplicated values were removed and their
  JCM counterparts now apply, which changes three numbers:
  latent heat of fusion `3.34e5 → 3.33e5 J/kg` (`c.alhf`, which JCM *derives*
  as `alhs - alhc`), dry-air specific heat `1004.0 → 1004.64 J/K/kg`
  (`c.cpd`, the ECHAM-6.3 value JCM's own `rd = akap*cpd` is built on) and the
  solar constant `1367 → 1361 W/m2` (`c.solc`, read by nothing in `jem`).
- `JCMComponent`'s clock-drift check scales with float32 resolution:
  `clock_tolerance_seconds(sim_time)` is one second or eight float32 ulps of
  the elapsed time, whichever is larger, so it neither fires on the rounding of
  a long run's clock nor stops noticing a real disagreement.
- `Coupler` logs at DEBUG instead of printing; nothing under `jem/` prints any
  more (the Veros checkpoint writer and the forcing/topography generator were
  the last two).

### Deprecated

- `jem.components.jcm_component.make_jem_compatible(model, coupling_timestep)`
  and `jem.components.veros_component.make_jem_compatible(model,
  coupling_timestep)` now return a `JCMComponent` / `VerosComponent` and warn.
  The `coupling_timestep` argument is ignored — the coupler supplies it, with
  the start date and calendar, through `bind()`. Unlike the old functions they
  do **not** attach methods to the wrapped model, so anything that called
  `model.initialize()` / `model.generate_step_function()` must call them on
  the returned component instead.

### Removed

- **`jem.base.interface`** (whole module): `resolve_interface`, the
  `__JEM_CUSTOMIZED_MAPPING__` method-remapping hook, `MethodNotFoundError`,
  `MemberNotFoundError`, `MemberTypeNotMatchError` and
  `NumberOfMethodParametersNotMatchError`. A component is now checked against
  the `Component` protocol; an object whose methods have other names is
  adapted by a wrapper class, which is what `JCMComponent` and
  `VerosComponent` are.
- **`jem.base.typing`** (whole module): `JEMComponent`, `MapperFunction`,
  `StepFunction`, `StepFunctionGenerator`, `TrajectoryFunction`,
  `PredictionsToXarrayFunction`, `GetInfoFunction`, `InitializeFunction`,
  `ComponentCarry`, `CoupledCarry`, `SimulationTime`, `Predictions`,
  `Workflow`, `ComponentName`, `Pytree`. The aliases that survive
  (`Carry`, `Diagnostics`, `CoupledCarry`, `Exchanger`) live in
  `jem.base.component`, and `CoupledCarry` is now a struct, not a dict alias.
- `Coupler.run(...)` — build the function and call it:
  `run = coupler.generate_trajectory_function(iterations)`, then
  `final_carry, diagnostics = run(coupler.initialize())`. It returned
  `(initial_carry, final_carry, predictions)`; the initial carry is now
  whatever you passed in.
- `Coupler.generate_step_function()` in its old form (it took a workflow and
  returned a function of `(carry, step)`); `Coupler.step_function()` replaces
  it.
- `Coupler.predictions_to_xarray(...)` → `Coupler.to_xarray(...)`.
- `Coupler.get_info()` → `repr(coupler)`.
- `Coupler.add_mapper` / `remove_mapper` / `.mappers` → `add_exchanger` /
  `remove_exchanger` / `.exchangers`.
- `jem.base.coupler.adhoc_scan` and the `jitted=False` debug path: use
  `generate_trajectory_function(..., jit=False)`, or `jax.disable_jit()`.
- The `verbose`, `show_progress`, `tqdm_kwargs`, `checkpoint` and
  `reuse_last_available_trajectory` parameters, and the `trajectory_holder`
  attribute that backed the last of them. A trajectory function is now built
  once by the caller and reused by calling it again; `remat=` replaces
  `checkpoint=`.
- Component-side `get_info()` and `predictions_to_xarray()` hooks (the
  coupler no longer looks for either name).
- `generate_slab_grid`, the `"JCM::T31"` specification-string DSL it parsed,
  `load_jcm_fractional_mask`, and `generate_slab_grid_from_scrip` →
  `SlabGrid.from_coords` and `SlabGrid.from_scrip`. The atmosphere's own grid
  object is the single source of truth for the grid a coupled run uses, and
  the land fraction comes from `jcm.terrain.TerrainData.fmask`, which is
  already on `SlabGrid`'s `(n_lon, n_lat)` layout.
- `jem.utils.tree_tools` (whole module): `print_tree`, `print_dict_tree`,
  `tree_to_dict`. It existed to render `Coupler.get_info()`; `repr(coupler)`
  replaced both.
- The `jem.constants` values `jcm.constants` already owns — `g0`,
  `stephan_boltzmann_const`, `freezing_point_K`, `ice_melting_point_K`,
  `ice_density`, `ice_latent_heat_fusion`,
  `atmosphere_specific_heat_capacity_at_constant_pressure` and `solar_const`
  — use `c.grav`, `c.sbc`, `c.tmelt`, `c.rhoi`, `c.alhf`, `c.cpd`, `c.solc`.
  `default_mld_min`/`default_mld_max` and
  `default_land_depth_min`/`default_land_depth_max` went with no replacement:
  a default for a component parameter belongs to that component's parameters
  dataclass.
- `jem.utils.bulk_op` (whole module): `stack_objects` and
  `unwrap_leading_dims`. They existed because the old coupler stacked each
  step's predictions by hand; `lax.scan` does it, so nothing called them.
- `jem.utils.time.TIME_ATTRS` → `TimeAxis.attrs`.
- The dependency `typeguard`; nothing imports it now that `jem.base.typing`
  is gone.

### Fixed

- The seasonal cycle no longer restarts at the beginning of each chunk of a
  chunked run: the coupled step counter is part of the carry, so
  `year_fraction` continues across trajectory calls and checkpoint restarts.
  It is also computed in exact integer arithmetic modulo the steps in a year
  whenever the coupling step divides the year, so a float32 `sim_time` cannot
  quantise the annual cycle away in a century-long run.
- Two components can no longer disagree about the date: there is one clock,
  and `bind()` refuses a coupling timestep that is not a whole multiple of a
  component's internal timestep, or (for JCM) a start date or calendar that
  differs from the coupler's. `JCMComponent.step` additionally reports at
  ERROR if the dycore state's own `sim_time` has drifted from the coupler's,
  which can only happen if the carry came from a different run.
- `Coupler.step_function()` snapshots the components and exchangers when it is
  called, so registering a component afterwards cannot silently change an
  already-compiled step.
- The coupled step never mutates its input carry; it rebuilds the carries dict
  and returns a new `CoupledCarry`.
- A run resumed from a checkpoint no longer restarts its seasonal cycle:
  `save_coupled_carry` writes the coupled step counter and `load_coupled_carry`
  restores it. A checkpoint written without one is refused with a `ValueError`
  naming the missing file, rather than resuming at step 0 or having its step
  guessed from a batch index — a guess that is only right while every batch has
  the same length.

### Known gaps

- The land model's ice-sheet branch is never reached in practice: nothing wires
  a real surface albedo into `SlabLandModel`, so `params.surface_albedo = 0.2`
  applies everywhere and every land cell is soil. Tracked as jax-esm#109 and
  cross-referenced from the model's docstring.
- The ECHAM surface exchange is not implemented: `exchange_fields.echam()`
  raises `NotImplementedError` naming jax-gcm#754. Coupled runs need SPEEDY
  physics until that lands.

## [0.2.0] - release blockers and repository health

Phase 0: what `jem.__version__` reports today. Not yet tagged.

### Added

- The monthly-climatology loader written for `SlabOceanModel` is public on the
  slab base module as `jem.components.slab.base.load_monthly_climatology`, so
  the other slab components can read their boundary conditions through the
  same name-based, grid-checked path.
- `CHANGELOG.md` (this file). Every pre-1.0 API removal or rename is recorded
  here rather than only in a commit message.

### Changed

- The version is single-sourced from `jem.__version__` (setuptools reads it
  via `[tool.setuptools.dynamic]`), so the package attribute and the installed
  distribution metadata can no longer disagree. Bumped to `0.2.0`.

- `pyproject.toml` now declares only the packages `jem` actually imports at
  runtime. Plotting (`matplotlib`, `cartopy`), documentation (`sphinx`,
  `shibuya`, `nbsphinx`, `myst-parser`) and the Veros ocean moved to the
  `plot`, `docs` and `veros` extras, so a model-running install no longer
  pulls a plotting or documentation stack.
- The `jcm` requirement is now `jcm>=2.1.0b0`, the first jax-gcm release with
  the 2-D surface-flux layout the JCM adapter is being rewritten against.
  **That release is not on PyPI yet**: the floor is currently satisfied only
  by an editable install of a jax-gcm `dev` checkout, which is what CI does
  (`pip install -e ./jax-gcm` before `pip install -e ".[dev]"`).
- CI (`.github/workflows/tests.yml`) now lints the *whole* repository with the
  pinned `ruff==0.15.17` (it previously ran an unpinned ruff over `jem/`
  only), runs `mypy jem/ --ignore-missing-imports` in the same job, checks out
  jax-gcm at `dev` and installs it editable *before* `pip install -e ".[dev]"`
  so `jcm>=2.1.0b0` resolves, and drops the jax-gcm `PYTHONPATH` export.
  The test matrix is `ubuntu-latest` x Python 3.11/3.12 — macOS and 3.13 were
  dropped (**maintainer decision, defaulted**: macOS added run time for a
  pure-Python package with no platform-specific code, and 3.13 is ahead of
  what the JAX/Veros stack is tested against). Notebook and `run.sh` example
  tests moved to their own job that runs on pull requests only, and the
  Codecov upload condition moved from Python 3.10 (which the matrix never
  contained) to 3.11.
- The `veros` extra is deliberately empty: the coupler needs a *jittable* fork
  of Veros that is not published on PyPI and must be installed from git (see
  the README).

### Fixed

- `jem.components.veros_component` now selects the JAX backend through a
  public `configure_veros_runtime()` (still run at import time, because the
  jittable Veros fork locks `runtime_settings` as soon as `veros.core` is
  imported) and raises a clear `RuntimeError` naming the import-order fix
  when Veros was already bound to another backend, instead of failing later
  inside `make_jem_compatible`. It no longer prints while doing so.
- `jem.utils.cycles.evaluate_periodic` mis-weighted points in the wrap-around
  interval (between the last and the first tick) when `x` lay above the last
  tick: the distance from the left tick was computed as `x + 1 - t0` there,
  over-counting by a full cycle. It now uses `(x - t0) mod 1`. The function
  also accepts traced indices, so `vmap_evaluate_periodic` works.
- `jem.components.slab.slab_ocean_model.__all__` listed `OceanForcing` and
  `OceanState` without importing them, so `from ... import *` raised
  `AttributeError`. Both are now imported.

### Removed

- Unused code, none of which had a caller anywhere in `jem`, `tests`,
  `examples` or `docs`:
  - `jem.utils.datetime_tools` (whole module).
  - `jem.utils.domain_grid_tools` (whole module).
  - `jem.utils.bulk_op.concat_objects` and `mean_leaf`.
  - `jem.utils.esmf_regrid.create_regridder_pair`,
    `create_regridder_from_xarray`, `example_usage` and the module's
    `__main__` demo block.
  - `jem.components.slab.grid.generate_slab_grid_from_ugrid` and its private
    helpers `_reshape_ugrid_face_field` / `_load_ugrid_fractional_mask`. The
    SCRIP reader (`generate_slab_grid_from_scrip`) is unaffected.
- Dependencies `dataclasses-json`, `typing-extensions`, `jax_tqdm` and `tqdm`:
  none of them are imported by `jem` any more. `coordax` is no longer a
  runtime dependency either: its only user, `jem.utils.cycles.evaluate_periodic`
  (kept at the maintainers' request although currently unused), imports it
  lazily, so it is needed only when that function is called (it is in the
  `dev` extra so the function stays tested).
- **Progress bars.** `Coupler.run`, `Coupler.generate_trajectory_function` and
  `Coupler.generate_step_function` no longer display a `tqdm` progress bar.
  Their `show_progress` and `tqdm_kwargs` parameters are still accepted so
  existing callers keep working, but are deprecated and ignored; they will be
  dropped in a later release once per-chunk logging replaces them.
- The dead, string-quoted `sys.modules` stub-injection block at the top of
  `tests/unit/test_coupler.py` (~70 lines). The tests have always run against
  the real dependencies.
- `[tool.black]` and `[tool.isort]` configuration: `ruff` (pinned to 0.15.17,
  matching jax-gcm) is the only linter.
