# JAX-ESM API hardening plan (implementation spec, v2)

Companion to `api_and_usability_review.md`. That document says *what is
wrong*; this one says *exactly what to build*, in an order that keeps `main`
releasable after every PR. It is written so that a reviewer can check each
task against the review and an implementer can execute it without re-deriving
design decisions. Where a decision belongs to the maintainer it is marked
**DECISION** and the plan states the default that applies if no answer is
given.

v2 incorporates the Codex review of v1: the core API contract now lands
*before* the driver and examples; the coupled clock is persistent across
chunks; the coupler owns the whole coupled-model definition; the JCM
dependency is gated on a pinned revision; and JAX-ESM gets its own coupled
`experiment` group. Everything below was checked against
`jax-esm main @ a9c4f8a`, `jax-gcm dev @ 226172e4` and jax-gcm PR #750
(`45d74c7e`) unless stated otherwise.

---

## 0. Ground rules for every PR

- Branch from `main`, one PR per numbered task (a task may contain several
  commits). Task numbers below are the PR titles: `T1.2: …`.
- Gate before pushing (same as jax-gcm):
  ```bash
  ruff check .                                   # must be clean (ruff==0.15.17)
  JAX_PLATFORMS=cpu pytest tests -q -m "not slow"  # must pass
  JAX_PLATFORMS=cpu mypy jem/ --ignore-missing-imports
  ```
- **Python is the primary interface; config is a thin wrapper.** `Coupler`,
  the component classes, `Exchange` and `jem.run_chunked()` are the API and
  carry every default. `jem/config/*.yaml` may contain only *wiring*: a
  `_target_`, required inputs (`???`), and the non-default choices that
  define a named configuration. **Never write a physics parameter or its
  default value in YAML** — `relaxation_time`, `depth_soil`,
  `ice_fraction_thickness_scale`, ... exist once, as Python defaults, and are
  overridden from the CLI with Hydra's append syntax
  (`+ocean.params.relaxation_time=2592000`), the convention JCM's
  `physics/speedy.yaml` already documents. `runners.py` contains no
  per-component code: every node is built with `hydra.utils.instantiate`,
  so a YAML key *is* a constructor keyword and cannot drift from it.
- **Do not** add `print(...)` calls anywhere under `jem/`, `jem/main.py`
  included; use `logging.getLogger(__name__)`.
- **Do not** add `argparse` scripts or `run.sh` files. Runnable
  configurations are Hydra overrides or `experiment` files (Phase 2).
- **Do not** cast pytrees to a dtype wholesale (`asfloat64`-style); fix the
  leaf that is wrong.
- **Do not** read JCM private attributes (`_prepare_initial_dycore_state`,
  `_predictions`, `_final_*`) outside `jem/components/jcm/`; inside it, each
  such read is one helper function with a `TODO(jax-gcm#N)` naming the issue
  from §Phase 5 that will remove it.
- Pre-1.0 API removals are allowed but are *communicated*: every task that
  removes or renames a public name adds a line to `CHANGELOG.md` (created in
  T0.6). Phase 0 is "no behavioural change to the coupling core", not "no
  API change".
- Docstrings: NumPy style; comments explain *why* (jax-gcm `CLAUDE.md`).
- Every task lists its tests. A task is not done until they exist and pass.
- Commit messages: imperative subject ≤ 72 chars, body explains why.

## Dependency order

```
Phase 0  T0.1 CLAUDE.md · T0.2 adapter vs jcm 2.1 · T0.3 land bug · T0.4 qflux · T0.5 docs · T0.6 lint/deps/CI · T0.7 license
            (independent; land in any order; each ≤ 1 day)
Phase 1  T1.1 Component protocol ─► T1.2 JCMComponent + physics carry ─► T1.3 coupler owns time/clock ─► T1.4 params ─► T1.5 grid ─► T1.6 output names/constants ─► T1.7 coupler cleanup
Phase 2  T2.0 jcm dependency gate ─► T2.1 config + experiment group ─► T2.2 runners/driver ─► T2.3 exchange/output/checkpoint
Phase 3  T3.1 examples rewritten once, against the final API ─► T3.2 docs
Phase 4  T4.1 residual cleanup and API freeze
Phase 5  jax-gcm asks (issues filed at the start of Phase 1; T1.2's TODOs point at them)
```
Phase 0 makes `main` releasable (v0.2). Phase 1 is the API contract
(v1.0.0a). Phase 2 is the usability release (v1.0.0b). Phase 4 is the freeze
(v1.0.0).

## Phase 0 — immediate correctness and repository health

### T0.1 Replace `CLAUDE.md` with jax-gcm's conventions

**Files:** `CLAUDE.md` (rewrite), `DEVELOPER.md` (delete),
`docs/source/design/architecture.md` (new, ≤ 150 lines).

**Change:** copy the structure of `jax-gcm/CLAUDE.md` and adapt: project
overview (differentiable coupler; `jem` package), "No bespoke run scripts —
configurations go through Hydra" (verbatim rule from jax-gcm),
"Documentation lives with the change" (design docs under
`docs/source/design/`), the gate from §0, and the JAX conventions section
(pure functions, `tree_math.struct` state, `flax.struct.dataclass`
parameters, no Python `if` on traced values). Remove the argparse /
bash-script mandates and the "required packages" list. Do **not** yet
document output dimension names (they change in T1.6; that task updates
`CLAUDE.md`). `DEVELOPER.md`'s still-true content (carry structure,
workflow, adapter carry layout) moves to `architecture.md`.

**Acceptance:** `grep -i argparse CLAUDE.md` returns nothing;
`architecture.md` is in `docs/source/design.rst`'s toctree.

### T0.2 Fix the JCM adapter for jax-gcm ≥ 2.1 surface-flux layout

**Files:** `jem/components/jcm_component.py`, `pyproject.toml`,
`.github/workflows/tests.yml`, `tests/unit/test_jcm_component.py` (new).

**Change (exact):**
1. In `generate_step_function.step_function`, replace
   ```python
   land_heat_flux  = - physics_no_time_dimension["_surface_flux"].hfluxn[:, :, 0]
   ocean_heat_flux = - physics_no_time_dimension["_surface_flux"].hfluxn[:, :, 1]
   total_heat_flux = - physics_no_time_dimension["_surface_flux"].hfluxn[:, :, 2]
   evaporation     =   physics_no_time_dimension["_surface_flux"].evap[:, :, 2]
   ```
   with
   ```python
   sf = physics_no_time_dimension["_surface_flux"]
   total_heat_flux = -sf.hfluxn          # (ix, il); jcm publishes the grid mean, downward positive
   evaporation = sf.evap                 # (ix, il), g m-2 s-1, upward positive
   ```
   and drop `land_heat_flux` / `ocean_heat_flux` from `JCMDerived` (jcm no
   longer publishes per-tile fluxes; jax-gcm release notes #645). Grep
   `examples/` and `docs/` for `land_heat_flux|ocean_heat_flux` and remove
   the references (no mapper uses them).
2. Delete `asfloat64` and both call sites. Verified on jax-gcm dev:
   `default_forcing(...)` and the initial dycore state contain only float
   leaves, so the cast is unnecessary; the physics carry contains `int32`
   and `bool` leaves that must *not* be cast (T1.2 threads it).
3. `pyproject.toml`: `"jcm>=2.1.0b0"` (**DECISION**: the jax-gcm release
   that ships the 2-D flux layout; default `>=2.1.0b0`, tightened by T2.0).
4. CI: check out `jax-gcm` at the revision T2.0 will pin (until then:
   `dev`) and install it with `pip install -e ./jax-gcm` *before*
   `pip install -e ".[dev]"`; delete the `PYTHONPATH=` export.

**Tests (`tests/unit/test_jcm_component.py`):**
- `test_initialize_carry_structure`: T31 aquaplanet `Model`, adapter,
  `initialize()` returns keys `state/derived/forcing`;
  `derived.total_heat_flux.shape == (96, 48)`.
- `test_two_coupling_steps_run`: `Coupler` with the adapter +
  `SlabOceanModel`, `run(iterations=2)` completes; final
  `total_heat_flux` finite. `@pytest.mark.slow` if > 60 s on CI CPU.
- `test_no_float64_warning`: under `warnings.catch_warnings(record=True)`
  no `UserWarning` mentioning `float64`.

**Acceptance:** the review's §9.1 smoke test passes against jax-gcm dev.

### T0.3 Fix `SlabLandModel._idealized_land_temperature` axis bug

**Files:** `jem/components/slab/slab_land_model/slab_land_model.py`,
`tests/unit/test_slab_land_model.py` (new).

**Change:** the method receives `shape = grid.shape = (n_lon, n_lat)` but
unpacks `nlat, nlon = shape`. Rewrite it to take the grid, not a shape:

```python
def _idealized_land_temperature(self) -> jnp.ndarray:
    """Idealised monthly land-temperature climatology, shape (n_lon, n_lat, 12).

    Latitude dependence comes from the grid's own 2-D latitude field so the
    axis order can never be confused (the previous version assumed a
    (n_lat, n_lon) shape and produced a longitude-varying field on JCM's
    (n_lon, n_lat) grid).
    """
    lat = self.grid.latitude_radian                      # (n_lon, n_lat)
    months = jnp.arange(12)
    base_T = 273.15 + 25.0 * jnp.cos(lat)                # (n_lon, n_lat)
    seasonal_amp = 15.0 * jnp.sin(jnp.abs(lat)) ** 2
    phase = 2 * jnp.pi * (months - 2) / 12.0             # peak in March
    return base_T[..., None] + seasonal_amp[..., None] * jnp.cos(phase)[None, None, :]
```
Update the two call sites in `initialize()` (drop the `jnp.transpose`).

**Tests:** `test_idealized_climatology_varies_with_latitude_only`: T31 grid,
`std(T[:, j, 0]) == 0` for every `j`, `std(T[i, :, 0]) > 1`.
`test_land_step_runs_and_is_finite`: 3 steps with zero heat flux;
temperature finite and within 150–350 K.

### T0.4 Load `Q_flux_file` in `SlabOceanModel`, by named dimensions

**Files:** `slab_ocean_model.py`, `tests/unit/test_slab_ocean_model.py` (new).

**Change:** add one loader used for *both* `SST_clim_file` and
`Q_flux_file` (the SST loader today does a bare `jnp.array(ds["sst"])` and
accepts any array whose shape happens to fit):
```python
def _load_monthly_climatology(path, var: str, grid: SlabGrid) -> jnp.ndarray:
    """Return (n_lon, n_lat, 12) for variable `var`, transposed BY NAME from whatever
    order the file uses (dims must be named lon/lat/time or longitude/latitude/time),
    with 12 time records, and with the file's lon/lat coordinates matching the grid
    to 1e-4 degrees. Raises ValueError naming the file and the failed check."""
```
In `initialize()`, `forcing_method == "Qflux"` with `Q_flux_file` set loads
`var="qflux"` (**DECISION**: variable name; default `qflux`) into
`OceanForcing.q_flux`; `SST_clim_file` goes through the same loader with
`var="sst"`.

**Tests:** `test_qflux_file_is_loaded` (write a `(time, lat, lon)`-ordered
12-month file to `tmp_path`; loaded array equals it after transposition),
`test_climatology_rejects_wrong_coords` (shift lon by 1°; `ValueError`
names the file), `test_qflux_zero_without_file`,
`test_relaxation_matches_analytic` (uniform grid, one step, closed-form
Euler-backward update), `test_freeze_melt_energy_sign`.

### T0.5 Fix the README and docs quick start (stopgap)

**Files:** `README.md`, `docs/source/quick_start.rst`,
`docs/source/tutorial.rst`, `docs/source/developers.rst`,
`docs/source/index.rst`, `docs/requirements.txt`.

**Change:** replace the `BasicMapper` quick start with the code of
`examples/01_basic/01_aquaplanet.ipynb` cells 1–5 (function mapper,
`generate_slab_grid("JCM::T31")`, `Coupler`, `run`, `predictions_to_xarray`).
This is a stopgap that T3.2 replaces with the final tested quick start.
Remove `tutorial.rst` sections that reference `_prepare_initial_modal_state`,
`predictions.physics.surface_flux`, `land_model_active`. Replace the
placeholder text in `index.rst`; `[jcm,plot]` → `[dev]` in
`developers.rst`; `docs/requirements.txt` lists `shibuya` and drops
`sphinx-rtd-theme`.

**Tests:** `tests/unit/test_readme_quickstart.py` extracts the first
```` ```python ```` block from `README.md` with a regex and `exec`s it with
`iterations` patched to 1 (`@pytest.mark.slow`). This test survives T3.2
and is what keeps the quick start honest.

### T0.6 Lint, dependencies, CI hygiene, changelog

**Files:** `pyproject.toml`, `.github/workflows/tests.yml`, `CHANGELOG.md`
(new), `jem/__init__.py`, offending sources.

**Change:**
1. `pyproject.toml` dependencies → exactly:
   `jax, jaxlib, numpy, xarray, netcdf4, tree_math, jax_datetime, jcm>=…,
   flax, hydra-core, omegaconf`. Extras: `plot = [matplotlib, cartopy]`,
   `veros = []` (**DECISION**: the jittable fork is not on PyPI; default is
   an empty extra plus the git install documented in the README),
   `docs = [sphinx, shibuya, nbsphinx, myst-parser]`,
   `dev = [pytest, pytest-cov, pytest-xdist, mypy, ruff==0.15.17, nbformat, nbconvert]`.
   Remove `dataclasses-json`, `jax_tqdm`, `tqdm` (progress becomes one
   `logging` line per chunk), `coordax`, `sphinx`, `shibuya`, `matplotlib`,
   `cartopy` from the main list. `typeguard` stays until T1.1 deletes its
   only user. Delete `[tool.black]`, `[tool.isort]`. Add `[tool.ruff.lint]`
   copied from jax-gcm (`extend-select = ["D"]` plus its ignore list) and
   `per-file-ignores` for `*.ipynb`.
2. Single-source the version: `version = {attr = "jem.__version__"}` under
   `[tool.setuptools.dynamic]`; bump `jem.__version__` to `0.2.0`.
3. Delete unused code: `jem/utils/datetime_tools.py`,
   `jem/utils/domain_grid_tools.py`, `evaluate_periodic` and
   `vmap_evaluate_periodic` in `cycles.py`, `concat_objects` and `mean_leaf`
   in `bulk_op.py`, `generate_slab_grid_from_ugrid` and its two helpers in
   `grid.py`, `example_usage` and the `__main__` block in `esmf_regrid.py`,
   `create_regridder_pair`, `create_regridder_from_xarray`. Record each in
   `CHANGELOG.md` under "Removed".
4. Fix the two E402s in `veros_component.py` by moving the
   `runtime_settings` mutation into `_configure_veros_runtime()` called at
   the top of `make_jem_compatible` (import-time side effects go away too).
5. `ruff check .` → clean (fix the 59 findings in examples/tests;
   `ruff check --fix` for the auto-fixable ones and review the diff).
6. CI: lint on Python 3.11 with `ruff==0.15.17` and `ruff check .`; test
   matrix `ubuntu-latest × {3.11, 3.12}` (**DECISION**: drop macOS and
   3.13; default drop); the notebook/example job runs on PRs only. Codecov
   condition → `3.11`.

**Acceptance:** `ruff check .` clean; `pip install -e .` in a fresh venv pulls
no plotting or docs packages (`pip show cartopy` fails); `python -c
"import jem, importlib.metadata as m; assert jem.__version__ == m.version('jax-esm')"`.

### T0.7 License (**DECISION**)

MIT today, jax-gcm is Apache-2.0. Default if unanswered: keep MIT. Either
way: the repository root must contain the matching `LICENSE` file, the
`license` field and classifier in `pyproject.toml` must agree with it, and
`MANIFEST.in`/`sdist` must include it. One-commit PR.

## Decision log

Decisions taken after this plan was written, in review of the phase PRs.
They override the text below where they differ.

- **Phase 0 landed as one PR** (#108, merged 2026-09-04) rather than one PR per
  task; each later phase is also one PR per phase, at the maintainer's
  request, so a reviewer sees the whole contract change together.
- **"mapper" → "exchanger"** (#108 review, meteorologytoday; confirmed by the
  maintainer). "mapper" reads as regridding, whereas the function may regrid,
  compute fluxes, convert units or copy a field; the defining property is that
  it is the one place information crosses a component boundary. The agent
  noun keeps the registry symmetrical with "component": `exchangers=`,
  `Coupler.add_exchanger`, the `Exchanger` type
  `Callable[[dict[str, Carry], CouplingTime], dict[str, Carry]]` (an exchanger
  receives the clock so lagged or ramped coupling needs no state of its own).
  Applied in Phase 1 (T1.1/T1.3); no aliases for the old names (pre-1.0), the
  rename is recorded in `CHANGELOG.md`. Wherever T2.x below says
  `jem/exchange.py`, `default_exchanges` or `Exchange`, read
  `jem/exchangers.py`, `default_exchangers`, `Exchanger`.
- **`jem/base/component.py` is written first** (Phase 1 commit 26b7f63) and the
  coupler, the JCM wrapper and the slab models are built on it in parallel;
  `CoupledCarry`, `CouplingTime` (with a `year_fraction` property) and
  `TimeAxis` live there rather than in `coupler.py`.
- **`evaluate_periodic` is kept** (maintainer request in #108) with a lazy
  `coordax` import; T0.6's deletion list no longer includes it.
- **Veros adapter** is converted to the `Component` protocol in Phase 1
  (with T1.2), not left on the old monkey-patching path, so the experimental
  Veros examples keep running in the examples CI job.

## Phase 1 — core API contract

### T1.1 `Component` protocol; delete `resolve_interface`

**Files:** `jem/base/component.py` (new), `jem/base/coupler.py`,
delete `jem/base/interface.py`, `jem/base/typing.py` (aliases move to
`component.py`), `tests/unit/test_interface.py` (delete),
`tests/unit/test_coupler.py` (trim to coupler behaviour), `pyproject.toml`
(drop `typeguard`).

```python
from typing import Protocol, runtime_checkable

Carry = Any            # per-component pytree
Diagnostics = Any      # per-component per-step output pytree

@runtime_checkable
class Component(Protocol):
    name: str
    def initialize(self) -> Carry: ...
    def step(self, carry: Carry, time: "CouplingTime") -> tuple[Carry, Diagnostics]: ...

@runtime_checkable
class SupportsXarray(Protocol):          # optional capability
    def to_xarray(self, diagnostics: Diagnostics, time: "TimeAxis") -> xr.Dataset: ...

@runtime_checkable
class SupportsCheckpoint(Protocol):      # optional capability (Veros)
    def save_state(self, carry: Carry, directory: Path) -> None: ...
    def load_state(self, directory: Path) -> Carry: ...

@runtime_checkable
class SupportsBind(Protocol):            # optional: receive the coupler's clock at registration
    def bind(self, *, coupling_timestep: jdt.Timedelta, start_date: jdt.Datetime, calendar: str) -> None: ...
```
`Coupler.add_component(name, component)` asserts `isinstance(component,
Component)` (runtime-checkable protocol → attribute presence) and stores
the object; `JEMComponent` and `raw_component` go away —
`coupler.components[name]` *is* the object. Optional capabilities are
checked with `isinstance(c, SupportsXarray)` at the use site, never
`hasattr`.

**Tests:** `test_component_protocol_rejects_missing_step`,
`test_optional_capabilities_detected`.

### T1.2 `JCMComponent` wrapper; thread the physics carry

**Files:** `jem/components/jcm/{__init__,component,exchange_fields}.py`
(new); `jem/components/jcm_component.py` keeps only a deprecated
`make_jem_compatible` that returns `JCMComponent(model, …)` and warns.

```python
class JCMComponent:
    name = "atm"
    def __init__(self, model: Model, *, forcing: ForcingData | None = None):
        self.model = model
        self.forcing = forcing if forcing is not None else default_forcing(model.coords.horizontal)
        self._days = None                 # set by bind()
    def bind(self, *, coupling_timestep, start_date, calendar):
        # check model.dt_si divides coupling_timestep (existing code); check model.start_date == start_date
        # and model.calendar == calendar (raise ValueError naming both otherwise); self._days = days(coupling_timestep)
    def initialize(self) -> dict:
        self.model.bootstrap_state()                      # public jcm API
        physics_carry = _physics_carry(self.model)        # TODO(jax-gcm#<T5.2>): reads model._final_physics_state
        empty = self.model.physics.get_empty_data(self.model.coords)   # diagnostics template, no integration
        return {"state": _dycore_state(self.model), "physics": physics_carry,
                "derived": JCMDerived.zeros(nodal_shape, physics=empty), "forcing": self.forcing}
    def step(self, carry, time):
        # Clock consistency: the dycore state carries its own sim_time (seconds). It must equal the
        # coupler's; a mismatch means a checkpoint from a different run or a bind() error.
        checkify-free version: jax.debug.callback that logs at ERROR if |state.sim_time - time.sim_time| > 1 s
        state, physics_carry, preds = self.model.run_from_state_with_carry(
            carry["state"], carry["forcing"], save_interval=self._days, total_time=self._days,
            output_averages=True, initial_physics_state=carry["physics"])
        diag = jax.tree.map(lambda x: x[0], preds.physics)      # strip the length-1 save axis
        ex = exchange_fields.detect(diag)(diag)                  # SurfaceExchange
        return ({"state": state, "physics": physics_carry,
                 "derived": JCMDerived(diag, **ex._asdict()), "forcing": carry["forcing"]},
                preds)
    def to_xarray(self, preds, time):
        return _with_context(preds, self.model).to_xarray()      # TODO(jax-gcm#<T5.3>): reads preds._predictions
```
`exchange_fields.py`:
```python
class SurfaceExchange(NamedTuple):
    net_heat_flux: Array          # W m-2, UPWARD positive (sign flip of jcm's downward hfluxn)
    evaporation: Array            # kg m-2 s-1, upward
    precipitation: Array          # kg m-2 s-1, downward
    u0: Array; v0: Array          # near-surface wind, m s-1
def speedy(diag) -> SurfaceExchange   # _surface_flux.{hfluxn,evap,u0,v0}, _convection.precnv, _condensation.precls; g→kg /1000
def echam(diag) -> SurfaceExchange    # raise NotImplementedError naming jax-gcm issue T5.1 until it lands
def detect(diag)                      # "_surface_flux" in diag → speedy; "echam_surface" in diag → echam; else KeyError listing keys
```
Verified on jax-gcm dev: the physics carry treedef is identical before and
after a step, so it scans; it contains `int32`/`bool` leaves — never cast
it. `total_freshwater_flux = evaporation - precipitation` stays as a derived
field for the existing exchanges.

**Tests:** `test_physics_carry_is_threaded` (two steps; returned
`carry["physics"]` differs from the initial one; a 2-step run equals two
1-step runs threaded by hand to 1e-6), `test_initialize_does_not_integrate`
(spy on `run_from_state_with_carry`; `initialize()` must not call it),
`test_speedy_exchange_shapes_and_signs` (downward `hfluxn` of +10 gives
`net_heat_flux == -10`), `test_echam_not_implemented_names_issue`,
`test_bind_rejects_mismatched_start_date`.

### T1.3 The coupler owns the coupled-model definition and a persistent clock

**Files:** `jem/base/coupler.py`, `jem/base/component.py`, all slab models.

```python
@flax.struct.dataclass
class CoupledCarry:
    components: dict[str, Carry]
    step: jax.Array                     # int32 scalar; authoritative coupled step counter

@flax.struct.dataclass
class CouplingTime:                      # what every component.step receives
    step: jax.Array                     # int32, from CoupledCarry.step
    sim_time: jax.Array                 # float seconds since start_date = step * dt
    dt: float = struct.field(pytree_node=False)                    # coupling timestep, seconds
    year_offset_seconds: float = struct.field(pytree_node=False)   # seconds from Jan 1 of start year to start_date
    days_per_year: float = struct.field(pytree_node=False)         # from jcm.date.days_per_year(calendar)

class Coupler:
    def __init__(self, components: dict[str, Component], exchange: Callable | None = None, *,
                 coupling_timestep: jdt.Timedelta, start_date: jdt.Datetime, calendar: str = "365_day",
                 workflow: Sequence[str] | None = None):
        # workflow None -> ["exchange", *components] (exchange first) — jem.exchange.default_workflow
        # every component that SupportsBind gets bind(...) called here
    def initialize(self) -> CoupledCarry            # step = 0
    def step_function(self) -> Callable[[CoupledCarry], tuple[CoupledCarry, dict[str, Diagnostics]]]
        # builds CouplingTime from carry.step, runs the workflow, returns carry.replace(step=carry.step + 1)
        # immutable: dict(carry.components, **{name: new}) and .replace(); never mutates its input
    def generate_trajectory_function(self, iterations: int, *, remat: bool = False, jit: bool = True)
        # lax.scan over iterations; the scan index is NOT the clock — CoupledCarry.step is
    def time_axis(self, first_step: int, n: int) -> TimeAxis   # datetimes for output, from start_date + step*dt
```
Slab models lose `sim_time` from `*State`, and lose the `start_datetime`,
`timestep`, `calendar` constructor arguments; `SlabModelBase._year_fraction`
becomes `(time.sim_time + time.year_offset_seconds) / (86400 * time.days_per_year) % 1`.
The seasonal cycle therefore continues correctly across chunk boundaries
and checkpoint restarts, because `CoupledCarry.step` is part of the scanned
and checkpointed state (T2.3 stores it).

**Tests:** `test_clock_persists_across_trajectory_calls` (two 5-step
trajectory calls give `carry.step == 10` and `sim_time == 10*dt`),
`test_components_share_clock` (two slabs report the same `sim_time`),
`test_year_fraction_wraps`, `test_step_does_not_mutate_input`,
`test_continuous_equals_chunked` (10 steps in one call vs 2×5 vs 5×2 on a
toy coupler: final carry equal to 1e-12; repeated with the JCM adapter at
1e-6, `@pytest.mark.slow`).

### T1.4 Slab parameters as `flax.struct.dataclass`

**Files:** each slab model; new `params.py` next to each.
```python
@struct.dataclass
class SlabOceanParameters:
    relaxation_time: jnp.ndarray = 60 * 86400.0
    mixed_layer_depth_min: jnp.ndarray = 40.0
    mixed_layer_depth_max: jnp.ndarray = 60.0
    initial_sst: jnp.ndarray = 288.15
    forcing_method: str = struct.field(pytree_node=False, default="none")   # none | qflux | relaxation
    @classmethod
    def default(cls): return cls()
```
(`SlabLandParameters`: `depth_soil, depth_lice, tdland, flandmin, land_threshold`;
`SlabSeaiceParameters`: `initial_ice_thickness, min_ice_thickness, ice_fraction_thickness_scale`.)
Constructors become `SlabOceanModel(grid, params=SlabOceanParameters(), *, sst_clim_file=None, q_flux_file=None)`.
`initialize()` becomes pure w.r.t. `self`: climatology arrays are loaded in
`__init__` (data, not state); **no** `cd_factor`/`time_factor`/`rhcapl`/
`cdland` attributes — compute them inside `step` from `params`. `params`
travels in the carry as `carry["params"]` so `jax.grad` w.r.t. it works
through `Coupler` without special casing (**DECISION**: carry vs closure;
default carry).

**Tests:** `test_grad_wrt_relaxation_time_is_finite` (relaxation mode,
5 steps, finite and non-zero), one `jax.test_util.check_vjp` per slab step
on a 4×3 grid, `test_params_in_yaml_never_needed` (constructing every slab
with no `params` argument equals constructing with `Parameters.default()`).

### T1.5 `SlabGrid.from_coords`; drop the `"JCM::T31"` DSL

**File:** `jem/components/slab/grid.py`.
`SlabGrid.from_coords(horizontal, fractional_mask=None, threshold=0.5)`
builds from a dinosaur horizontal grid (radians, `(n_lon, n_lat)`).
`generate_slab_grid` and `load_jcm_fractional_mask` are deleted;
`generate_slab_grid_from_scrip` → `SlabGrid.from_scrip`. `TerrainData.fmask`
is the mask source.

**Tests:** `test_from_coords_matches_scrip_t31` (compare with
`jem/data/JCM_T31.SCRIP.nc` centres to 1e-6 rad).

### T1.6 Output names consistent with JCM; constants from `jcm.constants`

- Slab `to_xarray(diagnostics, time)`: dims `("lon", "lat")` with 1-D degree
  coords when the grid came from `from_coords` (same values JCM writes);
  curvilinear grids keep 2-D `lat`/`lon` auxiliary coordinates with a CF
  `coordinates` attribute. The time coordinate comes from the coupler's
  `TimeAxis` (datetimes), not hours-since. Test: `xr.merge([ds_atm, ds_ocn])`
  succeeds on the default run. Update `CLAUDE.md`'s output section.
- `jem/constants.py`: delete every value JCM has (`grav`, `cpd`, `sbc`,
  `tmelt`, `rhoi`, `alf`); use `import jcm.constants as c; c.<name>`. Keep
  the ocean/land/ice/bulk values JCM lacks in a
  `@dataclass(frozen=True) SurfaceConstants` singleton with
  `set_constants(**kw)` mirroring JCM's.

### T1.7 Coupler cleanup

- `Coupler.run()` deleted (the loop lives in `jem.driver.run_chunked`,
  T2.2). Components return a plain diagnostics pytree per step; the coupler
  stacks over steps (remove the double `unwrap_leading_dims` and every
  `stack_objects([result])`).
- Delete `adhoc_scan` (**DECISION**; default delete — `jax.disable_jit()`
  covers debugging), `get_info`, `tree_tools.py`, `verbose`/`show_progress`;
  add `Coupler.__repr__`; log at DEBUG.

## Phase 2 — driver and configuration (built against the Phase 1 API)

### T2.0 jax-gcm dependency gate

**Files:** `jem/components/jcm/contract.py` (new), `.github/workflows/tests.yml`,
`pyproject.toml`, `tests/unit/test_jcm_contract.py` (new).

1. **Pin.** `JCM_SUPPORTED_REV = "<tag or sha>"` in `contract.py`
   (**DECISION**: the first jax-gcm release containing PR #750's config
   schema; default: the tag jax-gcm cuts from `dev` after #750 merges, or
   `dev @ <sha>` until then). The required CI job checks out exactly that
   revision; a second, `continue-on-error: true` canary job tracks `dev`.
   `pyproject.toml` pins the same floor.
2. **Record the integration points.** `contract.py` lists, as a module
   constant with a comment each, every jax-gcm name JAX-ESM calls:
   `jcm.runners.build_model`, `jcm.runners.build_forcing`,
   `jcm.forcing.default_forcing`, `Model.bootstrap_state`,
   `Model.run_from_state_with_carry`, `Model.dt_si`, `Model.start_date`,
   `physics.get_empty_data`, `jcm.date.parse_duration_days`,
   `jcm.date.days_per_year`, `jcm.diagnostics.check_health`,
   `jcm.constants`, `pkg://jcm.config` composition, the diagnostics keys in
   `exchange_fields`. `test_jcm_contract.py` imports each and, for the
   private ones, asserts they still exist so a jax-gcm rename fails here
   with a clear message instead of deep inside a run.
3. **Composed options must take effect.** `test_atmosphere_options_take_effect`
   composes a representative matrix (`physics ∈ {speedy, held_suarez, echam}`
   × their native grid, `terrain ∈ {aquaplanet, from_file}`,
   `init ∈ {isothermal, jw}`, `run.time_step=7`) and asserts on the *built*
   model: `model.dt_si == 7 min`, physics term names match the option,
   `coords.horizontal.total_wavenumbers` matches the grid, terrain kind
   reflected in `terrain.orog` being non-zero for `from_file`. A composed
   but ignored option fails this test.
4. **Option-name parity.** `test_atmosphere_subgroup_options_match_jcm`
   lists `importlib.resources.files("jcm.config") / <group>` for every
   imported subgroup and asserts each option composes under
   `<group>@atmosphere.<group>` (cheap: compose only, no build).

### T2.1 `jem/config/` composing JCM's groups under `atmosphere`, plus a coupled `experiment` group

**Files (new):**
```
jem/config/config.yaml
jem/config/ocean/{none,slab,slab_qflux,slab_relax,veros}.yaml
jem/config/land/{none,slab_speedy}.yaml
jem/config/seaice/{none,slab}.yaml
jem/config/coupling/daily.yaml
jem/config/regrid/{same_grid,esmf}.yaml
jem/config/run/{default,smoke,longrun}.yaml
jem/config/experiment/{aquaplanet-slab,aquaplanet-slab-mixed-grid,earth-slab,veros-double-drake,veros-earth}.yaml
jem/config/hydra/help/custom_help.yaml   (copy jax-gcm's, rename app)
```
`pyproject.toml`: `jem = ["config/**/*.yaml", "data/*.nc", "py.typed"]`
under `[tool.setuptools.package-data]`.

**`config.yaml` (verified to compose and build on jax-gcm dev):**
```yaml
hydra:
  searchpath:
    - pkg://jcm.config          # jcm ships its YAML as package data
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: multirun/${hydra.job.num}
defaults:
  - _self_
  # --- atmosphere: jcm's own config groups, re-rooted under `atmosphere`; names/options identical to jcm ---
  - physics@atmosphere.physics: speedy
  - grid@atmosphere.grid: speedy_t31_l8
  - dycore@atmosphere.dycore: dinosaur
  - run@atmosphere.run: default
  - init@atmosphere.init: isothermal
  - terrain@atmosphere.terrain: aquaplanet
  - forcing@atmosphere.forcing: default
  - nudging@atmosphere.nudging: none
  - diffusion@atmosphere.diffusion: default
  # --- jem's own groups ---
  - ocean: slab
  - land: none
  - seaice: slab
  - coupling: daily
  - regrid: same_grid
  - run: default
  - override hydra/help: custom_help
atmosphere:
  constants: {}       # jcm.runners applies +atmosphere.constants.grav=... before build
```
Override spelling (document in `custom_help.yaml`):
`physics@atmosphere.physics=echam grid@atmosphere.grid=echam_t63_l47_hybrid`,
`atmosphere.run.time_step=20`, `+experiment@atmosphere=speedy-t31`
(a jax-gcm experiment bundle re-rooted — **verified**: its `override
/physics` etc. resolve to `atmosphere.*`, its `terrain/forcing/run` values
land, CLI overrides still apply on top), `+experiment=earth-slab` (a JAX-ESM
coupled experiment).

**Group files — wiring only.** Each node is a Hydra `_target_` block, built
with `hydra.utils.instantiate(node, grid=grid)` (the runner injects objects
that come from other components; YAML never names them). No physics
parameter appears here; no Python default is repeated here.

```yaml
# ocean/slab.yaml                      -- the default ocean: everything from Python defaults
_target_: jem.components.SlabOceanModel
params:
  _target_: jem.components.slab.SlabOceanParameters   # wiring only, so +ocean.params.<field>=<value> works

# ocean/slab_relax.yaml                -- a named, non-default configuration worth highlighting
_target_: jem.components.SlabOceanModel
sst_clim_file: ???                     # required input, e.g. jcm-data://t30/clim/forcing.nc
params:
  _target_: jem.components.slab.SlabOceanParameters
  forcing_method: relaxation           # the choice that defines this configuration

# ocean/slab_qflux.yaml
_target_: jem.components.SlabOceanModel
q_flux_file: ???
params:
  _target_: jem.components.slab.SlabOceanParameters
  forcing_method: qflux

# ocean/veros.yaml
_target_: jem.components.VerosComponent
setup: ???                             # importable dotted path to a VerosSetup subclass,
                                       # e.g. jem.components.veros.setups.double_drake.Setup (never a file path)
grid_file: ???                         # SCRIP file; the runner builds the SlabGrid from it

# ocean/none.yaml, land/none.yaml, seaice/none.yaml
null                                   # the group resolves to None; the runner skips it

# land/slab_speedy.yaml
_target_: jem.components.SlabLandModel
params:
  _target_: jem.components.slab.SlabLandParameters

# seaice/slab.yaml
_target_: jem.components.SlabSeaiceModel
params:
  _target_: jem.components.slab.SlabSeaiceParameters

# coupling/daily.yaml                  -- coupling wiring; `null` means "use the Python default"
timestep: "1 day"                      # jcm.date.parse_duration_days string form
workflow: null                         # null -> jem.exchange.default_workflow(components)
exchanges: null                        # null -> jem.exchange.default_exchanges(components), defined ONCE in Python
mapper: null                           # dotted path to a Python callable used INSTEAD of `exchanges` (Veros)

# regrid/same_grid.yaml
null                                   # identity
# regrid/esmf.yaml
_target_: jem.regrid.ESMFRegridders    # dict-like of named ESMFRegridder built from weight files
weights:
  a2o_conserve: ???                    # jem://<file> resolves into importlib.resources.files("jem.data")
  a2o_bilinear: ???
  o2a_conserve: ???
  o2a_bilinear: ???

# run/default.yaml                     -- keyword arguments of jem.run_chunked(); defaults live on that function
total_time: "30 days"
output_dir: null                       # null -> the Hydra run dir
log_level: INFO                        # consumed by jem.main only (sets the "jem" logger); not a run_chunked argument
# run/smoke.yaml
defaults: [default, _self_]
total_time: "2 days"
chunk: "2 days"
# run/longrun.yaml
defaults: [default, _self_]
total_time: "3600 days"
chunk: "30 days"
output_averages: true
checkpoint_path: checkpoint.msgpack
```

**Coupled experiments (`experiment/*.yaml`)** follow jax-gcm PR #750's
convention exactly — `# @package _global_`, a WHY comment per non-default
setting, absolute `override` lines, required local inputs left as `???`:
```yaml
# experiment/earth-slab.yaml
# @package _global_
# Earth-like: realistic orography + climatological SST/land relaxation, JCM T31 SPEEDY + slab ocean + SPEEDY slab land.
#  - terrain/forcing from the packaged T30 climatology bundle (jcm-data://) so the example runs offline.
#  - ocean=slab_relax (30-day relaxation to the file's SST) because a free slab drifts within months (issue #1 in the docs).
#  - land=slab_speedy: consumes stl/snowc/soilw from the same bundle.
defaults:
  - override /terrain@atmosphere.terrain: from_file
  - override /forcing@atmosphere.forcing: from_file
  - override /ocean: slab_relax
  - override /land: slab_speedy
atmosphere:
  terrain: {file: jcm-data://t30/clim/terrain.nc}
  forcing: {file: jcm-data://t30/clim/forcing.nc}
ocean:
  sst_clim_file: jcm-data://t30/clim/forcing.nc
  params: {relaxation_time: 2592000}      # 30 days: a NON-default choice that defines this experiment
land:
  land_clim_file: jcm-data://t30/clim/forcing.nc
```
Names match jax-gcm experiments only when the atmospheric part is
scientifically equivalent; a coupled experiment may reuse one by adding
`- /experiment@atmosphere: speedy-t31` to its defaults (**verify in
`test_experiments_compose`**; the CLI form is verified, the in-file form is
not yet).

Rules that keep this thin, checked at review and by tests:
- A key may appear in a group or experiment file only if it is (a)
  `_target_`, (b) a required input marked `???`, or (c) a value that differs
  from the Python default *and* is what the named configuration is about.
  `test_config_has_no_python_defaults` instantiates every group option and
  every experiment and fails if any supplied kwarg equals the target's
  default (`inspect.signature(...).parameters[k].default` and the params
  dataclass fields).
- The runner never reads a physics parameter. Objects derived from other
  objects (the slab grid, the coupling timestep) are injected; nothing is
  computed in the runner.
- `jem.run_chunked()` owns the run defaults; `run/default.yaml` is the
  complete `run` schema (same convention as jax-gcm PR #750's
  `run/default.yaml`), and the other run files inherit it via
  `defaults: [default, _self_]` so every run key is overridable without `+`.

**Tests (`tests/unit/test_config.py`):** compose every group option
(parametrised), every experiment (`test_experiments_compose`, including
one that reuses a jax-gcm experiment under `atmosphere`), and assert the
`run` group has identical key sets across its files;
`test_config_has_no_python_defaults`; `test_installed_wheel_has_config`
(build a wheel, install in a temp venv, `importlib.resources.files("jem.config")`
lists `config.yaml` and `experiment/`).

### T2.2 `jem/driver.py` (Python API) and `jem/runners.py` (config → Python)

`jem/driver.py` — what a Python user calls and what every example calls:
```python
@dataclass(frozen=True)
class RunResult:
    final_carry: CoupledCarry
    steps_completed: int
    completed: bool                     # False if the health gate stopped the run
    reports: list[dict]
    paths: list[Path]

HealthCheck = Callable[[dict[str, xr.Dataset], int, float], tuple[bool, dict]]

def default_health_check(datasets, chunk_index, elapsed_days):
    """jcm.diagnostics.check_health on datasets["atm"] when present; otherwise (True, {"skipped": "no atmosphere"})."""

def run_chunked(
    coupler: Coupler, *,
    total_time: str | float,                 # "30 days" or days (jcm.date.parse_duration_days)
    chunk: str | float = "30 days",
    initial_carry: CoupledCarry | None = None,
    output_dir: Path | str = "outputs",
    output_averages: bool = False,
    subsample: int = 1,
    health_check: HealthCheck | None = default_health_check,   # None -> no gate
    bail_on_unhealthy: bool = True,
    checkpoint_path: Path | str | None = None,
) -> RunResult:
    """The one chunked run loop (body in T2.3). All defaults live HERE. Uses coupler.workflow/exchange —
    there is no second workflow argument."""
```
`jem/runners.py` — generic; no component names except the atmosphere:
```python
GROUP_TO_NAME = {"ocean": "ocn", "land": "lnd", "seaice": "seaice"}   # the only table in the module

def build_atmosphere(cfg) -> JCMComponent:
    """jcm.runners.build_model(cfg.atmosphere) + jcm.runners.build_forcing(...) -> JCMComponent(model, forcing=...).
    Sets cfg.atmosphere.run.{total_time,save_interval} = coupling step and output_averages=True before building
    (the atmosphere is integrated one coupling step per call) — wiring, not science. Calls
    jcm.runners.warn_on_config_traps(cfg.atmosphere, ...) when available."""
def build_grid(node, atm) -> SlabGrid:
    """node.grid_file absent -> SlabGrid.from_coords(atm.model.coords.horizontal, atm.terrain.fmask);
    present -> SlabGrid.from_scrip(resolve(node.grid_file))."""
def build_component(node, **injected):
    """hydra.utils.instantiate(node, **injected); None node -> None."""
def build_coupler(cfg) -> Coupler:
    atm = build_atmosphere(cfg)
    components = {"atm": atm}
    for group, name in GROUP_TO_NAME.items():
        node = cfg.get(group)
        if node is not None:
            components[name] = build_component(node, grid=build_grid(node, atm))
    regridders = build_component(cfg.regrid) or {}
    exchange = (hydra.utils.get_method(cfg.coupling.mapper) if cfg.coupling.mapper
                else Exchange(cfg.coupling.exchanges or default_exchanges(components), regridders))
    return Coupler(components, exchange,
                   coupling_timestep=parse_timedelta(cfg.coupling.timestep),
                   workflow=cfg.coupling.workflow,                 # None -> Python default
                   start_date=atm.model.start_date, calendar=atm.model.calendar)
def run(cfg) -> RunResult:
    """kwargs = {k: v for k, v in OmegaConf.to_container(cfg.run).items() if k != "log_level"};
    output_dir None -> HydraConfig.get().runtime.output_dir; return driver.run_chunked(build_coupler(cfg), **kwargs)"""
```
`jem/main.py`:
```python
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    logging.getLogger("jem").setLevel(cfg.run.log_level)
    logger.info("Composed config:\n%s", OmegaConf.to_yaml(cfg))   # no print()
    result = runners.run(cfg)
    logger.info("Finished: %d steps, completed=%s, %d files", result.steps_completed, result.completed, len(result.paths))
```
Also add `[project.scripts] jem = "jem.main:main"`.

**Tests (`tests/unit/test_runners.py`):** `test_runners_has_no_component_kwargs`
(none of the `SlabOceanParameters` field names appear in `runners.py`
source), `test_build_coupler_default` (names `{atm, ocn, seaice}`, workflow
`[exchange, atm, ocn, seaice]`), `test_cli_and_python_construction_agree`
(the Python construction from `docs/source/python_api.md` and
`build_coupler(compose("config"))` give the same component types, workflow,
coupling timestep, start date and `initialize()` treedef),
`test_missing_component_filters_exchanges`.

### T2.3 `jem/exchange.py`, `jem/output.py`, `jem/checkpoint.py`, the loop body

**`jem/exchange.py`** — reinstates the declarative mapper deleted in #103,
in its minimal form:
```python
@dataclass(frozen=True)
class ExchangeSpec:
    src: str        # "component.section.field"  (section in {state, derived, forcing})
    dst: str
    regrid: str | None = None

class Exchange:
    def __init__(self, specs: Sequence[ExchangeSpec | dict], regridders: Mapping[str, Callable]): ...
    def validate(self, initial_carry: CoupledCarry) -> None    # KeyError naming the spec; called once by Coupler.__init__
    def __call__(self, carry: CoupledCarry) -> CoupledCarry     # returns a NEW carry; never mutates

def default_exchanges(components: Mapping[str, Component]) -> list[ExchangeSpec]
    # the standard atm/ocn/lnd/seaice wiring (the list from v1 §T1.1), filtered to the components present
def default_workflow(components) -> list[str]        # ["exchange", *components]
```
Document in the module docstring, and in `architecture.md`, that this is
**lagged coupling**: with `[exchange, atm, ocn]`, the ocean at step *n* uses
the atmosphere's fluxes from step *n−1* and the atmosphere uses the SST
from the end of step *n−1*; the first ocean step sees zero fluxes. Same-step
sequencing is a follow-up (§Follow-ups).

**`jem/output.py`:**
```python
def diagnostics_to_datasets(coupler, diagnostics, time_axis) -> dict[str, xr.Dataset]   # only SupportsXarray components
def postprocess(ds, *, output_averages: bool, subsample: int) -> xr.Dataset
def write_chunk(datasets, output_dir: Path, chunk_index: int) -> list[Path]   # <output_dir>/<component>-<chunk:05d>.nc
```
**`jem/checkpoint.py`:** the pattern of `jcm/checkpoint.py` (flax msgpack of
flattened leaves + treedef rebuilt from a template + per-leaf shape check +
atomic `.tmp` rename) for a `CoupledCarry`: `save(carry, path)` /
`load(template_carry, path) -> CoupledCarry`. **`CoupledCarry.step` is in
the payload and is the source of truth on resume**; `elapsed_days` is
derived from it, never the reverse. Components that are
`SupportsCheckpoint` (Veros) get their sub-carry delegated to
`save_state/load_state` in a sibling directory. Delete
`jem/utils/checkpoints.py`; `save_veros_carry/load_veros_carry` become
those two methods on `VerosComponent`.

**`driver.run_chunked` body:**
```
carry = initial_carry or coupler.initialize()
if checkpoint_path and exists: carry = checkpoint.load(carry, path)          # carry.step restored exactly
steps_per_chunk = chunk_days / coupling_days   (must be an integer; raise otherwise)
n_steps_total = total_days / coupling_days     (integer; raise otherwise)
trajectory = coupler.generate_trajectory_function(steps_per_chunk)             # compiled ONCE
paths, reports = [], []
while int(carry.step) < n_steps_total:
    chunk_index = int(carry.step) // steps_per_chunk
    carry, diagnostics = trajectory(carry)
    datasets = {k: postprocess(v, ...) for k, v in diagnostics_to_datasets(coupler, diagnostics, coupler.time_axis(...)).items()}
    paths += write_chunk(datasets, output_dir, chunk_index)
    if checkpoint_path: checkpoint.save(carry, checkpoint_path)
    if health_check is not None:
        ok, report = health_check(datasets, chunk_index, elapsed_days(carry)); reports.append(report)
        log one INFO line; if not ok and bail_on_unhealthy: return RunResult(carry, int(carry.step), False, reports, paths)
return RunResult(carry, int(carry.step), True, reports, paths)
```
(A final partial chunk is not supported: `total_time` must be a multiple of
`chunk`; raise with a message that names both.)

**Tests:** `test_exchange_roundtrip` (values moved; input carry unchanged),
`test_exchange_unknown_field_raises_at_validate`,
`test_checkpoint_roundtrip_restores_step`,
`test_run_chunked_python_api` (a `Coupler` built in Python with **no
atmosphere** — two slab components on a 4×3 grid; `health_check` default
skips; two files per component; `result.final_carry.step == n`),
`test_continuous_chunked_resumed_agree` (10 steps in one chunk vs 5+5 vs
5, checkpoint, new process-equivalent `run_chunked` resuming 5: identical
`final_carry.step` and carries to 1e-12 on the toy coupler, 1e-6 with JCM,
`@pytest.mark.slow`), `test_run_smoke` (`+experiment=aquaplanet-slab
run=smoke` through `jem.main`, `@pytest.mark.slow`).

## Phase 3 — examples and user documentation, written once

### T3.1 Examples

`examples/README.md` lists every command. Ordinary examples are commands or
experiment files; bespoke notebooks contain only their unique operation and
start from `runners.build_coupler(compose(...))` — a shared, tested
builder — so no two notebooks construct components:

| example | becomes |
|---|---|
| 01 aquaplanet | `python -m jem.main +experiment=aquaplanet-slab` + a 15-line plotting cell reading `outputs/.../atm-00000.nc` (helpers in `jem/plot.py` behind `[plot]`). |
| 02 custom initial SST | 3 cells: `cfg = compose("config", overrides=["+experiment=aquaplanet-slab"]); coupler = runners.build_coupler(cfg)`; `carry = coupler.initialize(); carry = replace_field(carry, "ocn.state.sea_surface_temperature", sst + bump)`; `run_chunked(coupler, total_time="60 days", initial_carry=carry, output_dir=...)`. |
| 03 response via jvp | same first cell; `trajectory = coupler.generate_trajectory_function(5)`; the existing `jax.jvp` cell with the perturbed carry built by `replace_field` (no global mutation). |
| 04 mixed grid | `python -m jem.main +experiment=aquaplanet-slab-mixed-grid` (the experiment carries the SCRIP grid and the four `jem://` weight files). |
| earth | `python -m jem.main +experiment=earth-slab`. |
| long aquaplanet | `python -m jem.main +experiment=aquaplanet-slab run=longrun grid@atmosphere.grid=speedy_t106_l8`. |
| Veros double-drake / earth | `python -m jem.main +experiment=veros-double-drake`; the `VerosSetup` subclasses move to `jem/components/veros/setups/{double_drake,earth}.py` (importable targets, parameterised by grid file); `main.py`, `model_setup.py`, `run.sh`, `veros_helper.py` and the resurrected `02_experimental_JCM_Veros/` copy are deleted; `modify_jcm_terrain.py` → `jem/tools/idealised_terrain.py`; wind stress and the swamp-ice mask → `jem/fluxes.py` (`bulk_wind_stress`, `mask_fluxes_under_ice`), referenced from the experiment via `coupling.mapper: jem.fluxes.veros_exchange`. |

`tests/examples/test_examples.py` runs every `experiment/*.yaml` with
`run=smoke` (`@pytest.mark.slow`; Veros ones skipped unless `veros` is
importable) and executes the two notebooks.

### T3.2 Documentation

- `docs/source/python_api.md` (new): the complete direct Python
  construction — `JCMComponent`, `SlabOceanModel`, `SlabSeaiceModel`,
  `Exchange`, `Coupler`, `run_chunked` — ~25 lines, **executed by
  `test_readme_quickstart.py`** (the README quick start is this block; the
  test extracts it from `python_api.md` and asserts the README contains the
  identical block).
- `getting_started.rst`: CLI section mirroring jax-gcm's (`--help`,
  `--cfg job`, override spelling, `+experiment=`), then the Python API page.
- `architecture.md`: carry layout, `CoupledCarry.step` clock, lagged
  coupling, checkpoint contents.
- Delete `tutorial.rst`'s monkey-patching walkthrough; `Component` protocol
  page replaces it ("Adding a component": implement `initialize`/`step`,
  optionally `to_xarray`, `save_state`/`load_state`, `bind`).

## Phase 4 — residual cleanup and API freeze

- Remove `make_jem_compatible` shims (**DECISION**: deprecation window;
  default: one minor release, i.e. removed in 1.1).
- `py.typed` shipped; `mypy --strict` on `jem/base/` (**DECISION**; default
  yes for `base/` only).
- Tag `v1.0.0`; `CHANGELOG.md` lists every removal since 0.1.0.

## Phase 5 — asks of jax-gcm (file as issues at the start of Phase 1)

Each issue: title by the gap, where the hooks are, reference formulation,
what JAX-ESM deletes once it lands.

- **T5.1 Surface-exchange contract.** A `SurfaceExchange`-like struct
  published by every physics package (`ComposablePhysics` gathers it the way
  it gathers units tables): net downward heat flux into the surface,
  evaporation, total precipitation, near-surface wind or stress, per tile
  where tiles exist. Hooks: `speedy_surface_flux.py` (`hfluxn`, `evap`),
  `echam/surface_physics.py` (`SurfaceFluxes` NamedTuple already has
  sensible/latent/longwave_net/shortwave_net/ground_heat per type),
  `echam_1m.py` (`precip_rain/precip_snow`), SPEEDY `precnv/precls`.
- **T5.2 Public initial-state / carry API.** `Model.initial_state()`,
  `Model.physics_carry` replacing `_prepare_initial_dycore_state`,
  `_final_dycore_state`, `_final_physics_state`.
- **T5.3 `ModelPredictions.with_context(coords, physics)`.**
- **T5.4 Document `pkg://jcm.config` composition and `experiment@<pkg>`
  re-rooting** as supported (both verified working; JAX-ESM depends on
  them).
- **T5.5 Expose `Model._date_from_sim_time` publicly** so JAX-ESM's
  `CouplingTime` and JCM agree on calendar arithmetic.

## Definition of done

- `python -m jem.main` runs the default coupled aquaplanet and writes
  `atm-00000.nc`, `ocn-00000.nc`, `seaice-00000.nc` under the Hydra run dir;
  the same run from the documented Python API produces the same component
  topology, workflow, clock and initial carry (`test_cli_and_python_construction_agree`).
- Every ordinary example is a command or an `experiment` file; every
  bespoke notebook contains only its unique operation.
- All imported jax-gcm subgroup option names compose, and a representative
  matrix builds with the option visibly in effect (T2.0 tests), at the
  pinned jax-gcm revision.
- A multi-chunk run does not reset model time; continuous, chunked and
  checkpoint-resumed runs agree to the documented tolerance.
- `run_chunked()` returns `RunResult` and works for a coupler with no
  atmosphere and no `SupportsXarray` component.
- The public quick start is executed by the test suite; the installed wheel
  contains the config and data.
- `grep -rn "print(" jem/` → 0; `grep -rn "argparse" jem/ examples/` → 0; no
  numeric physics value in `jem/config/**/*.yaml`.
- `jax.grad` w.r.t. a slab parameter and `jax.jvp` w.r.t. initial SST both
  have tests. CI runs `ruff check .`, unit tests with `--cov-fail-under=80`
  (**DECISION**; default 80, 90 once slab tests exist), the smoke
  experiments on PRs, and the jax-gcm `dev` canary.

## Follow-up issues (file now; out of scope for this plan)

1. **Phase-aware, unit-explicit exchange contracts.** Decide and document
   the coupling scheme (the current scheme is lagged, see T2.3); support
   named exchange stages for same-step sequencing; define units, sign, grid
   location and extensive/intensive status of every public exchange field;
   add conservation and regridding tests; coordinate with T5.1.
2. **End-to-end run provenance.** Extend/consume `jcm.provenance` for a
   coupled run: resolved config, effective component parameters, code
   revisions and dirty state, JAX environment, input and weight files;
   stamp datasets and write a sidecar; config hash and checkpoint schema
   version.
3. **Release engineering.** Build and inspect wheel/sdist; install the wheel
   outside the checkout and run CLI/config/data smoke tests in CI; release
   checklist and compatibility policy (the version single-sourcing, `LICENSE`
   in the artifact and `py.typed` are already in T0.6/T0.7/T2.1).
