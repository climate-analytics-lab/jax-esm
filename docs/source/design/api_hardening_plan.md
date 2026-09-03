# JAX-ESM API hardening plan (implementation spec)

Companion to `api_and_usability_review.md`. That document says *what is
wrong*; this one says *exactly what to build*, in an order that keeps `main`
releasable after every PR. It is written so that a reviewer can check each
task against the review and an implementer can execute it without re-deriving
design decisions. Where a decision belongs to the maintainer it is marked
**DECISION** and the plan states the default that will be used if no answer
is given.

Everything below was checked against `jax-esm main @ c7b5bda` and
`jax-gcm dev @ 226172e4` (installed editable, CPU) unless stated otherwise.

---

## 0. Ground rules for every PR

- Branch from `main`, one PR per numbered task (a task may contain several
  commits). Task numbers below are the PR titles: `T1.2: …`.
- Gate before pushing (same as jax-gcm):
  ```bash
  ruff check .                                   # must be clean (ruff==0.15.17)
  JAX_PLATFORMS=cpu pytest tests/unit -q         # must pass
  JAX_PLATFORMS=cpu mypy jem/ --ignore-missing-imports
  ```
  From T1.1 onward also `JAX_PLATFORMS=cpu pytest tests -q -m "not slow"`.
- **Do not** add new `print(...)` calls; use `logging.getLogger(__name__)`.
- **Do not** add `argparse` scripts or `run.sh` files. Runnable configurations
  are Hydra overrides (T1.x).
- **Do not** cast pytrees to a dtype wholesale (`asfloat64`-style); fix the
  leaf that is wrong.
- **Do not** read JCM private attributes (`_prepare_initial_dycore_state`,
  `_predictions`, `_final_*`) outside `jem/components/jcm/` (T2.2 confines
  the ones that still have no public equivalent to one helper each, with a
  comment naming the JCM issue that will remove them).
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
- Docstrings: NumPy style; comments explain *why* (jax-gcm `CLAUDE.md`).
- Every task lists its tests. A task is not done until they exist and pass.
- Commit messages: imperative subject ≤ 72 chars, body explains why.

## Phase 0 — release blockers (no API change; each PR ≤ 1 day)

### T0.1 Replace `CLAUDE.md` with jax-gcm's conventions

**Files:** `CLAUDE.md` (rewrite), `DEVELOPER.md` (delete).

**Change:** copy the structure of `jax-gcm/CLAUDE.md` and adapt: project
overview (differentiable coupler; `jem` package), "No bespoke run scripts —
configurations go through Hydra" (verbatim rule from jax-gcm), "Documentation
lives with the change" (design docs under `docs/source/design/`), the gate
from §0, the JAX conventions section (pure functions, `tree_math.struct`
state, `flax.struct.dataclass` parameters, no Python `if` on traced values),
and a "Reading model output" section that documents the slab output dims
after T2.6. Remove the argparse / bash-script mandates and the
"required packages" list. Delete `DEVELOPER.md`; its still-true content
(carry structure, workflow, adapter carry layout) moves to
`docs/source/design/architecture.md` (new, ≤ 150 lines).

**Acceptance:** `grep -i argparse CLAUDE.md` returns nothing; the design
doc is in `docs/source/design.rst`'s toctree.

### T0.2 Fix the JCM adapter for jax-gcm ≥ 2.1 surface-flux layout

**Files:** `jem/components/jcm_component.py`, `pyproject.toml`,
`tests/unit/test_jcm_component.py` (new).

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
   longer publishes per-tile fluxes; see jax-gcm release notes #645). Grep
   `notebooks/` and `docs/` for `land_heat_flux|ocean_heat_flux` and remove
   the references (none are used by a mapper today).
2. Delete `asfloat64` and both call sites. Verified on jax-gcm dev:
   `default_forcing(...)` and the initial dycore state contain only float
   leaves, so the cast is unnecessary; the physics carry contains `int32`
   and `bool` leaves that must *not* be cast (T2.2 threads it).
3. `pyproject.toml`: `"jcm>=2.1.0b0"` (**DECISION**: pin to the jax-gcm
   release that ships the 2-D flux layout; default `>=2.1.0b0`).
4. `.github/workflows/tests.yml`: check out `jax-gcm` at `dev` (`ref: dev`
   in the `actions/checkout` step) and install it with
   `pip install -e ./jax-gcm` *before* `pip install -e ".[dev]"`; delete the
   `PYTHONPATH=` export. **DECISION**: `dev` vs a tag; default `dev`, since
   the maintainer said hardening is happening on `dev`.

**Tests (`tests/unit/test_jcm_component.py`):**
- `test_initialize_carry_structure`: T31 aquaplanet `Model`, adapter,
  `initialize()` returns keys `state/derived/forcing`; `derived.total_heat_flux.shape == (96, 48)`.
- `test_two_coupling_steps_run`: `Coupler` with the adapter + `SlabOceanModel`,
  `run(iterations=2)` completes; `final["atm"]["derived"].total_heat_flux`
  finite. Mark `@pytest.mark.slow` if > 60 s on CI CPU (it is ~20 s locally).
- `test_no_float64_warning`: run under `warnings.catch_warnings(record=True)`
  and assert no `UserWarning` mentioning `float64`.

**Acceptance:** the smoke test in the review (§9.1) passes against jax-gcm dev.

### T0.3 Fix `SlabLandModel._idealized_land_temperature` axis bug

**File:** `jem/components/slab/slab_land_model/slab_land_model.py`,
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
`std(T[:, j, 0]) == 0` for every `j` (zonally uniform), `std(T[i, :, 0]) > 1`.
`test_land_step_runs_and_is_finite`: 3 steps with zero heat flux; temperature
finite and within 150–350 K.

### T0.4 Load `Q_flux_file` in `SlabOceanModel`

**File:** `slab_ocean_model.py`, `tests/unit/test_slab_ocean_model.py` (new).

**Change:** in `initialize()`, when `forcing_method == "Qflux"` and
`Q_flux_file` is set, read it with `xr.open_dataset(...)["qflux"]`
(**DECISION**: variable name and layout; default `qflux` with dims
`(lon, lat, time)` of length-12 time, i.e. the same layout `SST_clim` uses)
and put it into `OceanForcing.q_flux`. Validate `shape == grid.shape + (12,)`
and raise `ValueError` naming the file otherwise.

**Tests:** `test_qflux_file_is_loaded` (write a 12-month file to `tmp_path`,
check `carry["forcing"].q_flux` equals it); `test_qflux_zero_without_file`;
`test_relaxation_matches_analytic` (uniform grid, one step, compare against
the closed-form Euler-backward update); `test_freeze_melt_energy_sign`.

### T0.5 Fix the README and docs quick start

**Files:** `README.md`, `docs/source/quick_start.rst`,
`docs/source/tutorial.rst`, `docs/source/developers.rst`,
`docs/source/index.rst`, `docs/requirements.txt`.

**Change:** replace the `BasicMapper` quick start with the code of
`notebooks/01_basic/01_aquaplanet.ipynb` cells 1–5 (function mapper,
`generate_slab_grid("JCM::T31")`, `Coupler`, `run`, `predictions_to_xarray`)
— this is a stopgap until T1.4 replaces it with a CLI one-liner. Remove
`tutorial.rst` sections that reference `_prepare_initial_modal_state`,
`predictions.physics.surface_flux`, `land_model_active`; keep steps 1–4 with
the current adapter code pasted from `jcm_component.py`. Replace the
placeholder text in `index.rst`. Replace `[jcm,plot]` in `developers.rst` by
`[dev]`. Make `docs/requirements.txt` list `shibuya` (the theme `conf.py`
uses) and drop `sphinx-rtd-theme`.

**Tests:** `tests/unit/test_readme_quickstart.py` extracts the first
```` ```python ```` block from `README.md` with a regex and `exec`s it with
`iterations` patched to 1 (`@pytest.mark.slow`). This is what keeps the
quick start honest from now on.

### T0.6 Lint, dependencies, CI hygiene

**Files:** `pyproject.toml`, `.github/workflows/tests.yml`, offending sources.

**Change:**
1. `pyproject.toml` dependencies → exactly:
   `jax, jaxlib, numpy, xarray, netcdf4, tree_math, jax_datetime, jcm>=…,
   flax, hydra-core, omegaconf`. Extras: `plot = [matplotlib, cartopy]`,
   `veros = [veros]` (**DECISION**: the jittable fork is not on PyPI; default
   is to document the git install in the README and leave the extra empty),
   `docs = [sphinx, shibuya, nbsphinx, myst-parser]`,
   `dev = [pytest, pytest-cov, pytest-xdist, mypy, ruff==0.15.17, nbformat, nbconvert]`.
   Remove `dataclasses-json`, `typeguard` (after T2.1), `jax_tqdm`, `tqdm`
   (progress bar becomes a `logging` line per chunk), `coordax`, `sphinx`,
   `shibuya`, `matplotlib`, `cartopy` from the main list. Delete
   `[tool.black]`, `[tool.isort]`. Add `[tool.ruff.lint]` copied from
   jax-gcm (`extend-select = ["D"]` plus its ignore list) and
   `per-file-ignores` for `*.ipynb`.
2. Delete unused code: `jem/utils/datetime_tools.py`,
   `jem/utils/domain_grid_tools.py`, `evaluate_periodic` and
   `vmap_evaluate_periodic` in `cycles.py`, `concat_objects` and `mean_leaf`
   in `bulk_op.py`, `generate_slab_grid_from_ugrid` and its two helpers in
   `grid.py`, `example_usage` and the `__main__` block in `esmf_regrid.py`,
   `create_regridder_pair`, `create_regridder_from_xarray`.
3. Fix the two E402s in `veros_component.py` by moving the
   `runtime_settings` mutation into a function `_configure_veros_runtime()`
   called at the top of `make_jem_compatible` (module-level side effects on
   import go away at the same time).
4. `ruff check .` → clean (fix the 59 findings in notebooks/tests; use
   `ruff check --fix` for the auto-fixable ones and review the diff).
5. CI: lint on Python 3.11 with `ruff==0.15.17` and `ruff check .`;
   test matrix `ubuntu-latest × {3.11, 3.12}` only (**DECISION**: macOS and
   3.13 add cost without coverage of anything JAX-specific; default drop);
   notebooks move to a separate job that runs on PRs only (they get replaced
   by example configs in T1.4). Codecov condition → `3.11`.

**Acceptance:** `ruff check .` clean; `pip install -e .` in a fresh venv pulls
no plotting or docs packages (`pip show cartopy` fails).

### T0.7 License (**DECISION**)

MIT today, jax-gcm is Apache-2.0. Default if unanswered: keep MIT. If the
answer is Apache-2.0: replace `LICENSE`, set `license = "Apache-2.0"` in
`pyproject.toml`, add the classifier. One-commit PR.

## Phase 1 — the driver layer (one PR per task; T1.1–T1.3 land together if preferred)

### T1.1 `jem/config/` with JCM's groups composed under `atmosphere`

**Files (new):**
```
jem/config/config.yaml
jem/config/ocean/{none,slab,slab_qflux,slab_relax}.yaml
jem/config/land/{none,slab_speedy}.yaml
jem/config/seaice/{none,slab}.yaml
jem/config/coupling/daily.yaml
jem/config/regrid/{same_grid,esmf}.yaml
jem/config/run/{default,smoke,longrun}.yaml
jem/config/hydra/help/custom_help.yaml   (copy jax-gcm's, rename app)
```
`pyproject.toml`: add `jem = ["config/**/*.yaml", "data/*.nc"]` under
`[tool.setuptools.package-data]`.

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
  # --- atmosphere: jcm's own config groups, re-rooted under `atmosphere` ---
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
Override spelling users will use (document it in `custom_help.yaml`):
`physics@atmosphere.physics=echam grid@atmosphere.grid=echam_t42_l8_sigma`,
`atmosphere.run.time_step=20`, `terrain@atmosphere.terrain=from_file
atmosphere.terrain.file=/path/terrain.nc`.

**Group files — wiring only.** Each node is a Hydra `_target_` block, built
with `hydra.utils.instantiate(node, grid=grid)` (the runner injects objects
that come from other components; YAML never names them). No physics
parameter appears here; no Python default is repeated here. The complete
set of files:

```yaml
# ocean/slab.yaml                      -- the default ocean: everything from Python defaults
_target_: jem.components.SlabOceanModel
params:
  _target_: jem.components.slab.SlabOceanParameters   # wiring only, so that
                                                       # +ocean.params.<field>=<value> works from the CLI

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
setup: ???                             # dotted path to a VerosSetup subclass, e.g. examples.veros.double_drake.Setup
grid_file: ???                         # SCRIP file; the runner builds the SlabGrid/regridders from it

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
workflow: null                         # null -> jem.exchange.default_workflow(components): exchange first, then components in group order
exchanges: null                        # null -> jem.exchange.default_exchanges(components): the standard atm/ocn/lnd/seaice wiring, defined ONCE in Python
mapper: null                           # dotted path to a Python callable used INSTEAD of `exchanges` (Veros example)

# regrid/same_grid.yaml
null                                   # identity
# regrid/esmf.yaml
_target_: jem.regrid.ESMFRegridders    # a dict-like of named ESMFRegridder built from weight files
weights:
  a2o_conserve: ???                    # jem://<file> resolves into importlib.resources.files("jem.data")
  a2o_bilinear: ???
  o2a_conserve: ???
  o2a_bilinear: ???

# run/default.yaml                     -- keyword arguments of jem.run_chunked(); defaults live on that function
total_time: "30 days"
output_dir: null                       # null -> the Hydra run dir
# run/smoke.yaml
total_time: "2 days"
chunk: "2 days"
# run/longrun.yaml
total_time: "3600 days"
chunk: "30 days"
output_averages: true
checkpoint_path: checkpoint.msgpack
```

Rules that keep this thin, to be checked at review:
- A key may appear in a group file only if it is (a) `_target_`, (b) a
  required input marked `???`, or (c) a value that differs from the Python
  default *and* is what the named configuration is about. A test
  (`test_config_has_no_python_defaults`) instantiates every group option and
  fails if any supplied kwarg equals the target's default (compare against
  `inspect.signature(...).parameters[k].default` and the params dataclass
  fields).
- The runner never reads a physics parameter. If a value needs computing
  from other objects (the slab grid, the coupling timestep), the runner
  passes the *object* as an injected kwarg; it does not compute anything.
- `jem.run_chunked()` owns the run defaults (`chunk`, `output_averages`,
  `subsample`, `health_check`, `checkpoint_path`, `log_level`). The `run`
  group files list only what differs, so `run/default.yaml` is two lines.

**Tests (`tests/unit/test_config.py`):** compose every group option with
`hydra.compose` (parametrised over `ocean`, `land`, `seaice`, `run`) and
assert it composes; `test_atmosphere_groups_present` asserts
`set(cfg.atmosphere) ⊇ {physics, grid, dycore, run, init, terrain, forcing,
nudging, diffusion}`.

### T1.2 `jem/runners.py` builders

**Files (new):** `jem/driver.py` (the Python API) and `jem/runners.py`
(config → Python API, nothing else).

`jem/driver.py` — this is what a Python user calls, and what every example
calls; the CLI adds nothing on top of it:
```python
def run_chunked(
    coupler: Coupler, *,
    total_time: str | float,                 # "30 days" or days
    chunk: str | float = "30 days",
    initial_carry: CoupledCarry | None = None,
    output_dir: Path | str = "outputs",
    output_averages: bool = False,
    subsample: int = 1,
    health_check: bool = True,
    checkpoint_path: Path | str | None = None,
) -> list[dict]:
    """The one chunked run loop (see T1.3 for the body). All defaults live HERE."""
```

`jem/runners.py` — generic; contains no component names except the
atmosphere (which is built by jcm's own runner):
```python
def build_atmosphere(cfg) -> JCMComponent:
    """jcm.runners.build_model(cfg.atmosphere) + jcm.runners.build_forcing(...) -> JCMComponent(model, forcing=...).
    Sets cfg.atmosphere.run.{total_time,save_interval} = coupling step and output_averages=True before
    building (the atmosphere is integrated one coupling step per call) — wiring, not science."""
def build_grid(node, atm) -> SlabGrid:
    """node.grid_file absent -> SlabGrid.from_coords(atm.model.coords.horizontal, atm.terrain.fmask);
    present -> SlabGrid.from_scrip(resolve(node.grid_file)). (T2.5; until then the equivalent generate_slab_grid calls.)"""
def build_component(node, **injected):
    """hydra.utils.instantiate(node, **injected) — grid and any other object kwargs are injected; None node -> None."""
def build_coupler(cfg) -> Coupler:
    atm = build_atmosphere(cfg)
    components = {"atm": atm}
    for group in ("ocean", "land", "seaice"):
        node = cfg.get(group)
        if node is not None:
            components[GROUP_TO_NAME[group]] = build_component(node, grid=build_grid(node, atm))   # ocn / lnd / seaice
    regridders = build_component(cfg.regrid) or {}
    exchange = (hydra.utils.get_method(cfg.coupling.mapper) if cfg.coupling.mapper
                else Exchange(cfg.coupling.exchanges or default_exchanges(components), regridders))
    return Coupler(components, exchange,
                   coupling_timestep=parse(cfg.coupling.timestep),
                   workflow=cfg.coupling.workflow or default_workflow(components),
                   start_date=..., calendar=...)                       # from cfg.atmosphere.run
def run(cfg) -> list[dict]:
    """return driver.run_chunked(build_coupler(cfg), **OmegaConf.to_container(cfg.run), output_dir=<hydra run dir if null>)"""
```
`GROUP_TO_NAME = {"ocean": "ocn", "land": "lnd", "seaice": "seaice"}` is the
only table in the module. `default_exchanges` and `default_workflow` live in
`jem/exchange.py` (T1.3) and are the single definition of the standard
coupling; the YAML `null` defers to them.

`jem/main.py`:
```python
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    logging.getLogger("jem").setLevel(cfg.run.log_level)
    print(OmegaConf.to_yaml(cfg))
    runners.run(cfg)
```
Also add `[project.scripts] jem = "jem.main:main"` so `jem physics@…` works
as well as `python -m jem.main`.

**Tests (`tests/unit/test_runners.py`):** `test_runners_has_no_component_kwargs` (grep-style: `runners.py` never mentions a slab constructor keyword — assert none of the `SlabOceanParameters` field names appear in its source); `test_build_coupler_default`
builds the default config (T31 aquaplanet + slab ocean + sea ice) and asserts
component names `{atm, ocn, seaice}` and workflow
`[exchange, atm, ocn, seaice]`; `test_build_coupler_earth` with
`land=slab_speedy ocean=slab_relax terrain@atmosphere.terrain=from_file`
using `importlib.resources.files("jcm.data.bc.t30.clim")` files;
`test_missing_component_filters_exchanges`.

### T1.3 `jem/exchange.py`, `jem/output.py`, `jem/checkpoint.py`, `runners.run`

**`jem/exchange.py`** — reinstates the declarative mapper deleted in #103, in
its minimal form:
```python
@dataclass(frozen=True)
class ExchangeSpec:
    src: str        # "component.section.field"  (section in {state, derived, forcing})
    dst: str
    regrid: str | None = None

class Exchange:
    def __init__(self, specs: Sequence[ExchangeSpec | dict], regridders: Mapping[str, Callable]): ...
    def __call__(self, carry: CoupledCarry) -> CoupledCarry:
        # for each spec: value = _get(carry, src); value = regridders.get(name, identity)(value);
        # carry = _set(carry, dst, value)   -- _set returns a NEW carry: dict copies + struct.replace(); never mutates.
```
`_get/_set` support dict keys and `tree_math.struct` attributes (the
`strget/strset` helpers from the deleted `jem/mapping/mapper.py`, git
history `git show c7b5bda~1:jem/mapping/mapper.py`). Unknown component or
field → `KeyError` naming the spec, raised at construction time by a
`validate(initial_carry)` method the runner calls once.

**`jem/output.py`:**
```python
def predictions_to_datasets(coupler, predictions) -> dict[str, xr.Dataset]  # = Coupler.predictions_to_xarray, moved
def postprocess(ds, *, output_averages: bool, subsample: int) -> xr.Dataset   # the notebooks' isel/mean, once
def write_chunk(datasets: dict[str, xr.Dataset], output_dir: Path, chunk_index: int) -> list[Path]
    # writes <output_dir>/<component>-<chunk:05d>.nc with unlimited time dim; returns the paths
```
**`jem/checkpoint.py`:** copy the pattern of `jcm/checkpoint.py`
(flax msgpack of flattened leaves + treedef rebuilt from a template + per-leaf
shape check + atomic `.tmp` rename) for an arbitrary pytree:
`save_carry(carry, path, *, elapsed_days)` / `load_carry(template_carry, path) -> (carry, elapsed_days)`.
Components may override via optional `save_state(carry, dir)` /
`load_state(dir) -> carry` methods (Veros); `runners.run` checks
`hasattr(component, "save_state")`. Delete `jem/utils/checkpoints.py`; move
`save_veros_carry/load_veros_carry` into `veros_component.py` as those two
methods.

**`driver.run_chunked(coupler, ...)`** — the one copy of the loop that the examples
duplicate (`runners.run` only forwards `cfg.run` to it):
```
coupler, info = build_coupler(cfg)
carry = coupler.initialize()
n_chunks = ceil(total_days / chunk_days); steps_per_chunk = chunk_days / coupling_days (must be integer; raise otherwise)
if checkpoint_path exists: carry, elapsed = load_carry(carry, path); first_chunk = round(elapsed / chunk_days)
trajectory = coupler.generate_trajectory_function(workflow, iterations=steps_per_chunk)   # compiled ONCE
for c in range(first_chunk, n_chunks):
    carry, predictions = trajectory(carry)
    datasets = {k: postprocess(v, ...) for k, v in predictions_to_datasets(coupler, predictions).items()}
    paths = write_chunk(datasets, output_dir, c)
    if cfg.run.health_check: ok, report = jcm.diagnostics.check_health(datasets["atm"], c, elapsed); print_report; reports.append; if not ok: log + break
    if checkpoint_path: save_carry(carry, path, elapsed_days=...)
    log one INFO line per chunk (wall time, files)
return reports
```
`check_health` reads `temperature` and `specific_humidity` from the
atmosphere dataset; both exist in `ModelPredictions.to_xarray()` output.

**Tests:** `test_exchange_roundtrip` (two toy components, one spec each
direction, identity regrid; asserts values moved and the input carry object
was not mutated), `test_exchange_unknown_field_raises`,
`test_checkpoint_roundtrip_tmp_path`,
`test_run_chunked_python_api` (build a `Coupler` in Python with no config at all — two slab components on a 4×3 grid — and call `run_chunked(coupler, total_time="2 days", chunk="1 day", output_dir=tmp_path)`; two files per component), `test_run_smoke` (`run=smoke` through `jem.main`, `@pytest.mark.slow`; asserts three netCDFs
exist and `reports[0]["ok"]`), `test_run_resumes_from_checkpoint`
(run 2 chunks with `checkpoint_path`, delete chunk-1 output, run again,
assert chunk 0 not re-run — check file mtimes or log).

### T1.4 Rewrite the examples as configurations

**Files:** move `notebooks/` → `examples/` (as PR #106 does), then:

| example | becomes |
|---|---|
| `01_basic/01_aquaplanet.ipynb` | `examples/01_aquaplanet.md`: the one-liner `python -m jem.main run=smoke` plus a 15-line plotting cell reading `outputs/.../atm-00000.nc`. Plotting helpers go to `jem/plot.py` (`animate_surface(ds_atm, ds_ocn, ds_ice, path)`), used by every example, behind the `[plot]` extra. |
| `02_…customized_initial_condition.ipynb` | notebook of 3 cells, **pure Python, no config**: build `JCMComponent`, `SlabOceanModel`, `SlabSeaiceModel` and `Coupler` directly (this doubles as the README quick start); `carry = coupler.initialize(); carry = replace_field(carry, "ocn.state.sea_surface_temperature", sst + bump)`; `run_chunked(coupler, total_time="60 days", initial_carry=carry, output_dir=...)`. |
| `03_…using_gradient.ipynb` | notebook: same Python construction as 02, `trajectory = coupler.generate_trajectory_function(...)`, the existing `jax.jvp` cell (no global mutation: build the perturbed carry with `replace_field`). |
| `04_jcm_slabs_mixed_grid_aqua_planet.ipynb` | `python -m jem.main ocean.grid=jem://DisplacedPoleGrid.SCRIP.nc regrid=esmf regrid.weights.a2o_conserve=jem://weight_algo-conserve_JCM_T31_to_DisplacedPoleGrid.nc …` written as `examples/04_mixed_grid.yaml` (a Hydra config file) invoked with `--config-name`; plus the plotting cell. |
| `02_experimental/01_earth.ipynb` | `examples/earth.yaml`: `terrain@atmosphere.terrain: from_file`, `forcing@atmosphere.forcing: from_file`, `land: slab_speedy`, `ocean: slab_relax`, files from `jcm.data.bc.t30.clim` (resolve with a `jcm-data://` prefix in `_resolve_data_path`). |
| `03_long_aquaplanet.py` | delete; `python -m jem.main run=longrun grid@atmosphere.grid=speedy_t106_l8` (add that grid file to jcm or use `atmosphere.grid.spectral_truncation=106`). |
| Veros examples (PR #106) | `ocean=veros ocean.setup=examples/veros/double_drake_setup.py ocean.grid=jem://RotatedGaussianLatLon.SCRIP.nc regrid=esmf …`; `veros_case_setup.py` stays as the only bespoke file; `main.py`, `model_setup.py`, `run.sh`, `veros_helper.py` are deleted; `modify_jcm_terrain.py` becomes `jem/tools/idealised_terrain.py` (a library function, plus a `jem.tools.idealised_terrain` Hydra-free CLI is **not** added — call it from Python). Wind stress and the swamp-ice mask move to `jem/fluxes.py` (`bulk_wind_stress(u, v, *, cd=1e-3, rho=1.22, min_speed=1e-3)`, `mask_fluxes_under_ice(heat, fresh, sst)`) and are referenced from `coupling/veros.yaml` via `mapper: jem.fluxes.veros_exchange`. |

`examples/README.md` lists every command. `tests/examples/test_examples.py`
runs each YAML with `run=smoke` (`@pytest.mark.slow`).

**Tests:** the examples test above; `test_plot_import_optional` (importing
`jem` must not import matplotlib).

## Phase 2 — component contract and differentiable parameters

### T2.1 `Component` protocol; delete `resolve_interface`

**Files:** `jem/base/component.py` (new), `jem/base/coupler.py`,
delete `jem/base/interface.py`, `jem/base/typing.py` (keep the type aliases
in `component.py`), `tests/unit/test_interface.py` (delete),
`tests/unit/test_coupler.py` (trim to the coupler behaviour).

```python
class Component(Protocol):
    name: str
    def initialize(self) -> Carry: ...
    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]: ...
    def to_xarray(self, diagnostics: Diagnostics) -> xr.Dataset: ...   # optional; runner checks hasattr

@tree_math.struct
class CouplingTime:
    step: jnp.ndarray          # int32 coupling-step index
    sim_time: jnp.ndarray      # seconds since start_date (float)
    dt: float                  # coupling timestep, seconds (static)
```
`Coupler.add_component(name, component)` becomes: assert the two required
methods exist (`hasattr`), store the object. `JEMComponent` and
`raw_component` go away; `coupler.components[name]` is the object.

### T2.2 `JCMComponent` wrapper class; thread the physics carry

**File:** `jem/components/jcm/__init__.py` (+ `component.py`,
`exchange_fields.py`); `jem/components/jcm_component.py` keeps only a
deprecated `make_jem_compatible` that returns `JCMComponent(model, …)`
and warns.

```python
class JCMComponent:
    name = "atm"
    def __init__(self, model: Model, *, coupling_timestep: jdt.Timedelta, forcing: ForcingData | None = None):
        # check dt divides coupling_timestep (existing code); store forcing or default_forcing(coords.horizontal)
    def initialize(self) -> dict:
        self.model.bootstrap_state()                      # public jcm API
        physics_carry = self.model._final_physics_state   # TODO(jax-gcm#<T3.2 issue>): public accessor
        empty = self.model.physics.get_empty_data(self.model.coords)   # diagnostics template, no integration
        return {"state": self.model._final_dycore_state, "physics": physics_carry,
                "derived": JCMDerived.zeros(nodal_shape, physics=empty), "forcing": self.forcing}
    def step(self, carry, time):
        state, physics_carry, preds = self.model.run_from_state_with_carry(
            carry["state"], carry["forcing"], save_interval=self._days, total_time=self._days,
            output_averages=True, initial_physics_state=carry["physics"])
        diag = jax.tree.map(lambda x: x[0], preds.physics)      # strip the length-1 save axis
        ex = exchange_fields.speedy(diag)                        # SurfaceExchange struct, see below
        return {"state": state, "physics": physics_carry, "derived": JCMDerived(diag, **ex._asdict()),
                "forcing": carry["forcing"]}, preds
    def to_xarray(self, preds):
        return ModelPredictions(preds._predictions, self.model.coords, self.model.physics).to_xarray()  # TODO jax-gcm ask 3
```
`exchange_fields.py`:
```python
class SurfaceExchange(NamedTuple):
    net_heat_flux: Array          # W m-2, UPWARD positive (sign flip of jcm's downward hfluxn)
    evaporation: Array            # kg m-2 s-1, upward
    precipitation: Array          # kg m-2 s-1, downward
    u0: Array; v0: Array          # near-surface wind, m s-1
def speedy(diag) -> SurfaceExchange:  # keys _surface_flux.{hfluxn,evap,u0,v0}, _convection.precnv, _condensation.precls; g->kg /1000
def echam(diag) -> SurfaceExchange:   # keys echam_surface.{sensible_heat_flux,latent_heat_flux}, echam_1m_microphysics.{precip_rain,precip_snow},
                                      # radiation term's surface net sw/lw; raise NotImplementedError until jax-gcm ask 1 lands, with the issue number
def detect(diag) -> Callable:          # "_surface_flux" in diag -> speedy; "echam_surface" in diag -> echam; else raise with the key list
```
Verified on jax-gcm dev: the physics carry treedef is identical before and
after a step, so it scans; it contains `int32`/`bool` leaves — never cast it.
`total_freshwater_flux = evaporation - precipitation` stays as a derived
field for the existing exchanges.

**Tests:** `test_physics_carry_is_threaded` (two steps; the returned
`carry["physics"]` differs from the initial one), `test_initialize_does_not_integrate`
(patch `run_from_state_with_carry` with a spy; `initialize()` must not call
it), `test_speedy_exchange_shapes_and_signs` (downward `hfluxn` of +10
gives `net_heat_flux == -10`), `test_echam_not_implemented_names_issue`.

### T2.3 Coupler owns time; components drop clocks

**Files:** `jem/base/coupler.py`, all slab models, `JCMComponent`.

`Coupler.__init__(components, exchange=None, *, coupling_timestep: jdt.Timedelta, start_date: jdt.Datetime, calendar="365_day")`.
`generate_step_function` builds `CouplingTime(step=i, sim_time=i*dt, dt=dt)`
and passes it to every `component.step`. Slab `sim_time` leaves `*State`
(the coupler records `sim_time` in the diagnostics it stacks, so
`to_xarray` still gets a time coordinate); slab constructors lose
`start_datetime`, `timestep`, `calendar`; `SlabModelBase._year_fraction`
takes `time.sim_time` and the coupler-provided `start_day_offset`
(computed once in `Coupler.__init__` and passed via `CouplingTime` as a
static float `year_offset_seconds`). `JCMComponent` reads
`coupling_timestep` from the coupler at `bind(coupler)` — add
`Component.bind(self, coupling_timestep, start_date, calendar) -> None`
(optional method, called by `add_component`) so constructors need no time
arguments at all. `runners.build_*` stop passing them.

**Tests:** `test_components_share_clock` (two slabs after 3 steps have the
same `sim_time` in diagnostics), `test_year_fraction_wraps`.

### T2.4 Slab parameters as `flax.struct.dataclass`

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
`__init__` (they are data, not state) and stored as attributes; **no**
`cd_factor`/`time_factor`/`rhcapl`/`cdland` attributes — compute them inside
`step` from `self.params` (cheap). `params` is exposed as `self.params` and
included in the carry as `carry["params"]` so `jax.grad` w.r.t. it works
through `Coupler` without any special casing (**DECISION**: carry vs
closure; default carry, because it makes calibration a plain
`grad(lambda p: trajectory(replace_field(carry, "ocn.params", p)))`).
Nothing in `runners.py` changes: `params: {_target_: ...SlabOceanParameters}` in the group file plus
`+ocean.params.relaxation_time=...` on the CLI is the whole config surface, and every default stays in Python.

**Tests:** `test_grad_wrt_relaxation_time_is_finite` (`jax.grad` of mean
SST after 5 steps w.r.t. `relaxation_time`, relaxation mode, finite and
non-zero), one `jax.test_util.check_vjp` per slab step on a 4×3 grid.

### T2.5 `SlabGrid.from_coords`; drop the `"JCM::T31"` DSL

**File:** `jem/components/slab/grid.py`.
`SlabGrid.from_coords(horizontal, fractional_mask=None, threshold=0.5)`
builds from a dinosaur horizontal grid (`latitudes/longitudes` in radians,
`(n_lon, n_lat)` via `_broadcast_separable_grid_to_2d`). `generate_slab_grid`
and `load_jcm_fractional_mask` are deleted; `from_scrip` stays (renamed from
`generate_slab_grid_from_scrip`). `TerrainData.fmask` is the mask source.

**Tests:** `test_from_coords_matches_scrip_t31` (compare with
`jem/data/JCM_T31.SCRIP.nc` centres to 1e-6 rad).

### T2.6 Output names consistent with JCM; `jem.constants` from `jcm.constants`

- Slab `to_xarray`: dims `("lon", "lat")` when the grid came from
  `from_coords` (1-D `lon`/`lat` coords in degrees, same values JCM writes);
  curvilinear grids keep 2-D `lat`/`lon` auxiliary coordinates with a CF
  `coordinates` attribute. Time coordinate: `jdt`-derived datetimes, not
  hours-since. Test: `xr.merge([ds_atm, ds_ocn])` succeeds on the default run.
- `jem/constants.py`: delete every value JCM has (`grav`, `cpd`, `sbc`,
  `tmelt`, `rhoi`, ice latent heat `alf`); import `jcm.constants as c` and
  use `c.<name>` (attribute access, so `set_constants` overrides propagate).
  Keep `ocean_density`, `ocean_specific_heat_capacity`, `land_*`,
  `seawater_freezing_point_K`, `surface_air_density`,
  `bulk_drag_coefficient`, `atmosphere_column_mass` in a
  `@dataclass(frozen=True) SurfaceConstants` with a module-level singleton
  and a `set_constants(**kw)` mirroring JCM's.

### T2.7 Coupler cleanup

- `run()` → deleted from `Coupler` (lives in `runners.run`); keep
  `initialize`, `generate_step_function`, `generate_trajectory_function(workflow, iterations, *, remat=False)`.
- Diagnostics: components return a plain pytree per step; the coupler
  stacks over steps (remove the double `unwrap_leading_dims`; remove
  `stack_objects([result])` from every slab).
- `jitted` → `jit: bool = True`; delete `adhoc_scan` (**DECISION**: keep a
  Python loop for debugging? default: delete; `jax.disable_jit()` covers it).
- Immutable updates only (`dict(carry, **{name: new})`, `.replace()`); test
  `test_step_does_not_mutate_input`.
- Delete `get_info`, `tree_tools.py`; add `Coupler.__repr__`.
- Remove `verbose`/`show_progress`; log at DEBUG.

## Phase 3 — asks of jax-gcm (file as issues; link from the `TODO(jax-gcm#N)` comments)

Each issue body: title by the gap, where the hooks are, reference formulation, what JAX-ESM will delete once it lands.

- **T3.1 Surface-exchange contract.** A `SurfaceExchange`-like struct
  published by every physics package (`ComposablePhysics` gathers it the way
  it now gathers units tables): net downward heat flux into the surface,
  evaporation, total precipitation, near-surface wind or stress, per tile
  where tiles exist. Hooks: `speedy_surface_flux.py` (`hfluxn`, `evap`),
  `echam/surface_physics.py` (`SurfaceFluxes` NamedTuple already has
  sensible/latent/longwave_net/shortwave_net/ground_heat per type),
  `echam_1m.py` (`precip_rain/precip_snow`), `speedy_convection`/
  `speedy_condensation` (`precnv/precls`).
- **T3.2 Public initial-state / carry API.** `Model.initial_state()`,
  `Model.physics_carry` (read-only property) replacing
  `_prepare_initial_dycore_state`, `_final_dycore_state`,
  `_final_physics_state`.
- **T3.3 `ModelPredictions.with_context(coords, physics)`** so a
  `ModelPredictions` that went through a pytree transform can be
  re-attached without touching `_predictions`.
- **T3.4 Document `pkg://jcm.config` composition** in
  `getting_started.rst` (verified working; JAX-ESM depends on it).
- **T3.5 `jcm.date`-based `CouplingTime` helper** (optional): expose
  `Model._date_from_sim_time` publicly so JAX-ESM's coupler and JCM agree on
  the calendar arithmetic.

## Dependency order

```
T0.1 ─┐
T0.2 ─┼─ (independent, land in any order) ─► T1.1 ─► T1.2 ─► T1.3 ─► T1.4
T0.3 ─┤                                          │
T0.4 ─┤                                          └─► T2.1 ─► T2.2 ─► T2.3 ─► T2.4 ─► T2.7
T0.5 ─┤                                                        T2.5 ─► T2.6 (after T2.3)
T0.6 ─┘                                     T3.x filed at the start of Phase 2; T2.2's TODOs point at them
```
Phase 0 makes `main` releasable (v0.2). Phase 1 is the usability release
(v1.0.0b1). Phase 2 is the API freeze (v1.0.0).

## Definition of done for the whole plan

- `python -m jem.main` runs the default coupled aquaplanet and writes
  `atm-00000.nc`, `ocn-00000.nc`, `seaice-00000.nc` under the Hydra run dir.
- Every example in `examples/README.md` is a command or a ≤ 30-line
  notebook; `tests/examples` runs each at `run=smoke`.
- `grep -rn "print(" jem/` → 0; `grep -rn "argparse" jem/ examples/` → 0.
- No numeric physics value in `jem/config/**/*.yaml` (`test_config_has_no_python_defaults`); every example is reproducible from Python alone without Hydra.
- `jax.grad` w.r.t. a slab parameter and `jax.jvp` w.r.t. initial SST both
  have tests.
- CI runs `ruff check .`, unit tests with `--cov-fail-under=80`
  (**DECISION**: threshold; default 80, raise to 90 once slab tests exist),
  and the smoke examples on PRs.
