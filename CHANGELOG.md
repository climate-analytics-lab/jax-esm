# Changelog

All notable changes to JAX-ESM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and JAX-ESM aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from v1.0.0
onwards. Before v1.0.0 the public API may change in any release; every such
change is listed here.

## [Unreleased]

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
