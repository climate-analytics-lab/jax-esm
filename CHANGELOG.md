# Changelog

All notable changes to JAX-ESM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and JAX-ESM aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from v1.0.0
onwards. Before v1.0.0 the public API may change in any release; every such
change is listed here.

## [Unreleased]

### Added

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
- The `veros` extra is deliberately empty: the coupler needs a *jittable* fork
  of Veros that is not published on PyPI and must be installed from git (see
  the README).

### Removed

- Dependencies `dataclasses-json`, `coordax`, `typing-extensions`, `jax_tqdm`
  and `tqdm`: none of them are imported by `jem` any more.
- **Progress bars.** `Coupler.run`, `Coupler.generate_trajectory_function` and
  `Coupler.generate_step_function` no longer display a `tqdm` progress bar.
  Their `show_progress` and `tqdm_kwargs` parameters are still accepted so
  existing callers keep working, but are deprecated and ignored; they will be
  dropped in a later release once per-chunk logging replaces them.
- `[tool.black]` and `[tool.isort]` configuration: `ruff` (pinned to 0.15.17,
  matching jax-gcm) is the only linter.
