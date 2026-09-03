# CLAUDE.md

## Think Before Coding
This is a coupling framework: components are black boxes with their own state
layouts, their own timesteps and their own scientific conventions, and the bugs
that matter live in the seams between them. Don't assume. Don't hide confusion.
Surface tradeoffs.

Before implementing:

 - State your assumptions explicitly. If uncertain, ask.
 - If multiple interpretations exist, present them - don't pick silently.
 - If a simpler approach exists, say so. Push back when warranted.
 - If something is unclear, stop. Name what's confusing. Ask.

Always document these decisions in the comments, and if appropriate in the
documentation (and possibly in the high-level design documentation under
`docs/source/design/`).

Comments should always reference the current state of the code, and explain
*why* it is doing what it is doing, not how it is different to some previous
version of the code (which can get out of date and confusing).

## Finish the Job — No Half Implementations
When asked to fix or implement something, deliver the **complete, faithful**
solution by default — do not ship a partial fix, a band-aid, or a "good enough
for now" workaround and present it as done. In a coupler "faithful" specifically
means the coupled system is right, not just the file you edited: the flux the
mapper moves has the sign, the units and the grid the receiving component
expects, the carry structure that comes out of a step matches the one that went
in, and conservation across the exchange is checked rather than assumed.

 - If the correct fix turns out to be deeper than expected, do the deeper fix.
   Don't silently descope to the shallow version.
 - A workaround/cap/guard is acceptable **only** as an explicitly-labelled
   stopgap that the user has agreed to — never as a substitute for the real fix.
 - Validate that the full fix actually works (tests + a short coupled run where
   relevant) before calling it done.
 - The only time to stop short is a genuine blocking decision that is the user's
   to make (per "Think Before Coding" above) — surface it and ask. Effort or
   tedium is not such a reason.

## Related findings get rolled into the PR; only unrelated ones become issues
When a piece of work surfaces additional defects or gaps, the default is to
**fix them in the same PR** whenever they are related to the work at hand — same
subsystem, same convention, same failure class, or anything a reviewer would
naturally want to see together. The maintainer would much rather review one
comprehensive PR than a cluster of small follow-ups. The test is: *would the
reviewer be surprised to find this fix in the PR?* If not, roll it in.

File a GitHub issue **only** for findings genuinely unrelated to the current
work — a different subsystem, something needing its own validation campaign, or
something blocked on resources or decisions the current PR cannot wait for. When
an issue is warranted:

 - Title it by the gap (not the PR that found it), with enough context to start
   cold: what is missing, where the hooks already are, and what reference
   behaviour applies.
 - Cross-link it from the code comment or docstring that notes the gap, and from
   the PR.

Either way, a docstring note or PR-body mention alone is never the resting place
for a known defect — it is either fixed in the PR or tracked in an issue.
Deliberately parked/rejected directions don't get an issue — record the decision
and its evidence where the decision was made instead.

## No bespoke run scripts — new configurations go through Hydra
Every runnable configuration must be expressible as a single command with
config-group overrides — never as a standalone driver script:

 - The **target** is `python -m jem.main` with Hydra groups under
   `jem/config/` (atmosphere, ocean, land, seaice, coupling, run,
   experiment), mirroring how `jcm` is driven. That driver and its config
   tree do not exist yet; they are Phase 2 of
   `docs/source/design/api_hardening_plan.md`.
 - **Until it does exist, do not add new standalone command-line driver
   scripts (a hand-rolled CLI plus a `run.sh`).** A new runnable
   configuration is a notebook or a short snippet in
   the docs that calls the Python API (`Coupler`, the component classes,
   `Coupler.run`); it is not a new bespoke driver. The bespoke drivers still
   under `examples/02_experimental/` are legacy and are being folded into the
   Hydra tree, not extended.
 - **Python is the primary interface; config is a thin wrapper.** Every
   physical parameter and its default value lives once, as a Python default on
   the component class. YAML may carry only wiring (`_target_`, required
   inputs, the non-default choices that define a named configuration) — never a
   parameter default, which would immediately drift from the Python one.
 - Canonical/validated configurations get their own named config file with
   comments explaining WHY each setting is what it is, so a production run is
   one command.
 - One-off experiment scripts (personal paths, GPU indices, ad-hoc drivers) do
   not belong in the repo at all.

## Documentation lives with the change — no doc debt
Documentation updates are part of the change, not a follow-up. A PR that alters
user-facing behaviour is incomplete until the docs say so:

 - **Where things go.** General design decisions and analyses belong in
   `docs/source/design/*.md` (added to the toctree in
   `docs/source/design.rst`); implementation-specific details and gotchas
   belong in the PR description. Do **not** create ad-hoc top-level `*.md`
   files in the repo root.
 - **User-facing behaviour changes** (new or changed defaults, new component
   constructor arguments, new carry keys, new CLI/config knobs) must be
   reflected in `README.md` and/or `docs/source/quick_start.rst` in the same
   PR.
 - Keep code cross-references (docstrings/comments pointing at design docs)
   updated when a doc moves.

## Project Overview

JAX-ESM (`jem`) is a fully differentiable **coupler** for Earth system
components, written in JAX. It does not implement a climate model of its own:
it connects independently-developed components — the JCM spectral atmosphere
from the sibling project [jax-gcm](https://github.com/climate-analytics-lab/jax-gcm)
(package `jcm`), JEM's own slab ocean / land / sea-ice models, and the Veros
ocean GCM — into one JIT-compilable, end-to-end differentiable simulation loop.

- **Package name:** `jem`
- **Python:** >= 3.11 (strict requirement; `X | Y` annotations)
- **License:** MIT
- **Status:** Alpha

The design in three sentences:

- A **component** is any object exposing `initialize() -> carry` and
  `generate_step_function() -> step(carry, step_index) -> (new_carry, predictions)`,
  plus the optional `predictions_to_xarray()` and `get_info()`. There is no base
  class to inherit from — the contract is duck-typed, so an external model can
  be adapted without being rewritten.
- A **mapper** is a plain function `CoupledCarry -> CoupledCarry`. It is the
  only place where components exchange information; there is no hidden flux
  bus.
- The **`Coupler`** runs an ordered `workflow` — a list naming components and
  mappers — once per coupling timestep, and drives that step function under
  `jax.lax.scan` (or a Python loop when `jitted=False`, for debugging).

See `docs/source/design/architecture.md` for the carry layout, the interface
contract and the steps to add a component.

## Repository Structure

```
jem/                             # Main package
├── __init__.py                  # exports Coupler, typing
├── constants.py                 # physical constants used by the slab models
├── base/
│   ├── coupler.py               # Coupler — the engine (workflow, scan, run)
│   ├── interface.py             # resolve_interface: duck-type resolver
│   └── typing.py                # type aliases + the JEMComponent dataclass
├── components/
│   ├── jcm_component.py         # adapter for the JCM atmosphere (jax-gcm)
│   ├── veros_component.py       # adapter for the Veros ocean GCM (optional dep)
│   └── slab/
│       ├── base.py              # SlabModelBase: grid, clock, xarray output
│       ├── grid.py              # SlabGrid + generate_slab_grid()
│       ├── slab_ocean_model/    # SlabOceanModel  (mixed layer, frazil diagnostic)
│       ├── slab_land_model/     # SlabLandModel
│       ├── slab_seaice_model/   # SlabSeaiceModel (basal-only thickness)
│       └── slab_atmosphere_model/  # SlabAtmosphereModel (idealized, for tests)
├── data/                        # packaged grids, masks and regridding weights
└── utils/                       # bulk_op, cycles, esmf_regrid, tree_tools, ...
docs/                            # Sphinx documentation (RST + MyST, shibuya theme)
├── source/design/               # design documents (this is where they go)
examples/                        # example notebooks and experimental setups
tests/                           # tests/unit/ and tests/examples/
```

## Build & Install

```bash
pip install -e ".[dev]"
```

`jem` depends on `jcm` (jax-gcm). For development, install the sibling checkout
in editable mode *before* installing `jem`, so the pinned release does not
shadow it:

```bash
pip install -e ../jax-gcm
pip install -e ".[dev]"
```

## Testing and linting

Run all three gates locally before pushing; CI must confirm a result you have
already seen, not discover it:

```bash
ruff check .                                        # ruff==0.15.17, must be clean
JAX_PLATFORMS=cpu pytest tests -q -m "not slow"     # must pass
JAX_PLATFORMS=cpu mypy jem/ --ignore-missing-imports
```

`JAX_PLATFORMS=cpu` is required on GPU hosts: without it every test process
grabs the same GPU and XLA fails with `CUDA_ERROR_OUT_OF_MEMORY`.

Ruff is the only linter (config in `pyproject.toml`); no formatter, no
pre-commit hooks. Tests live under `tests/` (`tests/unit/`, `tests/examples/`),
named `test_*.py`. Mark tests over ~1 minute with `@pytest.mark.slow`.

## Key Coding Conventions

### Functional programming with JAX
- All functions must be **pure** (no side effects) to work with JAX
  transformations (`jit`, `grad`, `vmap`, and above all `lax.scan`).
- No Python `if`/`else` on JAX-traced values — use `jnp.where()` or
  `jax.lax.cond()`. Branching on a *static* Python configuration flag (e.g.
  `forcing_method`) at trace time is fine and is the intended pattern.
- Array shapes must be **statically known**; a step function must return a carry
  with exactly the pytree structure, shapes and dtypes it received, or
  `lax.scan` will reject it.
- Never cast a whole pytree to a dtype to make `scan` agree on structure — find
  and fix the leaf that is wrong.

### Data structures
Component state, forcing and derived quantities use `@tree_math.struct`, which
gives vector-math semantics and registers the class as a pytree:

```python
@tree_math.struct
class OceanState:
    sim_time: jnp.ndarray
    sea_surface_temperature: jnp.ndarray
    mixed_layer_depth: jnp.ndarray
```

**Differentiable component parameters** are the target pattern for the slab
models: a `flax.struct.dataclass` whose numeric tunables are pytree leaves you
can take gradients with respect to, while genuinely *static* configuration
(enums/flags selecting a code path at trace time) is marked `pytree_node=False`
and stays as Python aux data usable in ordinary `if` branches:

```python
@struct.dataclass
class SlabOceanParameters:
    relaxation_time: jnp.ndarray = 2592000.0                         # differentiable leaf
    mixed_layer_depth_max: jnp.ndarray = 60.0                        # differentiable leaf
    forcing_method: ForcingMethod = struct.field(pytree_node=False,
                                                 default=ForcingMethod.NONE)  # static aux
```

Today the slab models still hold these as plain attributes set in `__init__`,
which keeps them out of the gradient — converting them is a planned task
(`docs/source/design/api_hardening_plan.md`, T1.4). New numeric tunables must
not be made static: that hides them from `jax.grad`, which is the whole point of
the project.

### Logging, not printing
Use `logging.getLogger(__name__)`; do **not** add `print(...)` anywhere under
`jem/`. Several modules still print (the coupler's workflow banner, the slab
models' initialization messages); those are being converted, and no new ones
should appear. A `print` inside a traced step function is especially misleading:
it fires once at trace time, not once per step.

### Physical constants
Constants shared with the atmosphere must come from **`jcm.constants`**, so that
the coupler and the atmosphere cannot disagree about `grav`, `cpd`, `tmelt` and
friends. Read them by attribute access on the module (`import jcm.constants as
c; ... c.grav`) rather than `from`-import, so a process-global
`set_constants(...)` override is honoured.

`jem/constants.py` today **duplicates** several of these values with different
names and slightly different numbers (`g0 = 9.81` against JCM's `grav`), and the
slab models read the duplicate. That is a known defect scheduled for removal
(T1.6); the module keeps only genuinely JEM-specific constants (seawater and
land-slab properties, ice properties). Do not add a constant to
`jem/constants.py` that `jcm.constants` already defines.

### Naming
- **snake_case** for functions and variables, **PascalCase** for classes.
- Descriptive names for physical variables: `sea_surface_temperature`,
  `total_heat_flux`, `ice_frazil_melt_energy` — not `sst`, `hf`, `frzmlt`.
- Component and mapper names share one namespace in a `Coupler` and must be
  unique.

### Sign and mask conventions
- Heat flux is **positive upward** (out of the surface); freshwater flux is
  positive upward (evaporation minus precipitation). JCM publishes downward
  positive, so the adapter negates once, at the boundary.
- `ice_frazil_melt_energy` follows CESM's `frzmlt`: positive means the mixed
  layer went sub-freezing and new ice forms; negative means surplus heat is
  available to melt existing ice.
- In a `SlabGrid`, `binary_mask == 1` means land and `0` means ocean.

### Type hints and docstrings
- Type hints in public signatures; `mypy jem/ --ignore-missing-imports` is a
  gate.
- NumPy-style docstrings for public functions and classes.

### Testing
- Tests under `tests/unit/` (fast, no external data) and `tests/examples/`
  (notebook and example smoke tests).
- A coupling change is not tested until a two-step coupled `Coupler.run()`
  exercises it — a unit test of one component's step function does not catch a
  carry-structure mismatch.
- Include gradient checks for anything that claims to be differentiable.
