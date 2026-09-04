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
   `Coupler.generate_trajectory_function`); it is not a new bespoke driver. The bespoke drivers still
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

- A **component** is any object satisfying the `Component` protocol in
  `jem/base/component.py`: a `name`, an `initialize() -> carry` and a
  `step(carry, time) -> (new_carry, diagnostics)`, plus the optional
  capabilities `SupportsXarray`, `SupportsBind` and `SupportsCheckpoint`.
  There is still no base class to inherit from — it is a runtime-checkable
  `typing.Protocol` — so an external model is adapted by a thin wrapper class
  rather than being rewritten or monkey-patched.
- An **exchanger** is a plain function
  `(dict[str, carry], CouplingTime) -> dict[str, carry]`. It is the only place
  where components exchange information; there is no hidden flux bus.
- The **`Coupler`** owns the coupled model *and its clock*: it runs an ordered
  `workflow` naming components and exchangers once per coupling timestep, hands
  every component the same `CouplingTime`, and turns that step into a pure
  trajectory function with `generate_trajectory_function(iterations)`, driven
  by `jax.lax.scan`.

See `docs/source/design/architecture.md` for the carry layout, the contract
and the steps to add a component.

## Repository Structure

```
jem/                             # Main package
├── __init__.py                  # exports the coupling core (Coupler, the protocols)
├── constants.py                 # SurfaceConstants: what jcm.constants does not define
├── base/
│   ├── component.py             # the contract: Component + optional capabilities,
│   │                            #   CoupledCarry, CouplingTime, TimeAxis, Exchanger
│   └── coupler.py               # Coupler — the coupled model, its clock and its step
├── components/
│   ├── jcm/                     # the JCM atmosphere (jax-gcm)
│   │   ├── component.py         #   JCMComponent: wrapper class, threads the physics carry
│   │   └── exchange_fields.py   #   SurfaceExchange: JCM's diagnostics -> JEM's conventions
│   ├── jcm_component.py         # deprecated make_jem_compatible shim
│   ├── veros_component.py       # VerosComponent, the Veros ocean GCM (optional dep)
│   └── slab/
│       ├── base.py              # SlabModelBase: grid, climatologies, xarray output
│       ├── grid.py              # SlabGrid.from_coords / .from_scrip
│       ├── slab_ocean_model/    # SlabOceanModel  (mixed layer, frazil diagnostic)
│       ├── slab_land_model/     # SlabLandModel
│       ├── slab_seaice_model/   # SlabSeaiceModel (basal-only thickness)
│       └── slab_atmosphere_model/  # SlabAtmosphereModel (idealized, for tests)
│           # each model directory holds a params.py: its flax.struct parameters
├── data/                        # packaged grids, masks and regridding weights
└── utils/                       # cycles, checkpoints, esmf_regrid, time, ...
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
gives vector-math semantics and registers the class as a pytree. None of them
carries a clock — the coupler owns the only one:

```python
@tree_math.struct
class OceanState:
    sea_surface_temperature: jnp.ndarray
    mixed_layer_depth: jnp.ndarray
```

**Component parameters are differentiable.** Each slab model has a
`params.py` holding a `flax.struct.dataclass` whose numeric tunables are pytree
leaves you can take gradients with respect to, while genuinely *static*
configuration (flags selecting a code path at trace time) is marked
`pytree_node=False` and stays as Python aux data usable in ordinary `if`
branches:

```python
@struct.dataclass
class SlabOceanParameters:
    relaxation_time: float | jnp.ndarray = 60 * 86400.0              # differentiable leaf
    mixed_layer_depth_max: float | jnp.ndarray = 60.0                # differentiable leaf
    forcing_method: str = struct.field(pytree_node=False,
                                       default="none")               # static aux
```

The `float | jnp.ndarray` annotation is not decoration: mypy is a gate, and a
field annotated `jnp.ndarray` with a float default fails it, while the union
says exactly what the field accepts — a Python float from a default or a config,
or a traced array from an optimizer.

The parameters travel in the carry, as `carry["params"]`, rather than in a
closure over the model object, which is what makes `jax.grad` of a coupled run
with respect to a physical parameter work without any special casing in the
coupler. A new numeric tunable must not be made static: that hides it from
`jax.grad`, which is the whole point of the project.

### Logging, not printing
Use `logging.getLogger(__name__)`; there is **no** `print(...)` anywhere under
`jem/` and no new one should appear. A `print` inside a traced step function is
especially misleading: it fires once at trace time, not once per step. A library
must not write to a user's stdout either — the coupler logs its workflow at
DEBUG, and a driver or notebook is what prints.

### Physical constants
Constants shared with the atmosphere must come from **`jcm.constants`**, so that
the coupler and the atmosphere cannot disagree about `grav`, `cpd`, `tmelt` and
friends. Read them by attribute access on the module (`import jcm.constants as
c; ... c.grav`) rather than `from`-import, so a process-global
`set_constants(...)` override is honoured.

`jem/constants.py` holds only what `jcm.constants` genuinely does not define:
the properties of a seawater / land / ice *surface slab* and the bulk-formula
coefficients the idealized slab atmosphere uses. It follows the same pattern as
JCM's module — a frozen `SurfaceConstants` dataclass, a live singleton, a
`set_constants(...)` override (a whole instance or individual keyword fields)
and a module `__getattr__` — so `jem.constants.ocean_density` honours an
override made after import. Call `set_constants` *before* constructing the
components: a component reads the constants while building its initial carry and
while tracing its step.

Do not add a constant to `jem/constants.py` that `jcm.constants` already
defines. The values that used to be duplicated there were removed; the module
docstring tabulates each one against the `jcm.constants` name that replaced it,
including the three whose value changed.

### Naming
- **snake_case** for functions and variables, **PascalCase** for classes.
- Descriptive names for physical variables: `sea_surface_temperature`,
  `total_heat_flux`, `ice_frazil_melt_energy` — not `sst`, `hf`, `frzmlt`.
- Component and exchanger names share one namespace in a `Coupler` and must be
  unique. The functions that move fields between components are **exchangers**,
  not "mappers": one may regrid, compute a flux, convert units or simply copy a
  field, and "mapper" read as regridding.

### Sign and mask conventions
- Heat flux is **positive upward** (out of the surface); freshwater flux is
  positive upward (evaporation minus precipitation). JCM publishes downward
  positive, so the adapter negates once, at the boundary.
- `ice_frazil_melt_energy` follows CESM's `frzmlt`: positive means the mixed
  layer went sub-freezing and new ice forms; negative means surplus heat is
  available to melt existing ice.
- In a `SlabGrid`, `binary_mask == 1` means land and `0` means ocean.

### Output conventions
A coupled run writes one dataset per component, and they are only useful
together if `xr.merge` aligns them rather than producing an outer join. The
conventions are JCM's, and every component follows them:

- **Dimensions** are `("time", "lon", "lat")` for a separable lon/lat grid, and
  `("time", "x", "y")` with 2-D auxiliary `lat`/`lon` coordinates (plus a CF
  `coordinates` attribute on each variable) for a curvilinear one — CF and
  xarray forbid a 2-D variable named after one of its own dimensions.
- **Coordinate values** are degrees computed as `radians * 180 / pi` in float64
  (`jem.components.slab.grid.to_degrees`), which is character for character what
  `jcm.utils.data_to_xarray` does. Any other route risks a last-bit difference,
  which is enough to turn two 96-point longitude axes into a 119-point union.
- **The `time` coordinate** is an absolute `datetime64[ns]` axis — never
  "hours since <start>" — and record *k* is labelled with the **end** of the
  interval it covers, `start_date + (k+1)*dt`. `TimeAxis.datetimes()` is the
  single definition; `jem.utils.time.time_coordinate` unpacks it for xarray.
  The dates are proleptic Gregorian whatever the model calendar is.
- **Variable names**: state and derived quantities keep their plain names, and
  every variable that came from a component's *forcing* is written with a
  `forcing_` prefix — `jem.components.slab.base.FORCING_VARIABLE_PREFIX`,
  applied by `forcing_variable(name)`. Two components legitimately hold the
  same physical field — one produced it, the other received it — and without
  the prefix the merge of their datasets collides on the shared name. A new
  output variable that comes from `carry["forcing"]` goes through
  `forcing_variable()`; one that comes from `state` or `derived` does not.

### Type hints and docstrings
- Type hints in public signatures; `mypy jem/ --ignore-missing-imports` is a
  gate.
- NumPy-style docstrings for public functions and classes.

### Testing
- Tests under `tests/unit/` (fast, no external data) and `tests/examples/`
  (notebook and example smoke tests).
- A coupling change is not tested until two steps through
  `Coupler.generate_trajectory_function(2)` exercise it — a unit test of one
  component's step function does not catch a carry-structure mismatch, which
  only `lax.scan` sees.
- Include gradient checks for anything that claims to be differentiable.
