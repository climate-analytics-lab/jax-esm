# Architecture

How JEM couples black-box components. This is the reference for developers
adding a component or debugging an exchange; the user-facing walkthrough is
{doc}`../adding_a_component`. {doc}`../python_api` shows the same objects —
`Coupler`, the exchangers, `run_chunked` — built directly, for a reader who
wants the construction rather than the design rationale.

Every statement here, and in the docs it links to, is checkable against
`jem/base/component.py` and `jem/base/coupler.py`, which are the whole of the
coupling core; the layers built on that core — the declarative exchange
(`jem/exchangers.py`), the run loop (`jem/driver.py`, `jem/output.py`,
`jem/checkpoint.py`, `jem/accumulate.py`) and the configuration
(`jem/config/`, `jem/runners.py`, `jem/main.py`) — each have their own page,
linked below.

## Three concepts

- A **component** is any object satisfying the `Component` protocol in
  `jem/base/component.py`: a `name`, an `initialize() -> carry` and a
  `step(carry, time) -> (new_carry, diagnostics)`, plus the optional
  capabilities `SupportsXarray`, `SupportsBind` and `SupportsCheckpoint`.
  There is no base class to inherit from — `Component` is a
  runtime-checkable `typing.Protocol` — so an external model (JCM, Veros) is
  adapted by a thin wrapper class rather than being rewritten or
  monkey-patched. Its state travels in a **carry**: a pytree, by convention a
  `dict` with `state`/`forcing`/`derived`/`params` sections.
- An **exchanger** is a plain function
  `(dict[str, carry], CouplingTime) -> dict[str, carry]`. It is the only
  place where components exchange information; there is no hidden flux bus.
  Most exchanges only move fields, and are written as a declarative table
  instead of a function.
- The **`Coupler`** owns the coupled model *and its clock*: a carried
  `jax_datetime.Datetime`, advanced by the coupling timestep every step and
  never derived from a step count multiplied out. It runs an ordered
  `workflow` naming components and exchangers once per coupling timestep,
  hands every component the same `CouplingTime`, and turns that step into a
  pure trajectory function with `generate_trajectory_function(iterations)`,
  driven by `jax.lax.scan`. A `Coupler` itself satisfies `Component`, so a
  coupled model can be a component of a slower one, with no wrapper class.

## How one coupled step runs

The default workflow is every exchanger (in insertion order) followed by
every component (in insertion order), so information is exchanged first and
every component then sees the same exchanged state — which makes coupling
**lagged** by one step under the default order. `Coupler.generate_step_function()`
runs one such pass; `generate_trajectory_function(iterations)` scans it, and
because the clock lives in the carry, calling the returned function again on
its own output continues the run rather than restarting it. An element may
be listed more than once in the workflow, to run a component on a faster
clock than the rest of the model (a fast atmosphere/land loop inside a daily
ocean coupling, say) with no second `Coupler` — the same pattern a *nested*
`Coupler` expresses as an object in its own right.

```python
coupler = Coupler(
    {"atm": atm, "ocn": ocn},
    default_exchangers({"atm": atm, "ocn": ocn}),
    coupling_timestep=jdt.to_timedelta(1, "day"),
    start_date=start_date,
)
carry = coupler.initialize()
trajectory = coupler.generate_trajectory_function(30)
carry, diagnostics = trajectory(carry)
datasets = coupler.to_xarray(diagnostics)
```

## Where each topic lives

| Doc | Covers |
|---|---|
| {doc}`carry_and_clock` | The carry convention, process vs. initial-condition parameters, the component contract, the clock, the scan loop |
| {doc}`exchange` | Exchangers, the declarative exchange table, regridding, the coupling lag |
| {doc}`nesting` | Workflow nesting and multiplicity, nesting one `Coupler` inside another |
| {doc}`output` | Dimensions, coordinate and time conventions, variable naming and roles |
| {doc}`running` | `run_chunked`: chunking, the health gate, checkpointing and resume, output files, in-scan reductions |
| {doc}`configuration` | The Hydra config groups, jax-gcm's groups re-rooted under `atmosphere`, the wiring-only rule |
| {doc}`jcm_adapter` | The pinned jax-gcm revision and integration points; `JCMComponent`, the surface-exchange reader |

## Adding a new component

See {doc}`../adding_a_component` for the full worked walkthrough (built
around `JCMComponent`); the same shape applies to any model.

1. Write the class (or a wrapper class for an external model) under
   `jem/components/`. Give it a `name`, an `initialize()` and a
   `step(carry, time)`; follow the `state`/`forcing`/`derived` carry
   convention so exchangers stay readable, and put tunables in a
   `flax.struct` parameters dataclass carried as `carry["params"]` so they
   stay differentiable.
2. Keep `initialize()` pure with respect to `self`: load boundary data in
   `__init__` (it is configuration, not state). If a parameter is an
   *initial condition* rather than a process parameter, give it the
   `initialize(params=None)` signature the slab models have (see
   {doc}`carry_and_clock`'s *Parameters* section).
3. Add `bind(...)` if the model has an internal timestep, and raise
   `ValueError` when the coupling timestep does not divide it. Add
   `to_xarray(diagnostics, time)` if it produces output, and
   `save_carry`/`load_carry` if its carry cannot be checkpointed as a plain
   pytree.
4. Export it from `jem/components/__init__.py` (lazily, via the module's
   `__getattr__`, if it pulls in an optional dependency — as Veros does).
5. Register it: `Coupler({"mycomp": MyComponent(...)}, ...)`. If it is one
   of the standard surface components, register it under the name
   `default_exchanges` wires (`ocn`, `lnd`, `seaice`) and the standard
   coupling applies with no table of your own; otherwise give `Exchange` the
   rows it needs, or write an exchanger (see {doc}`exchange`).
6. To make it configurable, add a `jem/config/<group>/<option>.yaml` naming
   it as `_target_` — wiring only, no parameter defaults — and, if it is a
   new *kind* of component, one line in `jem.runners.GROUP_TO_NAME`. Nothing
   else in the runner changes.
7. Add tests under `tests/unit/`, including a two-step run through
   `Coupler.generate_trajectory_function(2)` — a component-only test cannot
   catch a carry-structure mismatch, which only `lax.scan` sees.
