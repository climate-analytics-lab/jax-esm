# Architecture

How JEM couples black-box components. This is the reference for developers
adding a component or debugging an exchange; the user-facing walkthrough is
{doc}`../tutorial`.

## Core concepts

### Carry

Every component owns a **carry**: a pytree (by convention a plain `dict`) that
holds everything passed from one coupling step to the next. The convention every
built-in component follows is:

```python
{
    "state":   <the component's prognostic state>,
    "forcing": <what other components send in>,
    "derived": <diagnostics other components read out>,
}
```

The split is not enforced by the coupler — a carry can be any pytree — but it is
what makes a mapper readable, because a mapper only ever moves a `derived` (or
`state`) field of one component into a `forcing` field of another.

The carry holds more than the mathematical state. It also holds (1) anything
that must participate in differentiability, such as forcing fields and tunable
parameters, and (2) quantities that are cheap to keep but expensive to
rediagnose, such as the surface turbulent fluxes.

`Coupler` holds a **`CoupledCarry`**, which is just `dict[component_name, carry]`.

### Workflow

A **workflow** is an ordered list (or any nested pytree) of strings driving one
coupling timestep. Each entry is either a component name — run that component's
step function on its slice of the coupled carry — or a mapper name — call that
mapper on the whole coupled carry. For example:

```python
workflow = ["mapper", "atm", "ocn", "seaice"]
```

Order is the coupling scheme: everything here is explicit and sequential within
a step, so where a mapper sits decides whether a component sees this step's
fluxes or the previous step's.

### Mappers

A **mapper** is any callable `CoupledCarry -> CoupledCarry`. It is the only
mechanism for exchanging information between components:

```python
def mapper(coupled_carry):
    atm, ocn = coupled_carry["atm"], coupled_carry["ocn"]
    ocn["forcing"].total_heat_flux = atm["derived"].total_heat_flux
    atm["forcing"].sea_surface_temperature = ocn["state"].sea_surface_temperature
    return coupled_carry
```

A mapper is traced along with everything else, so it must be pure with respect
to array values and must not change the pytree *structure* of the carry — the
coupler warns when the structure changes after a workflow element runs, and
`lax.scan` will reject it outright.

### The scan loop

`Coupler.run()` builds one step function from the workflow and drives it with
`jax.lax.scan` over `jnp.arange(iterations)`, returning
`(initial_carry, final_carry, predictions)`. Passing `jitted=False` swaps in
`adhoc_scan` — a Python `for` loop in `coupler.py` — which is for debugging only
and is not equivalent in performance or tracing behaviour.

## The interface contract

`Coupler.add_component()` calls `resolve_interface()` (`jem/base/interface.py`)
to bind a raw object's methods into a `JEMComponent` wrapper. No inheritance is
required.

Required on a component:

| Method | Signature | Purpose |
|---|---|---|
| `initialize()` | `() -> ComponentCarry` | Return the initial carry |
| `generate_step_function()` | `() -> StepFunction` | Return the step function |

`StepFunction` has signature
`(ComponentCarry, SimulationTime) -> (ComponentCarry, Predictions)`. The carry it
returns must match the structure `initialize()` produced. The leading axis of
every `Predictions` leaf is time; the coupler stacks predictions across steps
along it.

Optional (bound to `None` if absent): `predictions_to_xarray(predictions) ->
xr.Dataset` and `get_info() -> dict`.

A class whose methods have different names can supply a
`__JEM_CUSTOMIZED_MAPPING__` dict remapping them onto the expected names, so a
third-party object can be adapted without subclassing.

### Name uniqueness

Component names and mapper names share one namespace.
`Coupler._verify_name_uniqueness()` runs before every step-function build, so a
clash is caught at build time rather than producing a silently ignored mapper.

## The JCM adapter

`jem/components/jcm_component.py` adapts a `jcm.model.Model` (the spectral
atmosphere from jax-gcm). It does not subclass: `make_jem_compatible(model,
coupling_timestep)` attaches the four JEM methods to the model instance, after
checking that the coupling timestep is an integer multiple of JCM's own
timestep. Its carry is:

```python
{
    "state":   <jcm modal (spectral) state>,
    "forcing": <jcm ForcingData>,
    "derived": JCMDerived(
        physics,                # jcm's own physics carry, opaque passthrough
        ...,                    # surface heat fluxes, W/m^2, positive upward
        total_freshwater_flux,  # kg/m^2/s, positive upward (evap - precip)
    ),
}
```

Each coupling step calls `model.run_from_state_with_carry()` with the coupling
interval as both `save_interval` and `total_time`, then reads the surface fluxes
out of the returned physics diagnostics and flips their sign (JCM publishes
downward positive; JEM is upward positive). The exact set of flux fields on
`JCMDerived` tracks what jax-gcm publishes and is changing — see
`api_hardening_plan.md` T0.2 and T1.2.

## Adding a new component

1. Write the class (or an adapter for an external model) under
   `jem/components/`.
2. Implement `initialize()` and `generate_step_function()`; follow the
   `state`/`forcing`/`derived` carry convention so mappers stay readable.
3. Optionally add `predictions_to_xarray()` and `get_info()`.
4. Export it from `jem/components/__init__.py` (lazily, via the module's
   `__getattr__`, if it pulls in an optional dependency — as Veros does).
5. Register it: `Coupler(components={"mycomp": MyComponent(...)})`.
6. Add tests under `tests/unit/`, including a two-step `Coupler.run()` that
   exercises the component inside a workflow — a component-only test cannot
   catch a carry-structure mismatch.
