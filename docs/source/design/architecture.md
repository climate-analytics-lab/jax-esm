# Architecture

How JEM couples black-box components. This is the reference for developers
adding a component or debugging an exchange; the user-facing walkthrough is
{doc}`../tutorial`.

Every statement here is checkable against `jem/base/component.py` and
`jem/base/coupler.py`, which together are the whole of the coupling core.

## Core concepts

### Carry

Every component owns a **carry**: a pytree (by convention a plain `dict`) that
holds everything passed from one coupling step to the next. The coupler never
looks inside it, but every built-in component follows the same convention:

```python
{
    "params":  <the component's tunable parameters>,   # slab models
    "state":   <the component's prognostic state>,
    "forcing": <what other components send in>,
    "derived": <diagnostics other components read out>,
}
```

The split is not enforced — a carry can be any pytree — but it is what makes an
exchanger readable, because an exchanger only ever moves a `derived` (or
`state`) field of one component into a `forcing` field of another.

The carry holds more than the mathematical state. It also holds (1) anything
that must participate in differentiability and (2) quantities that are cheap to
keep but expensive to rediagnose:

- The four slab models put their `flax.struct` parameters in `carry["params"]`
  rather than closing over them, so `jax.grad` of a coupled run with respect to,
  say, `SlabOceanParameters.relaxation_time` works with no special casing in the
  coupler.
- `JCMComponent`'s carry has a fourth key, `"physics"`: JCM's cross-step physics
  carry (sub-cycled radiation, prior-step TKE, the tendencies one term hands to
  the next). It is threaded straight back into
  `Model.run_from_state_with_carry`, because dropping it between coupling steps
  would reset that memory once per coupling interval — a silent, systematic
  error. It contains integer and boolean leaves, so it must never be cast
  wholesale to a float dtype.
- `JCMComponent`'s `carry["derived"]` is a `JCMDerived` struct holding the
  surface exchange (`total_heat_flux`, `total_freshwater_flux`, `evaporation`,
  `precipitation`, `u0`, `v0`) plus `physics`, JCM's own per-step diagnostics
  dict, carried opaquely so an exchanger can reach any field JCM computes.

The coupler's own state is a **`CoupledCarry`** (`flax.struct.dataclass`):

```python
@struct.dataclass
class CoupledCarry:
    components: dict[str, Carry]   # one carry per component, keyed by name
    step: jax.Array                # int32; coupled steps completed
```

`step` lives in the carry rather than in the `lax.scan` index because the scan
index restarts at zero on every call: putting the clock in the carry is what
makes a chunked run (or a restart from a checkpoint) continue the same
simulation instead of replaying the first year.

For the same reason `step` is part of a **checkpoint**.
`jem.utils.checkpoints.save_coupled_carry(coupled_carry, directory)` writes one
`{name}_carry.pkl` per component — or a subdirectory, for a component that
implements `SupportsCheckpoint` and is passed in `component_savers` — plus
`coupled_step.pkl` holding the counter, and `load_coupled_carry` returns the
`CoupledCarry` a trajectory function can be handed straight back. A checkpoint
directory with no `coupled_step.pkl` is refused with a `ValueError`: its
position in the seasonal cycle is not recoverable, and resuming at step 0 (or
at a step reconstructed from a batch index) would silently move the run's
calendar. `save_component_carries` / `load_component_carries` are the
mapping-only halves, for a sub-carry that has no clock of its own — which is
how the Veros restart writer stores its `derived` and `forcing` structs.

### The component contract

`jem.base.component.Component` is a runtime-checkable `typing.Protocol`, so
"implementing" it means having the right attributes — there is no base class to
inherit from and nothing is monkey-patched onto the wrapped model:

| Member | Signature | Purpose |
|---|---|---|
| `name` | `str` | The component's name in the workflow, carry and output |
| `initialize()` | `() -> Carry` | Build the initial carry. Must not integrate |
| `step(carry, time)` | `(Carry, CouplingTime) -> (Carry, Diagnostics)` | Advance one coupling step |

`Coupler.add_component(name, component)` checks `isinstance(component,
Component)` and raises `TypeError` naming the missing members. The object itself
is stored, so `coupler.components[name] is component`.

Three capabilities are **optional**, and are tested for with `isinstance`
against their protocols at the one place that uses them — never with `hasattr`
at a random call site:

| Protocol | Member | Who implements it |
|---|---|---|
| `SupportsXarray` | `to_xarray(diagnostics, time) -> xr.Dataset` | slab models, `JCMComponent`, `VerosComponent` |
| `SupportsBind` | `bind(*, coupling_timestep, start_date, calendar)` | `JCMComponent`, `VerosComponent`, the slab models |
| `SupportsCheckpoint` | `save_state(carry, directory)` / `load_state(directory)` | `VerosComponent` |

`bind` is called by the coupler once per component, from `add_component` (hence
from the constructor for everything passed to it), and it is the only way a
component learns anything about the clock outside a step. A component with an
internal timestep uses it for the coupling interval — `JCMComponent` converts it
to the number of days it passes to JCM as `save_interval`/`total_time`,
`VerosComponent` to a count of tracer timesteps — and it is where a disagreement
about the clock is refused: both wrappers raise `ValueError` if the coupling
timestep is not a whole multiple of the model's own, and `JCMComponent`
additionally refuses a `start_date` or `calendar` that differs from the model's.
The slab models use it for the start date alone: `initialize()` takes no
argument, so `bind` is how a run starting on 1 July samples the July record of
its climatology rather than the January one. It reaches them as
`SlabModelBase.start_year_fraction`, computed by the shared
`jem.base.component.start_year_fraction(start_date, calendar)` — the same
function behind `CouplingTime.year_fraction`, so a climatology sampled in
`initialize()` and one sampled in `step()` cannot disagree about where the run
starts. A model that was never registered with a coupler reads 1 January, which
is what a bare `model.initialize()` in a test or a notebook gets.

`step` must be a pure function of `(carry, time)` and must return a carry with
exactly the pytree structure, shapes and dtypes it received, or `lax.scan`
rejects it.

### The clock

The coupler owns the only clock. Components hold no start date, no timestep and
no calendar of their own, so two of them cannot disagree about the date. Each
`step` is handed a `CouplingTime` built from `CoupledCarry.step`:

```python
@struct.dataclass
class CouplingTime:
    step: jax.Array          # int32, coupled steps completed before this one
    sim_time: jax.Array      # seconds since start_date; equals step * dt
    dt: float                       # static: coupling timestep in seconds
    year_offset_seconds: float      # static: 1 Jan of the start year -> start_date
    days_per_year: float            # static: jcm.date.days_per_year(calendar)
```

- `time.year_fraction` is the position in the annual cycle in `[0, 1)` at the
  *start* of the step; it is what a monthly climatology is interpolated with
  (`jem.utils.cycles.evaluate_cyclic_linear`). When the coupling step divides
  the year exactly — the usual case, daily steps in a 365-day year — the step
  count is reduced modulo the steps per year in exact integer arithmetic before
  the division, so a float32 `sim_time` cannot quantise the seasonal cycle away
  in a century-long run.
- `time.end_of_step()` returns the clock one step later, advancing `step` and
  `sim_time` together. A model that needs a boundary condition at both ends of a
  step (the slab models measure an anomaly against the climatology at the start
  and add it back at the end) must use it rather than adding `dt` to `sim_time`
  by hand, because `year_fraction` is derived from `step`.

The static fields are resolved once, in the coupler's constructor, so no
calendar arithmetic happens inside a traced function.

### Exchangers

An **exchanger** is the only mechanism for exchanging information between
components:

```python
Exchanger = Callable[[dict[str, Carry], CouplingTime], dict[str, Carry]]
```

It receives the mapping of every component's carry and the clock, and returns
the mapping to continue with. The clock is passed so a time-dependent coupling
(lagged exchange, ramped forcing) needs no state of its own.

```python
def atm_ocn_exchange(components, time):
    del time
    atm, ocn = components["atm"], components["ocn"]
    ocn = dict(ocn, forcing=ocn["forcing"].replace(
        total_heat_flux=atm["derived"].total_heat_flux))
    atm = dict(atm, forcing=atm["forcing"].replace(
        sea_surface_temperature=ocn["state"].sea_surface_temperature))
    return dict(components, atm=atm, ocn=ocn)
```

Two rules, both enforced by what `lax.scan` will accept:

1. **Do not mutate in place.** The carries handed to an exchanger are the ones
   the scan is carrying. Build new structs (`.replace(...)` on a `tree_math` or
   `flax.struct` struct, `dataclasses.replace`, a new `dict`) and return them.
   The coupler passes a *fresh* dict, so adding or replacing entries cannot
   reach the caller's carry, but the structs inside it are shared.
2. **Do not change the pytree structure.** After every workflow element the
   coupler compares the structure of the carries dict with the structure it had
   on entry and raises `RuntimeError` naming the element responsible. The check
   is at trace time, so it costs nothing per step and turns an opaque `lax.scan`
   error into a located one.

These were called "mappers" before v1.0. The name changed because "mapper" reads
as a regridding operation, whereas an exchanger may regrid, compute a flux,
convert units or simply copy a field.

### Workflow and the coupled step

A **workflow** is an ordered tuple of names driving one coupling timestep. Each
entry is either a component name — run that component's `step` on its carry and
record its diagnostics — or an exchanger name — call it on the whole mapping.
Components and exchangers share one namespace: a name may not be used twice, and
each may appear at most once in a workflow.

The default is every exchanger (in insertion order) followed by every component
(in insertion order):

```python
coupler.workflow  # ("atm_ocn_exchange", "atm", "ocn")
```

so information is exchanged first and every component then sees the same
exchanged state. Coupling is therefore **lagged**: the exchanger at step *n*
moves the fields the components produced during step *n-1*, and the first step
of a run exchanges the values that came out of `initialize()`. Moving a
component ahead of the exchanger in an explicit `workflow=` is what changes
that.

### The scan loop

`Coupler.step_function()` returns the pure function `CoupledCarry ->
(CoupledCarry, dict[str, Diagnostics])` that runs one workflow pass and returns
the carry with `step` incremented. It snapshots the components and exchangers as
they stand when it is called, so registering a component afterwards cannot
silently change an already-compiled step.

`Coupler.generate_trajectory_function(iterations, *, remat=False, jit=True)`
drives that step with `jax.lax.scan` over `iterations` steps and no `xs` (the
steps are identical; the only per-step input, the clock, comes from the carry).
It returns `carry -> (final_carry, diagnostics)`, where every diagnostics leaf
has gained a leading axis of length `iterations`. `remat=True` wraps the step in
`jax.checkpoint`, trading recomputation for memory when differentiating through
a long trajectory; `jit=False` leaves the scan unjitted.

Because the clock is the carry's own `step`, calling the trajectory function
again on the returned carry continues the run:

```python
run = coupler.generate_trajectory_function(30)
carry = coupler.initialize()
for chunk in range(12):
    carry, diagnostics = run(carry)
    datasets = coupler.to_xarray(diagnostics, first_step=chunk * 30)
```

## Output conventions

`Coupler.to_xarray(diagnostics, *, first_step=0)` returns one
`xarray.Dataset` per component that implements `SupportsXarray`; components that
do not are skipped, so an output-less component does not stop a run producing
output. `first_step` is the coupled step the first record covers — the `step` of
the carry the trajectory started from — and defaults to 0. **Pass it when
writing a chunked run**, or the second chunk is labelled with the first chunk's
dates.

Each component is handed a `TimeAxis` (start date, the record's coupled-step
indices, the timestep and the calendar) so every dataset from one run shares one
time coordinate. The conventions, which are JCM's:

- **Dimensions** are `("time", "lon", "lat")` for a separable lon/lat grid, and
  `("time", "x", "y")` with 2-D auxiliary `lat`/`lon` coordinates (and a CF
  `coordinates` attribute on each variable) for a curvilinear one — CF and
  xarray forbid a 2-D variable named after one of its own dimensions.
- **Coordinate values** are degrees computed as `radians * 180 / pi`, in
  float64, which is character for character what `jcm.utils.data_to_xarray`
  does. A last-bit difference would be enough for `xr.merge` to treat two
  96-point longitude axes as different axes and produce a 119-point union.
- **The time label is the END of the interval a record covers**, as an absolute
  `datetime64[ns]`: record *k* holds the average over
  `[start_date + k dt, start_date + (k+1) dt)` and is stamped
  `start_date + (k+1) dt`. This is JCM's convention, and `TimeAxis.datetimes()`
  is the one place it is written down — including the arithmetic, a float64
  count of days since the epoch multiplied into nanoseconds at the end, which
  is inexact but *identically* inexact for every component that goes through
  it. `jem.utils.time.time_coordinate` is the slab-side call site that unpacks
  it (values plus `TimeAxis.attrs`) for xarray. The dates are proleptic
  Gregorian whatever the model calendar is; the calendar governs the seasonal
  cycle and forcing selection, not the labels.
- **Variable names**: state and derived quantities keep their plain names, and
  every variable that came from a component's *forcing* is written with a
  `forcing_` prefix — `jem.components.slab.base.FORCING_VARIABLE_PREFIX`,
  applied by `forcing_variable(name)`, which is what a slab model's
  `_create_xarray_data_vars` calls. Two components legitimately hold the same
  physical field — one produced it, the other received it — and without the
  prefix the merge collides on the shared name. So the slab atmosphere and the
  slab land model write `forcing_total_heat_flux` while the ocean writes its
  own derived `total_heat_flux`; the sea ice writes
  `forcing_ice_frazil_melt_energy` for the field the ocean published as
  `ice_frazil_melt_energy`.

Together these are what make `xr.merge([datasets["atm"], datasets["ocn"]])` an
N-long join rather than a 2N-long outer union.

## The JCM adapter

`jem/components/jcm/component.py` wraps a `jcm.model.Model` (the spectral
atmosphere from jax-gcm) as `JCMComponent`. It is a wrapper object, not an
in-place adaptation: the atmosphere JEM drives is the same object the user
configured, and nothing in JCM has to know JEM exists. Its carry is:

```python
{
    "state":   <jcm modal (spectral) dycore state>,
    "physics": <jcm's cross-step physics carry, threaded, opaque>,
    "forcing": <jcm ForcingData; holds sea_surface_temperature, sice_am, ...>,
    "derived": JCMDerived(physics, total_heat_flux, total_freshwater_flux,
                          evaporation, precipitation, u0, v0),
}
```

`initialize()` builds those pytrees from `Model.bootstrap_state()` and a
structural template of the diagnostics dict; it does **not** integrate. Each
`step` calls `model.run_from_state_with_carry()` with the coupling interval as
both `save_interval` and `total_time`, so JCM sub-steps internally at its own
timestep and returns exactly one saved record per coupling step, then reads the
surface exchange out of the returned physics diagnostics.

That read is isolated in `jem/components/jcm/exchange_fields.py`, which is the
single place JCM's package-specific diagnostics layout is translated into JEM's
conventions — heat flux **positive upward** (JCM publishes `hfluxn` downward
positive, so it is negated exactly here), water fluxes in `kg m-2 s-1` (JCM's
SPEEDY reports `g m-2 s-1`), wind in `m s-1`. `detect()` picks the reader from
the diagnostics keys; the ECHAM reader raises `NotImplementedError` naming
jax-gcm#754, the issue that will have every JCM physics package publish the same
surface-exchange struct.

Three JCM private attributes are still read, each in one helper tagged with the
jax-gcm issue that will remove it: `_final_dycore_state` and
`_final_physics_state` (jax-gcm#755, a public initial-state / physics-carry
API), and `ModelPredictions._predictions` (jax-gcm#756,
`ModelPredictions.with_context`). The atmosphere's output keeps JCM's own time
labelling because JEM cannot reproduce its calendar arithmetic while
`Model._date_from_sim_time` is private (jax-gcm#758).

Each `step` also compares the dycore state's own `sim_time` with the coupler's
and logs at ERROR if they have parted, which can only happen if the carry came
from another run. The tolerance is `clock_tolerance_seconds(sim_time)` — one
second, or eight float32 ulps of the elapsed time, whichever is larger — so the
check neither fires on the rounding of a long run's float32 clock nor stops
noticing a real disagreement.

## Adding a new component

1. Write the class (or a wrapper class for an external model) under
   `jem/components/`. Give it a `name`, an `initialize()` and a
   `step(carry, time)`; follow the `state`/`forcing`/`derived` carry convention
   so exchangers stay readable, and put tunables in a `flax.struct` parameters
   dataclass carried as `carry["params"]` so they stay differentiable.
2. Keep `initialize()` pure with respect to `self`: load boundary data in
   `__init__` (it is configuration, not state), so calling `initialize()` twice
   gives the same answer.
3. Add `bind(...)` if the model has an internal timestep, and raise `ValueError`
   when the coupling timestep does not divide it. Add `to_xarray(diagnostics,
   time)` if it produces output, and `save_state`/`load_state` if its carry
   cannot be checkpointed as a plain pytree.
4. Export it from `jem/components/__init__.py` (lazily, via the module's
   `__getattr__`, if it pulls in an optional dependency — as Veros does).
5. Register it: `Coupler({"mycomp": MyComponent(...)}, ...)`.
6. Add tests under `tests/unit/`, including a two-step run through
   `Coupler.generate_trajectory_function(2)` — a component-only test cannot
   catch a carry-structure mismatch, which only `lax.scan` sees.
