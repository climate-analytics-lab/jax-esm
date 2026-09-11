# Architecture

How JEM couples black-box components. This is the reference for developers
adding a component or debugging an exchange; the user-facing walkthrough is
{doc}`../tutorial`.

Every statement about the coupling core is checkable against
`jem/base/component.py` and `jem/base/coupler.py`, which are the whole of it;
the layers built on that core — the declarative exchange (`jem/exchangers.py`),
the run loop (`jem/driver.py`, `jem/output.py`, `jem/checkpoint.py`) and the
configuration (`jem/config/`, `jem/runners.py`, `jem/main.py`) — have a section
each below.

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
  coupler. Which of them can be varied *through the carry* is the subject of
  the next section.
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

For the same reason `step` is part of a **checkpoint**. A checkpoint directory
holds one `carry.msgpack` — every component that does not write itself, plus the
coupled step counter — beside one subdirectory per component that writes itself
(`VerosComponent`, a nested `Coupler`). A directory with no `carry.msgpack` is
refused with a `ValueError`: its position in the seasonal cycle is not
recoverable, and resuming at step 0 (or at a step reconstructed from a batch
index) would silently move the run's calendar.

The format is jax-gcm's: `jem.checkpoint.save(carry, path)` flattens any pytree
to a list of typed arrays serialised with flax's msgpack codec, and
`load(template, path)` rebuilds the tree from a *template*'s treedef. The
structure is not stored at all, which is what makes the format both small and
self-checking — every leaf is compared with the template's path, shape and
dtype, so a checkpoint written by another grid or another component composition
fails naming the leaf instead of deserialising into something that only explodes
later inside a `lax.scan`. The leaf paths are stored because leaf count, shape
and dtype together cannot tell two same-shaped carries apart, and silently
swapping two components' carries on resume is the failure that would follow.
Both the stored leaf and the template go through `jnp.asarray` first: a
component's parameter default is a Python float in `initialize()` and a float32
array in the carry `lax.scan` returns, and the two have to compare as one leaf.
It is also why `load_state` needs a template at all, and takes it from each
component's own `initialize()`.

**The coupler is what a driver checkpoints through**, in one call each way:

```python
model.save_state(final_carry, checkpoint_dir / f"step_{int(final_carry.step):08d}")
carry = model.load_state(saved)
```

Which components need writing by hand rather than pickling is a property of the
*components* — `VerosComponent` has to go through Veros' HDF5 restart writer
because a `VerosState` is not a pytree — and the coupler is the one object that
knows them all. `Coupler.save_state` therefore derives the savers itself, as
`{name: component.save_state for … if isinstance(component, SupportsCheckpoint)}`,
and `load_state` derives the loaders from the same capability. A driver never
enumerates them, so it cannot get the set wrong or forget one when a component
is swapped.

A `Coupler` implements `SupportsCheckpoint` itself, so this **recurses**: an
outer coupler sees a nested coupler as a component that writes itself, hands it
`directory / <the name it is registered under>`, and the inner coupler writes
its own components and its own `carry.msgpack` there — an inner checkpoint that
is complete in its own right, holding the inner clock. Without that, the outer
save would treat the inner `CoupledCarry` as a plain pytree and serialise it,
which works only while nothing inside it needs a format of its own; a nested
model containing Veros would silently bypass the restart path.

`jem.checkpoint.save_coupled(coupled_carry, directory, component_savers=…)` and
`load_coupled(directory, component_templates, component_loaders=…)` are the
underlying functions and still take the mappings explicitly, for a caller that
wants to override a saver or supply one for something that is not a component
capability. `Coupler.save_state` / `load_state` are the answer for every
ordinary case.

**The carry file is the marker**, and it is written last — published by renaming
a flushed, fsynced temporary over its final name, and removed first when an
existing checkpoint is overwritten, so a failure part-way through cannot leave a
stale clock beside freshly written component data. A separate `COMPLETE` file
would be a second thing to keep in step for no gain: a resume cannot do without
the clock, which lives in the carry file anyway, so that file's presence is
exactly the condition "this checkpoint can be resumed from".

A save interrupted part-way through therefore leaves a directory with no marker
— and, in a run that names its checkpoints `step_00000000`, `step_00000005`, …,
that directory sorts *newest*. A driver resuming from a directory of checkpoints
asks `latest_complete_checkpoint(root, pattern="step_*")` for the newest one
that holds a marker rather than the newest name; it warns about each incomplete
directory it steps over, since one means an earlier run died mid-save.
`jem.driver.run_chunked` keeps a single checkpoint directory instead (see
*Running a model*) and makes the same check on it.

A checkpoint is named after the coupled step it was written at, and how much of
a run is left is computed from the step counter *inside* the restored
checkpoint — `remaining_batches(steps_done, total_steps, steps_per_batch)`
returns the length of each batch still to run, the last one short when the
total is not a whole number of batches. A batch index in the name would mean
nothing across two runs that chose different batch lengths, whereas the coupled
step counts the same coupling steps in both.

### Parameters: process and initial-condition

A component's parameters divide into two kinds, and the difference decides how
a parameter study varies one. It is not a distinction the framework enforces —
it follows from *when* the parameter is read:

| | Read by | Varied by | Example |
|---|---|---|---|
| **Process parameter** | `step`, out of `carry["params"]`, every step | replacing that leaf in the carry | `SlabOceanParameters.relaxation_time`, `SlabLandParameters.tdland` |
| **Initial-condition parameter** | `initialize`, once | passing parameters to `initialize` | `SlabOceanParameters.initial_sst`, `SlabSeaiceParameters.initial_ice_thickness`, every field of `SlabAtmosphereParameters` |

A process parameter is varied in the carry, because that is where `step` reads
it from:

```python
carry = model.initialize()
carry["params"] = carry["params"].replace(relaxation_time=tau)   # differentiable
```

An initial-condition parameter **cannot** be: by the time a carry exists its
value has already been copied into the state, and `step` never looks at it
again, so replacing the leaf changes nothing and a gradient with respect to it
is zero. It is varied by handing the parameters to `initialize`, which builds
the initial state from them *and* puts them in `carry["params"]`, so the state
and the process parameters come from one object:

```python
# `coupled` is the Coupler; `ocn` the SlabOceanModel registered in it.
def loss(initial_sst):
    params = ocn.params.replace(initial_sst=initial_sst)
    _, diagnostics = trajectory(coupled.initialize({"ocn": params}))
    return jnp.mean(diagnostics["ocn"]["state"].sea_surface_temperature)

jax.grad(loss)(jnp.float32(288.15))     # non-zero
```

`Coupler.initialize(params)` takes `{component name: that component's
parameters}` and routes each one to that component's `initialize(params=…)`; a
component the mapping does not name is initialized exactly as it is without the
argument, `Coupler.initialize()` with no argument is unchanged, and a name the
coupler has no component for is a `ValueError`. Not every component can take
parameters — a wrapper around an external model initializes from that model's
own state — so naming one that cannot is a `TypeError` rather than a silently
ignored request. For a **nested** coupler the value is itself a mapping over
its components (`{"atm_lnd": {"atm": params}}`), because that is what its own
`initialize` takes.

Building the model inside `jax.grad` is not an alternative route to the same
gradient: a constructor validates its parameters, which means reading them as
concrete Python floats, and that cannot be done to a traced value. Validation
therefore stays at construction, where the values are concrete, and
`initialize(params)` is the differentiable entry point, which uses what it is
given untouched.

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

`initialize()` must be callable with no arguments — that is all the protocol
asks. A component may additionally accept `initialize(params=…)`, which is what
`Coupler.initialize({name: params})` calls and how an initial-condition
parameter is varied (see *Parameters*); the slab models and `Coupler` itself do,
`JCMComponent` and `VerosComponent` do not, because they initialize from the
wrapped model's own state.

Three capabilities are **optional**, and are tested for with `isinstance`
against their protocols at the one place that uses them — never with `hasattr`
at a random call site:

| Protocol | Member | Who implements it |
|---|---|---|
| `SupportsXarray` | `to_xarray(diagnostics, time) -> xr.Dataset \| Mapping[str, xr.Dataset]` | slab models, `JCMComponent`, `VerosComponent`, `Coupler` |
| `SupportsBind` | `bind(*, coupling_timestep, start_date, calendar)` | `JCMComponent`, `VerosComponent`, the slab models |
| `SupportsCheckpoint` | `save_state(carry, directory)` / `load_state(directory)` | `VerosComponent`, `Coupler` |

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

`to_xarray` normally returns one dataset, keyed in the coupler's output by the
name the component is registered under. A component that is itself a coupled
model — a `Coupler` nested in a slower one — has no single dataset to return, so
it may return a **mapping** of name to dataset, which the outer coupler flattens
into its result under those names (a collision with a name already there is a
`ValueError`). `Coupler` is the implementation of that case; see *Nesting
couplers*.

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

   The risk is easy to miss because it only shows outside `jit`. This
   exchanger runs and gives the right trajectory under the jitted scan:

   ```python
   def exchange_in_place(components, time):
       ocn, seaice = components["ocn"], components["seaice"]
       seaice["forcing"].ice_frazil_melt_energy = ocn["derived"].ice_frazil_melt_energy
       return components
   ```

   Inside `generate_trajectory_function` the carries are tracers, so the
   assignment cannot reach the caller's arrays and `carry0` is untouched.
   But run one step eagerly (`model.generate_step_function()(carry0)`, the natural thing
   to do when debugging, checking a gradient or comparing two workflows from
   one initial condition) and the struct being assigned into *is*
   `carry0.components["seaice"]["forcing"]`: the initial carry is silently
   overwritten, and the next run from `carry0` starts somewhere else. The
   same exchanger written as the contract asks,

   ```python
   def exchange(components, time):
       ocn, seaice = components["ocn"], components["seaice"]
       seaice = dict(seaice, forcing=seaice["forcing"].replace(
           ice_frazil_melt_energy=ocn["derived"].ice_frazil_melt_energy))
       return dict(components, seaice=seaice)
   ```

   behaves the same both ways. `tests/unit/test_coupler.py` pins this
   asymmetry (`test_in_place_exchange_corrupts_the_initial_carry_eagerly`),
   so the rule is not just advice.
2. **Do not change the pytree structure.** After every workflow element the
   coupler compares the structure of the carries dict with the structure it had
   on entry and raises `RuntimeError` naming the element responsible. The check
   is at trace time, so it costs nothing per step and turns an opaque `lax.scan`
   error into a located one.

These were called "mappers" before v1.0. The name changed because "mapper" reads
as a regridding operation, whereas an exchanger may regrid, compute a flux,
convert units or simply copy a field.

An exchange that only *moves* fields — which is most of a coupled model — does
not need to be written as a function at all; see *The declarative exchange*
below.

### Workflow and the coupled step

A **workflow** is an ordered tuple of names driving one coupling timestep. Each
entry is either a component name — run that component's `step` on its carry and
record its diagnostics — or an exchanger name — call it on the whole mapping.
Components and exchangers share one namespace, so one name may not be both; a
name may, however, appear in the workflow more than once (see *Multiplicity*
below).

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

#### Nesting

An explicit `workflow=` may be an arbitrarily **nested** sequence of names. The
nesting is notation only — it is flattened at construction and
`Coupler.workflow` is always the flat tuple actually executed — but it lets a
coupling scheme be written the way it is described:

```python
Coupler(components, exchangers,
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=start_date,
        workflow=[["atm_lnd_exchange", "atm", "lnd"] * 24,
                  "atm_ocn_exchange", "ocn"])
```

Strings are the leaves; any other leaf is a `TypeError`, because the
alternative — iterating it — would silently turn a stray object into a sequence
of characters. An unknown name is still a `ValueError` at construction.

#### Multiplicity: running a component on a faster clock

An element listed *n* times runs *n* times per coupled step, on a clock *n*
times faster. In the example above the atmosphere, the land and the exchanger
between them run hourly inside a daily ocean coupling — the GFDL-style
"fast loop" — with no second `Coupler` and no component-side sub-stepping code.

- **The sub-timestep is `coupling_timestep / n`, and must be a whole number of
  seconds.** `jdt.Timedelta` is integer-backed, so anything else would have to
  be rounded, and a rounded sub-step desynchronises the sub-stepped component
  from the coupled clock a little more every step. It is refused at
  construction, with a `ValueError` naming the element and the count.
- **A bindable component is bound with its own sub-timestep**, once — the step
  it actually advances by, not the coupled one, so a component that sub-cycles
  an internal timestep (JCM, Veros) sub-cycles the right number of times. A
  component an explicit workflow never names has multiplicity 0: it is neither
  bound nor run. A component registered *after* construction with
  `add_component` is bound with the multiplicity the current workflow gives it,
  or the full coupling timestep when nothing names it — which is the case for
  the default workflow, since that is derived from the components and the new
  one is not registered yet. Binding still happens before registering, so a
  component that rejects the clock never enters the coupler.
- **Each call gets its own clock.** The loop over the workflow is ordinary
  Python, run once at trace time, so which call this is — *k* of *n* — is a
  static number: call *k* of coupled step *s* is handed
  `Coupler.coupling_time_at_substep(s, k, n)`, whose `step` is the sub-step
  `s * n + k` (exact integer arithmetic on the int32 counter), whose `dt` is the
  sub-timestep and whose `sim_time` is `(s * n + k) * dt`. `year_fraction`
  keeps its exact integer reduction at the sub-rate too — an hourly sub-step
  still divides a 365-day year — so the seasonal cycle does not quantise away
  in a long run. Exchangers may be repeated as well and see the same clock.
- **`CoupledCarry.step` still counts coupled steps.** The sub-step count is
  derived from it, never stored, so checkpoints, resume and chunked runs are
  untouched: a checkpoint of a run with multiplicity restores the coupled
  counter and both clocks continue.
- **Diagnostics of a repeated component are stacked** along a new leading axis
  of length *n*, in the order the calls were made, so a trajectory returns
  `(steps, n, ...)` for it; `to_xarray` folds those two axes into one and
  labels the records at the sub-rate (below).

Everything about `n == 1` — the clock a component sees, the shape of its
diagnostics, its time axis, the traced operations — is exactly what it is in a
coupler with no multiplicity at all, so adding a fast loop to one part of a
model cannot perturb the rest of it.

### The scan loop

`Coupler.generate_step_function()` returns the pure function `CoupledCarry ->
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

## Nesting couplers

A `Coupler` satisfies `Component`: it has a `name` (the keyword-only
`name="coupled"` argument; the *registered* key in an outer coupler is what the
outer workflow uses), an `initialize()` returning its `CoupledCarry`, and a
`step(carry, time)`. It also implements `SupportsBind` and `SupportsXarray`. So
a coupled model can be a component of a slower coupled model with no wrapper
class — the GFDL pattern of a fast atmosphere/land loop inside a daily ocean
coupling:

```python
fast = Coupler(
    {"atm": atm, "lnd": lnd},
    {"atm_lnd_exchange": atm_lnd_exchange},
    coupling_timestep=jdt.to_timedelta(1, "hour"),
    start_date=start_date,
    name="atm_lnd",
)
model = Coupler(
    {"atm_lnd": fast, "ocn": ocn},
    {"srf_ocn_exchange": srf_ocn_exchange},
    coupling_timestep=jdt.to_timedelta(1, "day"),
    start_date=start_date,
    workflow=["srf_ocn_exchange", "atm_lnd", "ocn"],
)
```

- **`bind`** requires the outer timestep to be a whole multiple of the inner
  one, and the start date and calendar to be equal; anything else is a
  `ValueError`, as it is for any other component with an internal timestep. It
  records the ratio *r* (24 here). Binding again to the same clock is a no-op,
  to a different one a `ValueError`: one instance belongs to one coupled model.
- **`step`** runs *r* of the inner coupler's own coupled steps, through an
  unjitted trajectory (`lax.scan`, so the inner step appears once in the outer
  jaxpr rather than *r* times unrolled). The inner clock comes from the inner
  carry's own `step` counter exactly as in a standalone run, so it is
  continuous across outer steps and survives a checkpoint; the outer `time` is
  only checked against it — the static fields of a `CouplingTime` (`dt`,
  `days_per_year`, `year_offset_seconds`) are comparable at trace time, the
  step counter is a traced array. Calling `step` before `bind` is a
  `RuntimeError`. For `r == 1` the inner step is run directly and the
  diagnostics gain no extra axis, mirroring multiplicity 1.
- **The carry** of the inner coupler is a `CoupledCarry` living inside the
  outer one's `components`, so there are two step counters: the outer counts
  outer steps, the inner counts its own. `outer.save_state(carry, directory)`
  writes the inner model into `directory / <its registered name>` through the
  inner coupler's own `save_state`, and `outer.load_state(directory)` reads it
  back the same way, so a resume continues both clocks — and a component inside
  the inner model that needs its own format (Veros) still gets it.
- **Exchangers in the outer coupler** see the inner `CoupledCarry` under its
  registered name and reach inner components through `.components`.
  `jem.nested_carry(carries, outer_name, inner_name)` and
  `jem.with_nested_carry(carries, outer_name, inner_name, new_inner_carry)` are
  that read and that immutable write (`dataclasses.replace` on the inner
  `CoupledCarry`), written once:

  ```python
  def srf_ocn_exchange(components, time):
      del time
      land = nested_carry(components, "atm_lnd", "lnd")
      ocn = dict(components["ocn"], forcing=land["derived"].total_heat_flux)
      return dict(components, ocn=ocn)
  ```

- **Output.** The inner coupler's `to_xarray` returns one dataset per *its*
  components, and the outer coupler flattens them into its result under those
  names — the nested coupler's own registered name does not appear. The inner
  datasets carry the inner, faster time axis: `Coupler.to_xarray` supports both
  the run form `to_xarray(diagnostics, first_step=0)` and the component form
  `to_xarray(diagnostics, time)`, and in the second it takes `time.steps[0] * r`
  as its own first step and folds the outer coupler's leading axis of length *r*
  into the records first.

### The same model, written flat

Multiplicity expresses the same model in one coupler:

```python
model = Coupler(
    {"atm": atm, "lnd": lnd, "ocn": ocn},
    {"atm_lnd_exchange": atm_lnd_exchange, "srf_ocn_exchange": srf_ocn_exchange},
    coupling_timestep=jdt.to_timedelta(1, "day"),
    start_date=start_date,
    workflow=["srf_ocn_exchange",
              ["atm_lnd_exchange", "atm", "lnd"] * 24,
              "ocn"],
)
```

The two are **equivalent** — the same elements in the same order, on the same
clocks, producing bit-identical carries and datasets (`tests/unit/
test_nested_coupler.py::test_nested_and_flat_forms_are_the_same_run` is that
check). They differ only in bookkeeping:

| | Nested | Flat |
|---|---|---|
| Carry | two levels, two step counters | one level, one counter |
| Exchangers | outer ones go through `nested_carry` | all at one level |
| Fast loop | exists on its own: buildable, testable and runnable alone | is a rate, not an object |

Prefer the **nested** form when the fast loop is a thing in its own right — an
already-assembled surface model, something you also run standalone, or a piece
another model will reuse — and the **flat** form when it is only a rate: one
coupler, one carry and one workflow to read.

## Output conventions

`Coupler.to_xarray(diagnostics, *, first_step=0)` returns one
`xarray.Dataset` per component that implements `SupportsXarray`; components that
do not are skipped, so an output-less component does not stop a run producing
output. A component that returns a *mapping* of datasets (a nested `Coupler`)
contributes its entries under their own names, and a name that collides with one
already written is a `ValueError`. `first_step` is the coupled step the first record covers — the `step` of
the carry the trajectory started from — and defaults to 0. **Pass it when
writing a chunked run**, or the second chunk is labelled with the first chunk's
dates.

Each component is handed a `TimeAxis` (start date, the record's step indices,
the record interval and the calendar) so every dataset from one run shares one
time coordinate; `TimeAxis.datetimes()` and `TimeAxis.attrs` are the
`(values, attrs)` pair xarray wants, and every component's `to_xarray` calls
them directly.

A component the workflow runs *n > 1* times per coupled step wrote *n* records
per step, and its stacked diagnostics arrive as `(steps, n, ...)`. The two
leading axes are folded into one — they are already in time order, record
`s * n + k` being call *k* of step *s* — and the component is handed a
`TimeAxis` spaced at `coupling_timestep / n` and starting at sub-step
`first_step * n`. So `first_step` is always given in *coupled* steps, whatever
rate a component runs at, and an hourly component in a daily coupler writes 24
records per coupled step stamped at the end of each hour. Components with
`n == 1` are unchanged, and the datasets of a fast and a slow component are
deliberately *not* on one time axis: they are different sampling rates of one
run, and `xr.merge` of the two is an outer join by design.

The conventions, which are JCM's:

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
  it. Each `to_xarray` hands those values, plus `TimeAxis.attrs`, straight to
  xarray. The dates are proleptic Gregorian whatever the model calendar is;
  the calendar governs the seasonal cycle and forcing selection, not the
  labels.
- **Variable names**: state and derived quantities keep their plain names, and
  every variable that came from a component's *forcing* is written with a
  `forcing_` prefix — `jem.base.component.FORCING_VARIABLE_PREFIX`, applied by
  `forcing_variable(name)`, which is what a slab model's
  `_create_xarray_data_vars` and `VerosComponent.to_xarray` call. (It lives
  with the contract rather than in the slab package, and is re-exported from
  `jem.components.slab.base`, so that there is one definition of the prefix.
  It is a convention of the packaged output, not a protocol requirement: the
  coupler never inspects a dataset, a wrapper around an external model may
  keep that model's own names, and the helper leaves a name that already
  carries the prefix unchanged.) Two
  components legitimately hold the same physical field — one produced it, the
  other received it — and without the prefix the merge collides on the shared
  name. So the slab atmosphere and the slab land model write
  `forcing_total_heat_flux` while the ocean writes its own derived
  `total_heat_flux`; the sea ice writes `forcing_ice_frazil_melt_energy` for
  the field the ocean published as `ice_frazil_melt_energy`; and Veros writes
  `forcing_heat_flux`, `forcing_freshwater_flux`, `forcing_surface_taux`,
  `forcing_surface_tauy` and `forcing_surface_air_temperature` for the five
  fields an exchanger hands it, keeping plain names for the `temp`, `salt`,
  `u`, `v` and sea-surface fields it computes.
- **A variable's role is metadata, not a name to parse.** Every packaged
  component tags each output variable with `jem_role`
  (`jem.base.component.role_attrs`), whose value is the section of the carry
  the variable came from: `state` (what the component integrates), `derived`
  (what it diagnosed for others to read) or `forcing` (what it was given). So
  `ds.filter_by_attrs(jem_role="forcing")` is the whole query. This does not
  replace the `forcing_` prefix and is not redundant with it: the prefix exists
  to stop an `xr.merge` collision between a field one component computed and
  the lagged copy another received — they are genuinely different variables —
  while matching a prefix cannot tell a *received* `forcing_q_flux` from a
  model whose own field happens to start with the same word, and says nothing
  at all about the variables that are not forcing. A variable that is none of
  the three — a grid mask, a layer thickness, anything time-invariant that came
  from the component's configuration rather than its carry — is left untagged,
  which is a meaningful answer rather than an omission. The JCM wrapper returns
  jcm's own dataset and tags only the surface boundary conditions an exchanger
  writes into it; the rest of those names are jcm's, and their roles are not
  JEM's to assert.
- **A configuration-dependent variable is decided by the run, not by the
  component object.** `SlabOceanModel` writes `forcing_q_flux` only when the
  trajectory actually applied a Q-flux, and `step` follows the
  `forcing_method` in `carry["params"]` — which `initialize(params)` may set
  to something other than the method the model was constructed with. So the
  step publishes the Q-flux snapshot it applied as a key of its own
  diagnostics, and `_create_xarray_data_vars` writes the variable when that
  key is there. Keying it off `self.params` instead would drop an applied
  Q-flux from the output, or publish a constant zero as though a Q-flux were
  active. This is safe under `lax.scan` precisely because `forcing_method` is
  static (`pytree_node=False`): it cannot change during a run, so the
  diagnostics structure is constant even though it varies between runs.

Together these are what make `xr.merge([datasets["atm"], datasets["ocn"]])` an
N-long join rather than a 2N-long outer union.

## The declarative exchange

Almost every exchange in a coupled Earth-system model is the same shape: field
X of component A becomes field Y of component B, optionally regridded on the
way. `jem.exchangers` writes that as a table instead of a function.

```python
from jem import Coupler, default_exchangers

components = {"atm": atm, "ocn": ocn, "seaice": seaice}
coupler = Coupler(components, default_exchangers(components),
                  coupling_timestep=jdt.to_timedelta(1, "day"),
                  start_date=start_date)
```

`ExchangeSpec(src, dst, regrid=None)` is one row, addressing a field as
`"component.section.field"` with `section` one of `state`, `derived`, `forcing`
— the carry layout every packaged component shares. `Exchange(specs,
regridders)` executes a list of rows and *is* an ordinary `Exchanger`, so
nothing in the coupler knows the difference. A component whose carry is shaped
differently can still be coupled, with a hand-written exchanger.

`default_exchanges(components)` is the standard wiring, in one place:

| source | destination |
| --- | --- |
| `atm.derived.total_heat_flux` | `ocn.forcing.total_heat_flux` |
| `atm.derived.total_heat_flux` | `lnd.forcing.total_heat_flux` |
| `ocn.derived.ice_frazil_melt_energy` | `seaice.forcing.ice_frazil_melt_energy` |
| `ocn.state.sea_surface_temperature` | `atm.forcing.sea_surface_temperature` |
| `seaice.derived.ice_fraction` | `atm.forcing.sice_am` |
| `lnd.state.land_surface_temperature` | `atm.forcing.stl_am` |
| `lnd.state.snowc` | `atm.forcing.snowc_am` |
| `lnd.state.soilw` | `atm.forcing.soilw_am` |

A row survives only if **both** its components are present, so an aquaplanet
with no land model gets the four rows that do not mention `lnd`, and an
atmosphere/ocean pair gets two. The wiring is by *name* —
`("atm", "ocn", "lnd", "seaice")` — and a component registered under a name
that just misses one of those (`ice`, `ocean`, `land`) is left unconnected with
a warning, because the failure it would otherwise cause is silent: the sea ice
simply never receives anything.

Three properties are worth stating, because a hand-written exchanger has them
only by accident:

- **An exchange is simultaneous, not sequential.** Every source is read from
  the mapping as it arrives, before any destination is written, so reordering
  the table cannot change a run. Two rows writing the same destination is a
  `ValueError` at construction for the same reason.
- **It is checkable before the run.** `Exchange.validate(carries)` — which
  `jem.runners` calls with `coupler.initialize().components` — turns a mistyped
  component, section, field or regridder into an error naming the spec, before
  a model is integrated. The same lookups fail the same way at trace time for a
  caller that skips it, which is still far earlier than a wrong number.
- **It never mutates.** New section structs with `.replace(...)`, new carries
  with `dict(carry, ...)`, and a new mapping — the rule the exchanger contract
  states above, enforced here once for every table.

**Regridding.** A row that crosses the atmosphere/ocean grid boundary may name
a regridder, and `default_exchanges` names one from a mapping keyed by
*direction and kind*: `a2o`/`o2a` for the direction, `flux`/`state` for the
kind, the kind following the source section. That split is the one the
mixed-grid example makes by hand — extensive quantities (heat fluxes, the
freeze/melt energy, an areal ice fraction) are mapped conservatively so their
budgets survive the interface, while an intensive state variable such as SST is
interpolated bilinearly, which does not leave a conservative map's staircase in
a smooth field. Rows that stay on one grid never get a regridder.

```python
default_exchangers(components, regrid={
    "a2o_flux":  ESMFRegridder(a2o_conservative_weights),
    "o2a_flux":  ESMFRegridder(o2a_conservative_weights),
    "o2a_state": ESMFRegridder(o2a_bilinear_weights),
})
```

The maps themselves are `jem.regrid.ESMFRegridders`, an immutable named
collection built from ESMF weight files generated offline by
`ESMF_RegridWeightGen` — JEM applies weights, it does not compute them, because
they depend only on the two grids and never on the run.

### Coupling is lagged

None of this changes *when* fields move, and with the default workflow the
exchange is lagged by one coupling step. With `["exchange", "atm", "ocn"]`:

- `exchange` runs **first**, on the carries as they were left at the end of
  step *n−1*. So at step *n* the ocean is driven by the atmosphere's fluxes
  from step *n−1*, and the atmosphere sees the SST the ocean reached at the end
  of step *n−1*.
- On the **first** step there is no previous step, so each component receives
  whatever its `initialize()` put in its forcing section — zeros, for every
  packaged component. A run therefore begins with one step of uncoupled
  spin-up: the ocean's first step sees no heat flux at all.
- The lag is a property of the *workflow*, not of the exchanger. An
  `["atm", "exchange", "ocn"]` workflow hands the ocean the atmosphere's fluxes
  from the same step, at the cost of giving the atmosphere a two-step-old SST.
  Neither order gives every component same-step information; that needs an
  iterated (implicit) exchange or a partitioned workflow, which is a follow-up.

Writing the default coupling down in one place is what makes the lag reviewable
at all: before, every example spelled the same exchange out by hand and none of
them said which step the fields came from.

## Running a model

A `Coupler` produces functions, not runs. `jem.driver.run_chunked` is the one
loop that turns one into a run, and **every run default lives on its
signature** — the config group `coupled_run` names the same keys and repeats
none of the values.

```python
from jem import run_chunked

result = run_chunked(
    coupler,
    total_time="10 years",
    chunk="30 days",
    output_dir="output",
    output_averages=True,
    checkpoint_path="checkpoint",
)
```

Per chunk it integrates, labels and writes the output, checkpoints, and checks
the state is still healthy:

```
carry = initial_carry or coupler.initialize()          # or the checkpoint's
trajectory = coupler.generate_trajectory_function(steps_per_chunk)   # compiled ONCE
for each chunk:
    first_step = int(carry.step)
    carry, diagnostics = trajectory(carry)
    datasets = datasets_for_chunk(coupler, diagnostics, first_step=first_step, …)
    paths += write_chunk(datasets, output_dir, first_step // steps_per_chunk)
    coupler.save_state(carry, checkpoint_path)
    ok, report = health_check(datasets, chunk_index, elapsed_days)
```

It returns a `RunResult`: the `final_carry`, `steps_completed`
(`int(final_carry.step)` — the run's position on the clock, including whatever
a checkpoint restored, not the number of steps this call integrated),
`completed`, one `report` per chunk and every `path` written.

**Chunking rules.** `total_time` and `chunk` are `jcm.date.parse_duration_days`
strings or numbers of days, parsed on the *coupler's* calendar, so `"1 year"` is
as long as the atmosphere's year. Both must be whole multiples of the coupling
timestep — a coupled step is the smallest thing the loop can integrate — and
`total_time` must be a whole multiple of `chunk`. All three are checked before
anything is built or compiled, and each message names both quantities. A final
partial chunk is refused rather than accommodated: it would need a second
compiled trajectory for one call, and a run length that does not divide into
chunks is far more often a mistake in the configuration than a request.

**The health gate.** `default_health_check` runs `jcm.diagnostics.check_health`
on `datasets["atm"]`, so a coupled run stops on the same evidence an uncoupled
atmosphere does, and `bail_on_unhealthy` (the default) stops at the first chunk
it rejects rather than spending a queue slot integrating a broken state. The
output written so far is kept and `RunResult.completed` is False. A coupled
model with **no** atmosphere gets `{"skipped": "no atmosphere"}` — an
abstention, not a pass: a gate for the surface components would have to know
each one's physical ranges, which is the components' business.
`health_check=None` removes the gate entirely, and `bail_on_unhealthy=False`
logs and carries on, which is what a run studying the instability itself wants.

**Checkpoints and resume.** `checkpoint_path` is a **single directory**,
rewritten after every chunk, not a directory of dated restart points. That is
what makes resuming a run the same command as starting it: point at the path,
and the run either starts from scratch or continues from where it stopped. The
cost is that only the newest state survives; a run that wants a history of
restart points keeps its own directory of them and hands each one in as
`initial_carry`. Overwriting in place is safe because the carry file is written
last and removed first (see *Carry*), so an interrupted save leaves a directory
the loop refuses to resume from — it logs and starts from the initial carry
instead — rather than a mixture of two steps.

`CoupledCarry.step`, restored from the checkpoint, is the only source of truth
for how far the run has got; nothing is derived from a chunk index or a file
name. What is left is `remaining_batches(int(carry.step), total_steps,
steps_per_chunk)`, so a run resumed with a *different* chunk length — a
perfectly legitimate choice, since the chunk is a property of the run and not of
the checkpoint — finishes the part-chunk first in one short batch (one extra
compile, on that batch only) and still stops exactly at `total_time`.

**Output files.** One file per component per chunk,
`<output_dir>/<component>-<chunk index>.nc`, with the chunk index derived from
the coupled step counter — so a resumed run continues the numbering instead of
overwriting what it already wrote. Each chunk is labelled with its own dates,
because `first_step` is passed through to `Coupler.to_xarray`.

`subsample=n` keeps every *n*-th coupling step. `output_averages=True` is
defined against jcm's meaning of the same word rather than beside it: jcm
replaces each saved record with the mean over its save interval, labelled at the
interval's end, and the coupler's records are already one per coupling step — so
the coupler's output interval is the **chunk**, and the flag replaces a chunk's
records with their time mean, labelled with the chunk's last time and carrying
the CF `cell_methods = "time: mean"` that says so. Monthly-mean output is then a
30-day chunk. In a coupled run the atmosphere's per-step records are *already*
step means (the JCM wrapper integrates each coupling step with
`output_averages=True`), so averaging a chunk of them is the chunk mean exactly,
with no double counting. A variable with no time axis — a grid mask, a layer
thickness — is passed through by both reductions rather than averaged into a
one-record time series.

**Reductions that must not cost memory.** Writing every step out and reducing on
the host means holding a chunk's diagnostics — for an atmosphere, the largest
array in the run — until the chunk ends; and chunking *by calendar month* means
compiling a 28-, a 30- and a 31-day trajectory, because `iterations` is static.
`generate_trajectory_function(iterations, accumulate=(init, update))` does
neither: `update(accumulator, diagnostics, time)` runs inside the `lax.scan`
body, with the same `CouplingTime` that step's components were handed, and the
scan returns nothing per step. The call becomes `(carry, accumulator=None) ->
(carry, accumulator)`, so a chunked run threads the accumulator from one call to
the next and the chunk boundaries need not line up with anything.

```python
from jem.accumulate import monthly_mean

monthly = monthly_mean(coupler)
trajectory = coupler.generate_trajectory_function(365, accumulate=monthly)
carry, accumulator = trajectory(coupler.initialize())
means = monthly.finalize(accumulator)        # (12, …) per variable
```

`monthly_mean` takes only the coupler: the accumulator's shapes come from
`jax.eval_shape` of one coupled step, and a step's month is a lookup in a static
day-of-year table reached from the step counter reduced modulo the steps in a
year — exact integer arithmetic, and one compiled trajectory for a whole year.
Which month a step counts in follows the label JEM writes on its output record
(the *end* of the coupling interval), so `monthly.finalize(...)` and
`to_xarray(...).groupby("time.month").mean()` of the same run are the same
numbers. A calendar with no fixed day-of-year to month table (gregorian, with
its leap years) and a coupling step that does not divide the year are refused
with a message saying why, rather than binned approximately. The accumulator is
an ordinary pytree in the scan carry, so `jax.grad` of a monthly mean flows
through the reduction exactly as it flows through the trajectory. Without
`accumulate`, the generated function is what it always was.

## Configuration

Python is the primary interface. The configuration layer is a thin wiring layer
over it, and `python -m jem.main` (or the `jem` console script) is one command
for a coupled run:

```bash
python -m jem.main +configuration=aquaplanet-slab coupled_run=smoke
```

**jax-gcm's own groups, re-rooted under `atmosphere`.** `jem/config/config.yaml`
puts `pkg://jcm.config` on Hydra's search path and composes jcm's groups at
`atmosphere.*`, so `cfg.atmosphere` is exactly the config
`jcm.runners.build_model` expects and jcm's group and option names are
unchanged. The price is that the group's package has to be spelled out in an
override — `physics@atmosphere.physics=echam`,
`+configuration@atmosphere=speedy-t31` — and that spelling it wrong is quiet:
`+configuration=speedy-t31` composes that bundle at the *root*, where its
`physics`, `terrain` and `run` keys are nobody's and nothing reads them.
JAX-ESM's own groups (`ocean`, `land`, `seaice`, `coupling`, `regrid`,
`coupled_run`) sit at the top level, and `configuration` composes a named
coupled model out of all of them.

**The group-name collision, and why the run group is `coupled_run`.** Hydra
resolves a group option from the first search-path entry that has it, and the
primary config package precedes `pkg://jcm.config`. A group called `run` here
would therefore shadow jcm's own `run/default.yaml` and `run/longrun.yaml`: the
atmosphere would silently be handed the coupler's run keys, and every jax-gcm
configuration bundle that says `override /run: longrun` (16 of the 19 shipped)
would compose the wrong file. `atmosphere.run` also already exists and means
something else. So the coupled run keeps its own name at both ends —
`coupled_run=smoke` selects an option, `coupled_run.total_time="90 days"` sets
one key — and `test_jcm_run_group_is_not_shadowed` pins it down.

**YAML is wiring, and a test enforces it.** A key earns its place in a group or
configuration file only by being (a) `_target_`, (b) a required input marked
`???`, or (c) a value that differs from the Python default *and* is what the
named configuration is about. `test_config_has_no_python_defaults` instantiates
every group option and every configuration and fails if a supplied value equals
the target's own default, so a physics default cannot acquire a second home in
the configuration and drift from the class that owns it. `jem.runners` never
reads a physics parameter either: its one table is `GROUP_TO_NAME`, and
`test_runners_has_no_component_kwargs` fails if any component parameter's name
appears in its source at all. What the runner *does* supply is what a config
file cannot name — a surface component's `SlabGrid` (from the built
atmosphere's `coords.horizontal` and `terrain.fmask`, or from a SCRIP
`grid_file`), the regridders, and the coupling timestep — injected into
`hydra.utils.instantiate` after the keys that describe them are removed.

**Packaged data resolvers.** `${jcm_data:bc/t30/clim/forcing.nc}` and
`${jem_data:DisplacedPoleGrid.SCRIP.nc}` resolve to files inside the installed
`jcm.data` and `jem.data` packages; importing `jem.config` registers them. They
exist so the shipped configurations run **offline**: jax-gcm's own
configurations fetch boundary data from an `hf://` mirror, which needs the
network and a warm cache, while everything named through these is already on
disk beside the code. A path that does not exist is reported while composing,
naming the key, rather than much later as a netCDF open error.

`+atmosphere.constants.grav=9.7` reaches `jcm.runners.apply_constants_overrides`
before the model is built, because the dynamical core reads the live
`jcm.constants` singleton at construction. The override is process-global and
the surface components read the same singleton, so one such setting moves the
whole Earth system, not only the atmosphere.

## The jax-gcm dependency contract

JAX-ESM is built on jax-gcm but lives in its own repository and is installed
against a source *checkout* of it, not a PyPI release. Without a recorded pin,
"which jax-gcm does this work with?" has no answer, and a rename on the jax-gcm
side surfaces as an `AttributeError` or a `KeyError` deep inside somebody's
coupled run.

`jem/components/jcm/contract.py` records both halves of the contract:
`JCM_SUPPORTED_REV`, the revision every gate runs against, and
`JCM_INTEGRATION_POINTS`, every jax-gcm name JAX-ESM reaches for — the functions
it calls, the private attributes it still reads (each tagged with the jax-gcm
issue that will remove the need), the physics diagnostics fields the surface
exchange is read out of, the package data it resolves, and the constructors its
documented workflow asks a user to call. Each entry says what it is used for,
which is what makes it possible to decide whether an entry may be deleted.

`tests/unit/test_jcm_contract.py` walks that list against the installed `jcm`,
so a jax-gcm rename fails as "jax-gcm renamed or removed X, which JAX-ESM used
for Y, at revision Z" — at the cheapest possible moment, rather than mid-run.
The pin is a `dev` sha because no tagged jax-gcm release carries the two changes
Phase 2 is written against (#750's one run schema and `configuration` group,
#763's input-resolution engine); `pyproject.toml`'s `jcm>=2.1.0b0` is the
loosest true statement of the same thing, since jax-gcm bumps its version only
at release. Every required CI job checks that revision out through a
workflow-level `JCM_REV`, which the test asserts equals `JCM_SUPPORTED_REV`, and
a non-blocking `canary-jcm-dev` job keeps tracking `dev` so drift stays visible
without blocking a pull request. `contract.py`'s docstring is the procedure for
bumping the pin.

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

`VerosComponent.step` makes the same comparison against Veros'
`variables.time`, from the same tolerance
(`jem.components.clock.clock_tolerance_seconds`, which is where it lives so the
two wrappers cannot answer the question differently). Veros has no calendar, so
its counter is not seconds since the coupler's `start_date` but seconds since
its setup's own start: `bind` records the reading the setup holds when the
coupler adopts it, and the check compares `variables.time` minus that zero
point. A setup that was already integrated before it was wrapped therefore
starts the coupled run where it stands — JEM cannot know which absolute date
that state belongs to — while a *later* disagreement, such as a Veros restart
paired with a `CoupledCarry.step` from elsewhere in the run, is caught. Both
checks report through `jax.debug.callback` rather than raising: they run inside
the coupled `lax.scan`, where a Python exception cannot fire on a traced value.

## Adding a new component

1. Write the class (or a wrapper class for an external model) under
   `jem/components/`. Give it a `name`, an `initialize()` and a
   `step(carry, time)`; follow the `state`/`forcing`/`derived` carry convention
   so exchangers stay readable, and put tunables in a `flax.struct` parameters
   dataclass carried as `carry["params"]` so they stay differentiable.
2. Keep `initialize()` pure with respect to `self`: load boundary data in
   `__init__` (it is configuration, not state), so calling `initialize()` twice
   gives the same answer. If any parameter is an *initial condition* — read by
   `initialize` and never by `step` — give it the `initialize(params=None)`
   signature the slab models have, so that parameter can be varied and
   differentiated (see *Parameters*); a parameter that can only be set at
   construction is a dead leaf in the carry.
3. Add `bind(...)` if the model has an internal timestep, and raise `ValueError`
   when the coupling timestep does not divide it. Add `to_xarray(diagnostics,
   time)` if it produces output, and `save_state`/`load_state` if its carry
   cannot be checkpointed as a plain pytree.
4. Export it from `jem/components/__init__.py` (lazily, via the module's
   `__getattr__`, if it pulls in an optional dependency — as Veros does).
5. Register it: `Coupler({"mycomp": MyComponent(...)}, ...)`. If it is one of
   the standard surface components, register it under the name
   `default_exchanges` wires (`ocn`, `lnd`, `seaice`) and the standard coupling
   applies with no table of your own; otherwise give `Exchange` the rows it
   needs, or write an exchanger.
6. To make it configurable, add a `jem/config/<group>/<option>.yaml` naming it
   as `_target_` — wiring only, no parameter defaults — and, if it is a new
   *kind* of component, one line in `jem.runners.GROUP_TO_NAME`. Nothing else
   in the runner changes: a component is configured by a group file, never by a
   branch there.
7. Add tests under `tests/unit/`, including a two-step run through
   `Coupler.generate_trajectory_function(2)` — a component-only test cannot
   catch a carry-structure mismatch, which only `lax.scan` sees.
