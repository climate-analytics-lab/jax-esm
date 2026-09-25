# Architecture

How JEM couples black-box components. This is the reference for developers
adding a component or debugging an exchange; the user-facing walkthrough is
{doc}`../adding_a_component`. {doc}`../python_api` shows the same objects --
`Coupler`, the exchangers, `run_chunked` -- built directly, for a reader who
wants the construction rather than the design rationale.

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
- `JCMComponent`'s carry has three keys beyond the three every packaged
  component shares (`"state"`, `"derived"`, `"forcing"`): `"physics"`, JCM's
  cross-step physics carry (sub-cycled radiation, prior-step TKE, the
  tendencies one term hands to the next), and `"time"` / `"step"`, jax-gcm's
  own exact `RunState` clock (jax-gcm PR 878 — an absolute
  `jax_datetime.Datetime` and an integer JCM-timestep count). All three are
  threaded straight back into `Model.run_from_state_with_carry`, which now
  *requires* `initial_time` / `initial_step` explicitly rather than inferring
  them from the incoming dycore state: dropping any of the three between
  coupling steps would reset physics memory or lose the exact clock, both
  silent and systematic. `"physics"` contains integer and boolean leaves, so
  it must never be cast wholesale to a float dtype; `"time"` / `"step"` are
  threaded rather than recomputed from the coupler's own step counter each
  call because JCM's `RunState` is now the **authoritative** clock (jax-gcm
  v3, PR 878) and jax-gcm's own migration guide says to keep threading all of
  it — not because recomputing it would overflow (an int32-safe
  reduce-before-multiply decomposition, `jem.base.calendar.gregorian_instant`,
  computes exactly this instant from the coupler's own step count for
  `JCMComponent._report_authoritative_clock_drift`'s own drift check, so
  recomputing was never the problem) — but because threading is what
  guarantees JCM's clock and the coupler's can never disagree, which
  recomputing one from scratch cannot once a checkpoint or a
  differently-configured coupler is involved (see
  `jem.components.jcm.component`'s module docstring).
- `JCMComponent`'s `carry["derived"]` is a `JCMDerived` struct holding the
  surface exchange (`total_heat_flux`, `total_freshwater_flux`, `evaporation`,
  `precipitation`, `u0`, `v0`) plus `physics`, JCM's own per-step diagnostics
  dict, carried opaquely so an exchanger can reach any field JCM computes.
- `JCMComponent`'s `carry["forcing"]` is a whole `jcm.forcing.ForcingData`, so
  an exchanger addresses a boundary condition under JCM's own field name
  (`atm.forcing.sea_surface_temperature`). It is the one carry section that is
  both **read from a file and overwritten every step**, and the two want
  different shapes. With `forcing@atmosphere.forcing=from_file` JCM builds each
  time-varying boundary condition as a `jcm.forcing.TimeSeries` — values, time
  axis and alignment mode, three pytree leaves — and slices it by date on every
  internal timestep; an exchanger writes one `(ix, il)` array. A field that was
  a time series before the exchange and an array after it changes the carry's
  pytree structure, which `lax.scan` cannot carry (and which the coupler
  refuses by name — rule 2 of [Exchangers](#exchangers)).

  So the atmosphere is *told* which fields the coupling supplies —
  `JCMComponent.set_exchanged_forcing(names)` — and `initialize()` collapses
  exactly those to the climatology at the run's start date. From `initialize()`
  onward the section has the structure an exchange preserves. Every field no
  component supplies keeps its time series and goes on being sliced by JCM, so
  a land surface in a run built with `land=none` still follows the seasonal
  cycle.

  The names are a property of the *coupled model*, not of the atmosphere, which
  is why nothing assumes them: `jem.runners.build_coupler` reads them off the
  built coupling table with `jem.exchangers.exchanged_fields(exchangers,
  "atm")`, and a configuration coupled by a hand-written `coupling.exchanger` —
  a function, with nothing to read — lists them in
  `coupling.exchanged_forcing`. Assuming a fixed set instead would freeze an
  unexchanged climatology at its start-date value without saying so; a
  structure error that names the element responsible is much the better
  failure.

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
coupled step counter, plus the *name* of every component that does — beside one
subdirectory per component that writes itself (`VerosComponent`, a nested
`Coupler`). A directory with no `carry.msgpack` is
refused with a `ValueError`: its position in the seasonal cycle is not
recoverable, and resuming at step 0 (or at a step reconstructed from a batch
index) would silently move the run's calendar.

The format is jax-gcm's: `jem.checkpoint.save(carry, path)` flattens any pytree
to a list of typed arrays serialised with flax's **MessagePack** (`msgpack`)
codec — a compact binary serialization format, a binary cousin of JSON, reached
through `flax.serialization.msgpack_serialize`, which is where the
`carry.msgpack` name comes from — and
`load(template, path)` rebuilds the tree from a *template*'s treedef. The tree
itself is never stored — rebuilding it from the template is what keeps the
format small — but a manifest of it is, and that is what makes the format
self-checking. Every leaf is compared with the template's path, shape and
dtype, so a checkpoint written by another grid or another component composition
fails naming the leaf instead of deserialising into something that only explodes
later inside a `lax.scan`. The leaf paths are stored because leaf count, shape
and dtype together cannot tell two same-shaped carries apart, and silently
swapping two components' carries on resume is the failure that would follow.

The manifest also holds the repr of the whole `PyTreeDef`, compared after the
leaves — leaf-level checks name the offending leaf, which is more use than two
tree reprs, so the structure comparison is left as the catch-all for the
differences no leaf can show. There are three. A component whose carry holds no
arrays (`{}` or `None`) contributes no leaf at all, so renaming one would
otherwise load cleanly and resume a *different* composition at the saved step;
so does a component that checkpoints itself, whose carry never reaches the
shared file, which is why `save_coupled_carry` stores its name there as an empty
marker entry. A container that changed type without its contents moving (a list
for a tuple) is the second. The third is a **static** (`pytree_node=False`)
parameter that changed value: JAX keeps those inside the `PyTreeDef`, so
resuming with `forcing_method` edited from `"none"` to `"qflux"` is refused
rather than continued. That is deliberate — a static parameter selects a code
path at trace time, so the resumed run would be a different model — and the
mismatch message names it as one of the causes and shows the difference. A
*differentiable* parameter is a leaf, so it is restored from the checkpoint
instead: editing one between runs is overridden, not refused.
Both the stored leaf and the template go through `jnp.asarray` first: a
component's parameter default is a Python float in `initialize()` and a float32
array in the carry `lax.scan` returns, and the two have to compare as one leaf.
It is also why `load_carry` needs a template at all, and takes it from each
component's own `initialize()`.

**Loading is all-or-nothing.** The template built from `initialize()` supplies
the pytree *structure* and nothing else: every leaf of the restored carry comes
from the checkpoint, and the freshly-initialized values are discarded. A
component the checkpoint does not hold is a `ValueError`, never a component
quietly left at its initial state — that would continue one part of the model
from the saved step while another restarted at the start date, a run that is
neither a resume nor a cold start and that nothing downstream could detect.
`load_coupled_carry` logs at INFO which components it read from the shared
carry file, which read themselves back through their own `load_carry`, and the
step it restored, so every component's source is named.

**The coupler is what a driver checkpoints through**, in one call each way:

```python
model.save_carry(final_carry, checkpoint_dir / f"step_{int(final_carry.step):08d}")
carry = model.load_carry(saved)
```

Which components need writing by hand rather than pickling is a property of the
*components* — `VerosComponent` has to go through Veros' HDF5 restart writer
because a `VerosState` is not a pytree — and the coupler is the one object that
knows them all. `Coupler.save_carry` therefore derives the savers itself, as
`{name: component.save_carry for … if isinstance(component, SupportsCheckpoint)}`,
and `load_carry` derives the loaders from the same capability. A driver never
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

`jem.checkpoint.save_coupled_carry(coupled_carry, directory, component_savers=…)` and
`load_coupled_carry(directory, component_templates, component_loaders=…)` are the
underlying functions and still take the mappings explicitly, for a caller that
wants to override a saver or supply one for something that is not a component
capability. `Coupler.save_carry` / `load_carry` are the answer for every
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
| `SupportsCheckpoint` | `save_carry(carry, directory)` / `load_carry(directory)` | `VerosComponent`, `Coupler` |

`bind` is called by the coupler once per component, from `add_component` (hence
from the constructor for everything passed to it), and it is the only way a
component learns anything about the clock outside a step. A component with an
internal timestep uses it for the coupling interval — `JCMComponent` converts it
to the number of days it passes to JCM as `save_interval`/`total_time`,
`VerosComponent` to a count of tracer timesteps — and it is where a disagreement
about the clock is refused: both wrappers raise `ValueError` if the coupling
timestep is not a whole multiple of the model's own, and `JCMComponent`
additionally refuses a `start_date` that differs from `model.start_time`. jax-gcm
v3 (PR 878) removed `Model.calendar` — the atmosphere's clock is unconditionally
proleptic Gregorian now, with no calendar of its own to check against — so
`JCMComponent.bind` instead refuses any coupler calendar but `"gregorian"`
(`jem.runners.ATMOSPHERE_CALENDAR`): any other choice would silently run the
atmosphere's seasonal cycle out of phase with every other component's, which
reads its own calendar from the coupler.
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
    days_per_year: float            # static: jem.base.component.days_per_year(calendar)
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
  `days_per_year`, `year_offset_seconds`, and, since the 2026-09 migration
  review's exact Gregorian `year_fraction`, `start_day`/`start_second`) are
  comparable at trace time, the step counter is a traced array. Calling `step`
  before `bind` is a
  `RuntimeError`. For `r == 1` the inner step is run directly and the
  diagnostics gain no extra axis, mirroring multiplicity 1.
- **The carry** of the inner coupler is a `CoupledCarry` living inside the
  outer one's `components`, so there are two step counters: the outer counts
  outer steps, the inner counts its own. `outer.save_carry(carry, directory)`
  writes the inner model into `directory / <its registered name>` through the
  inner coupler's own `save_carry`, and `outer.load_carry(directory)` reads it
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
records per coupled step stamped at the midpoint of each hour. Components with
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
- **The time label is the MIDPOINT of the interval a record covers**, as an
  exact `datetime64[ms]`: record *k* holds the average over
  `[start_date + k dt, start_date + (k+1) dt)` and is stamped
  `start_date + (k + 1/2) dt`. This is jax-gcm v3's convention (PR 878,
  `docs/source/v2_to_v3.rst`, "One real datetime clock" — pre-878 it was the
  interval's *end*, as a float64 days-since-epoch product that was inexact
  past a 128 ns ulp), and `TimeAxis.datetimes()` is the one place it is
  written down: it computes each record's exact interval bounds with
  `jax_datetime` (whole-second arithmetic, so the bounds themselves are always
  exact), converts them to `datetime64[ms]` with jax-gcm's own
  `jcm.predictions.output_time_labels` (the published conversion that closed
  jax-gcm#862, so this calls it rather than reimplementing it), and takes the
  midpoint by plain NumPy arithmetic on the millisecond values — the same
  two-step recipe `ModelPredictions.to_xarray` uses for its own averaged
  output. This is exact for *any* coupling step, not merely one whose step is
  a power-of-two fraction of a day, and lets a midpoint fall on a half second
  for an odd-length interval without `jax_datetime.Timedelta`
  (whole-seconds-only) ever having to represent one. Each `to_xarray` hands
  those values, plus `TimeAxis.attrs`, straight to xarray. The dates are
  proleptic Gregorian whatever the coupler's own calendar is; that calendar
  governs the seasonal cycle and forcing selection, not the labels — and a
  `Coupler` bound to a real `jcm.model.Model` must use `"gregorian"` for it
  regardless, since jax-gcm's own clock is unconditionally Gregorian now (see
  *The clock*, and `JCMComponent.bind`). The leap-day consequence the midpoint
  change did **not** remove: a `365_day` coupler's year is a day shorter than
  a Gregorian leap year, so its labels fall a day further behind the model
  calendar at every Gregorian 29 February — a run started on 1 January 2000
  labels the instant the model calls 1 March 00:00 as `2000-02-29T12:00`, and
  everything downstream that bins by the *label* parts company there from
  everything that bins by the *model calendar* (see the `accumulate` section
  below). The inconsistency is JCM's, recorded upstream as jax-gcm#449; JEM
  mirrors it rather than emitting labels of its own, which would no longer
  merge with the atmosphere's on one time axis. Calendar-consistent labels for
  every component, the atmosphere's included, are tracked as #118.
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
  `u`, `v`, `psi` and sea-surface fields it computes. The ocean's `psi` is
  the barotropic streamfunction, and which of two things it is depends on
  how the setup solves the external mode. With
  `settings.enable_streamfunction` (Veros' own default) it is Veros'
  prognostic `variables.psi` at the current time level. With the linear
  free surface — what every Veros setup shipped with JEM selects — Veros
  reuses that same array for the surface pressure, a different quantity in
  different units on a different grid, so the wrapper diagnoses the
  streamfunction instead: it integrates the depth-integrated zonal
  transport northwards from a southern boundary where the streamfunction
  vanishes, using the discrete relation Veros inverts *when it does solve
  for a streamfunction* — `sum_k u dzt maskU = -(psi[i,j] - psi[i,j-1]) /
  dyt[j]`, the relation behind its barotropic-mode update. A free-surface
  run never reaches that code (it solves for a surface pressure, and the
  barotropic mode enters the momentum equation as a pressure gradient), so
  what carries over is the definition rather than that run's own
  arithmetic, applied to the transports it did produce. That the
  definition is the right one is pinned by a test that makes the diagnosis
  reproduce Veros' own `psi` where Veros has one: there the two agree to
  machine precision, up to the additive constant a streamfunction is
  defined up to (Veros fixes it by holding its first island at zero, the
  diagnosis by the southern boundary). The variable's `comment` attribute
  says which of the two the file holds, and the masks a reader needs are
  published beside it — `mask_surface_Z` for the zeta points `psi` sits on,
  whose land values the integration carries through rather than computes,
  and `mask_U` for the depth integral behind it. A free-surface run also
  publishes the sea surface height of its surface-pressure solve as `ssh`
  (m, on the T grid) — `ssh = psi / grav`, the relation Veros itself
  applies, evaluated on the `psi` of the record's own time level because
  Veros' `variables.ssh` is written before it permutes its time indices and
  so lags the rest of the record by one Veros timestep; a streamfunction run
  has no sea surface height and publishes none.
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

**One table per carry layout.** Component wrappers do not all name the same
physical field the same way, nor keep it in the same section: the Veros ocean
takes its surface heat flux as `forcing.heat_flux` and publishes its sea
surface temperature from `derived` (its `state` is Veros' own `VerosState`
object, not a struct of exchangeable fields), where a slab has
`forcing.total_heat_flux` and `state.sea_surface_temperature`. So the table
above is `STANDARD_EXCHANGES`, and a `VerosComponent` registered as `"ocn"`
selects `VEROS_OCEAN_EXCHANGES` instead — the same wiring in Veros' names, plus
the freshwater flux Veros also takes, and with no `ocn` → `seaice` row because
Veros publishes no freeze/melt potential (that combination is warned about).
The choice is made by *type*, which needs real components: called with a list
of names, `default_exchanges` cannot tell one ocean from another and gives the
slab table. The check looks the wrapper's module up in `sys.modules` rather
than importing it, so a JAX-ESM without the optional Veros dependency never
imports Veros to find out that it has no Veros ocean.

What the Veros table deliberately does **not** carry is the **wind stress**:
Veros integrates `forcing.surface_taux`/`tauy` and the atmosphere publishes a
near-surface *wind*, so getting from one to the other is a bulk drag law (and,
on a rotated grid, a rotation into its local frame) — a computation, not a
copy, and therefore a hand-written exchanger. The shipped `veros-*`
configurations give one:
`coupling.exchanger: jem.fluxes.VerosExchange` — a bulk drag law on the
atmosphere's near-surface wind, regridded then rotated into the ocean grid's
frame, plus a freezing-point mask on the heat and freshwater fluxes. See
`jem.fluxes` for the exchanger itself; the declarative table above carries
the rest of the coupling.

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
kind. The kind is written on each row of the table rather than inferred from
the carry section the field is read from, because the two do not agree — the
same intensive sea surface temperature comes from `state` on a slab and from
`derived` on a Veros ocean. That split is the one the
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
  packaged surface component, and for `JCMComponent` the boundary conditions it
  was built with, taken at the start date. A run therefore begins with one step
  of uncoupled spin-up: the ocean's first step sees no heat flux at all. (Under
  this workflow the atmosphere's own initial forcing is overwritten before it
  ever steps, since `exchange` runs first; under `["atm", "exchange", ...]` it
  is what the atmosphere integrates its first step on.)
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
    total_time="2190 days",      # a whole number of chunks
    chunk="30 days",
    output_dir="output",
    output_averages=True,        # one record per chunk: its 30-day-window mean
    checkpoint_path="checkpoint",
)
```

Per chunk it integrates, labels and writes the output, checkpoints, and checks
the state is still healthy:

```
carry = initial_carry or coupler.initialize()          # or the checkpoint's
for steps in batches:                # `steps` is the chunk, except that the
    first_step = int(carry.step)     #   last batch of a resume can be short
    trajectory = compiled[steps]     # one compiled trajectory per length
    carry, diagnostics = trajectory(carry)
    datasets = chunk_datasets(coupler, diagnostics, first_step=first_step)
    reduced  = postprocess_datasets(datasets, output_averages=…, subsample=…,
                                    first_step=first_step, steps=steps)
    paths += write_chunk(reduced, output_dir, first_step)
    ok, report = health_check(datasets, chunk_index, elapsed_days)   # UNreduced
    if ok or not bail_on_unhealthy:
        if due(int(carry.step)) or last chunk:      # see checkpoint_interval
            coupler.save_carry(carry, checkpoint_path)
```

It returns a `RunResult`: the `final_carry`, `steps_completed`
(`int(final_carry.step)` — the run's position on the clock, including whatever
a checkpoint restored, not the number of steps this call integrated),
`completed`, one `report` per chunk, every `path` written, and the
`accumulator` if the run was given a reduction.

**Chunking rules.** `total_time` and `chunk` are `jem.base.component.parse_duration_days`
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

The gate is given the chunk **as it was integrated** — every record — and not
the thinned or averaged datasets that were written. That distinction is the
difference between a working gate and one that cannot see: `check_health`
judges a chunk by its last record and by extremes, while `output_averages=True`
replaces the chunk with a mean that (xarray skips NaNs) drops a NaN entirely
and dilutes a finite extreme, and `subsample=n` need not keep the last record
at all. An atmosphere that blew up in the last hours of a month would then be
reported healthy and checkpointed. So the loop labels the chunk once with
`chunk_datasets`, hands *that* to the gate, and applies `postprocess_datasets`
only to the copy it writes; `datasets_for_chunk` remains the two composed, for
a caller that wants the reduced form alone. What the gate can resolve is one
**coupling step**: `JCMComponent` integrates each coupling step with JCM's own
`output_averages`, so the records being judged are already step means — a NaN
propagates through that mean, a finite excursion shorter than a coupling step
need not.

**How often it checkpoints.** `checkpoint_interval` (`None` by default: after
every chunk) saves less often than every chunk, for a run whose chunks are short
for one of the *other* reasons a chunk exists — a health check every few days,
an output file per day. It must be a whole multiple of `chunk`, because a chunk
boundary is the only place the loop stops, and it is counted in coupled steps
from the **start of the run** (`carry.step`, which a resume restored) rather
than of the call, so a run stopped and resumed checkpoints at the same points an
uninterrupted one does. Both are checked with the other durations, before
anything is compiled; an interval given with `checkpoint_path=None` is refused
rather than ignored, since it would otherwise leave a run that asked to
checkpoint less often checkpointing not at all.

Two guarantees make the interval safe to reach for, and they are why the loop
keeps the last accepted carry in memory. The last chunk of a **completed** run
is checkpointed whatever the interval says, so a finished run always leaves its
final restart state. And a run the health gate **stops** writes the last chunk
that *passed* before returning, naming the step it holds at INFO — so bailing
still leaves the restart point at the last healthy state, exactly as it does
without an interval. What the interval does give up is a run that is *killed*:
that falls back to the last interval boundary, re-integrates the chunks after it
on the resume and **rewrites** their output files, which is safe precisely
because a file is named after the coupled step its chunk starts at — the second
pass writes the same names from the same starting state.

Safe, that is, while the resume keeps the same `chunk`; the chunk belongs to the
run and not to the checkpoint, so a resume free to change it is also free to
write files at steps the earlier pass's files do not sit on. A checkpoint at
step 4 with one-day files at steps 4 and 5, resumed with two-day chunks, would
overwrite the step-4 file with the records for steps 5–6 and leave the step-5
file holding step 6 a second time — a duplicate no reader of the directory could
tell from a real one, and one no later chunk would ever rewrite. So a resumed
run tests, before anything is compiled, that every file of its own at or after
the restored step is one this call really writes over, which takes both halves
of what it is about to do: the file starts on **its** chunk grid
(`restored_step + k × steps_per_chunk`, which is where its chunks begin, the
short final batch included) **and** before `total_steps`, where it stops. Those
it reports at INFO and rewrites. Anything else is a `ValueError` naming the
files grouped by which half they fail — an overlap in the middle of the run, or
output at or past its end, which a resume asking for less simulated time than an
earlier pass already wrote leaves stranded — together with the restored step,
the chunk and the three ways out: resume under the chunk those files were
written with, remove them, or choose another `output_dir`. Deleting them for the
user was the alternative and was rejected: the driver cannot know which pass's
output is the one worth keeping. Files before the restart point are the run's
history, and files whose names this coupler would never write are neither
examined nor touched — `jem.output.output_file_step` matches a name against the
run's components, following a nested coupler into its inner ones and skipping
any component that has no `to_xarray`, since no file is ever written under such
a name. A run that writes no files at all — an accumulated one, or a call with
nothing left to integrate — is not checked, having nothing it could overlap.

Two configurations the loop cannot honour exactly are warnings rather than
refusals, because neither costs a restart point: a `total_time` that is not a
whole number of intervals (the last gap between saves is simply shorter than the
interval), and a run that starts part-way through a chunk — a resume under a
*different* chunk length, or an `initial_carry` handed in at such a step —
where no chunk before the last can end on a multiple of the interval and the
run would otherwise silently checkpoint only at the end.

The gate also runs **before** the checkpoint, and a chunk it rejects is not
checkpointed (unless `bail_on_unhealthy=False`, where the run carries on and so
must stay resumable). There is only one checkpoint directory and it is
overwritten in place, so saving a rejected state would replace the last healthy
restart point with a broken one and a resume would start from that, fail again,
and have nothing left to go back to. Bailing instead leaves the restart point
at the last chunk that passed, so the run resumes by repeating the chunk that
failed — which is why that chunk's output files are overwritten on the resume,
with the warning `write_chunk` logs.

**What state the run started from.** Resuming a run is the *same command* as
starting one — that is the point of a single checkpoint directory — so nothing
in the command says which happened, and a run that was meant to resume and
silently cold-started repeats simulated time that has already been paid for.
`run_chunked` therefore logs the provenance of the carry it is about to
integrate, at INFO, in exactly one line, before anything is compiled:

```
Starting from coupler.initialize() at coupled step 0 (no checkpoint was given).
Starting from the initial_carry argument at coupled step 3.
Resumed from checkpoint /scratch/run/checkpoint at coupled step 120.
```

A `checkpoint_path` that holds no complete checkpoint says so on the line
before, in as many words: every component starts from its initial state rather
than from a restart (or, if an `initial_carry` was passed, from that). The two
ways of failing to resume are not equally alarming, so they are not equally
loud. A directory an interrupted save left without its carry file is a
**WARNING** — a run died and its last chunk is gone. A path with nothing at it
is **INFO**: with checkpointing on by default into a fresh output directory,
that is what every first run sees, and a warning nobody can avoid is a warning
nobody reads. Both name the path, so a mistyped one is still visible in the
line the run always prints. `Coupler.load_carry` completes the picture
from the other end, naming each component's own source (the shared carry file,
or its own `load_carry`) and the step restored, so no part of a resumed model's
carry is unaccounted for.

**Checkpoints and resume.** `checkpoint_path` is a **single directory**,
rewritten after every chunk the health gate accepts, not a directory of dated
restart points. That is
what makes resuming a run the same command as starting it: point at the path,
and the run either starts from scratch or continues from where it stopped. The
cost is that only the newest state survives; a run that wants a history of
restart points keeps its own directory of them and hands each one in as
`initial_carry`.

Checkpointing is **on by default** (`checkpoint_path="checkpoint"`): a run long
enough to be worth chunking is a run worth being able to restart, and a default
of `None` made losing a week of compute the consequence of forgetting an
argument. A *relative* path — including that default — is resolved against
`output_dir`, not against the working directory. Hydra gives every run a fresh
output directory, so each run gets its own restart directory and two runs
launched from one shell cannot overwrite each other's; resuming is pointing a
second run at the first's output directory
(`coupled_run.output_dir=outputs/2026-09-16/11-04-02`), which is the same
action that would otherwise overwrite its files, so it is never accidental —
and the provenance line says which of the two happened. An absolute path is
used as given, for a run that checkpoints to scratch while writing output
elsewhere; `checkpoint_path=null` turns checkpointing off. Overwriting in place is safe because the carry file is written
last and removed first (see *Carry*), so an interrupted save leaves a directory
the loop refuses to resume from — it logs and starts from the initial carry
instead — rather than a mixture of two steps.

`CoupledCarry.step`, restored from the checkpoint, is the only source of truth
for how far the run has got; nothing is derived from a chunk index or a file
name. What is left is `remaining_batches(int(carry.step), total_steps,
steps_per_chunk)`, so a run resumed with a *different* chunk length — a
perfectly legitimate choice, since the chunk is a property of the run and not of
the checkpoint — runs whole chunks and then one short final batch (one extra
compile, on that batch only), and still stops exactly at `total_time`.

**Output files.** One file per component per chunk,
`<output_dir>/<component>-<first step:08d>.nc`, named after the coupled step
the chunk starts at. That step is the run's clock — the same number the
checkpoint holds and the records are labelled from — so the name is unique
however the run was chunked, and zero-padding it keeps a directory listing in
run order. A chunk *index* would not do: the chunk length belongs to the run
and not to the checkpoint, so a run resumed with a different `chunk` gives the
same simulated time a different index and would write over a file the earlier
run already wrote. `write_chunk` warns when it does overwrite an existing file. That
normally means a rerun into the same directory; the one other way to reach it
is a run killed after a chunk's output was written and before its checkpoint
was, which resumes at the step it already wrote — so the warning reports the
fact without asserting which happened. The
chunk index survives as what it is: a counter for the health check and the log
line. Each chunk is labelled with its own dates, because `first_step` is passed
through to `Coupler.to_xarray`.

`subsample=n` keeps every *n*-th coupling step — every *n*-th step of the
**run**, counting from its start, not of the chunk in hand, which is why
`postprocess` is told the chunk's `first_step` and its number of coupled
`steps` as well as the stride. The cadence then belongs to the run: an
uninterrupted run, the same run in chunks of any length and a run resumed from
a checkpoint all write exactly the same records, and step 0 is always one of
them. (A stride reapplied from each chunk's first record instead gives an
irregular cadence and more output than was asked for — three-step chunks with
`subsample=2` keeping global steps 0, 2, 3, 5 rather than 0, 2, 4.) The unit
is a coupled *step*, not a record: a component the workflow runs `n` times per
coupled step, or a nested coupler's inner steps, contributes `n` records per
step and they are kept or dropped together, so components recording at
different rates stay on one cadence. `postprocess` reads each component's
records per step off its own record count (`len(time) // steps`) and refuses a
count that is not a whole multiple of the chunk's steps, since the step a
record belongs to would then be undefined. A chunk can contain no coupled step
on the stride at all — a `subsample` longer than `chunk` gives that, and so
does the short final batch a resume under a different chunk length ends with —
and such a chunk gets **no file**, rather than a record the run's cadence does
not call for or an empty one that `xr.open_mfdataset` cannot read back. So
`RunResult.paths` holds one file per component per chunk *except* for the
chunks that kept nothing, and the skip is reported at INFO. `write_chunk` also
**removes** a file already at that name: this pass's output for that chunk is
nothing, the name is this chunk's, and on a rechunked resume an earlier pass's
file can sit on the new chunk grid — declared rewritable by the resume check
above — and would otherwise survive holding a record this pass writes under a
different name. That is the only thing the driver deletes, and it is a name it
is itself responsible for.

`output_averages=True` is defined against jcm's meaning of the same word
rather than beside it: jcm replaces each saved record with the mean over its
save interval, labelled at the
interval's end, and the coupler's records are already one per coupling step — so
the coupler's output interval is the **chunk**, and the flag replaces a chunk's
records with their time mean, labelled with the chunk's last time and carrying
the CF `cell_methods = "time: mean"` that says so. That label is the end of the
chunk whether or not `subsample` kept the record sitting there, so a run that
sets both still writes one evenly spaced mean per chunk; what varies is how
many records went into each one, since the number of a chunk's steps the
run-global stride keeps depends on where the chunk falls in the stride period.

The bins are therefore the
chunks: a 30-day chunk gives 30-day-*window* means, whose boundaries drift about
five days a year against the calendar on a 365-day year, not monthly means —
those are `jem.accumulate.monthly_mean(coupler)` (twelve bins, a climatology)
or `monthly_mean(coupler, total_time=…)` (one bin per month of the run), both
of which bin by each record's own label.
In a coupled run the atmosphere's per-step records are *already*
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

**From the driver.** `run_chunked(..., accumulate=monthly)` builds each
chunk's trajectory with the reduction and threads the accumulator across the
chunks, so a ten-year run reduces to twelve monthly means without ever holding
a chunk of diagnostics:

```python
monthly = monthly_mean(coupler)
result = run_chunked(
    coupler, total_time="2190 days", chunk="30 days",   # 73 whole chunks
    health_check=None,                 # required: see below
    accumulate=monthly,
)
means = monthly.finalize(result.accumulator)
```

An accumulated run has no per-step diagnostics — that is the point — and the
three consequences are chosen rather than inherited:

- **No files are written.** There is nothing for `chunk_datasets` to label, so
  `paths` is empty and `output_averages` / `subsample`, which reduce the
  *files*, do nothing (the run warns if they were set). The reduction is the
  output.
- **The health gate cannot run**, so `accumulate` together with a
  `health_check` is a `ValueError` rather than a gate quietly skipped. The gate
  defaults to *on* and a long accumulated run of an atmosphere is exactly the
  run that needs it, so losing it has to be something the caller asked for —
  `health_check=None` — not something a log line mentions weeks too late.
- **The accumulator is not checkpointed.** The checkpoint is the model's
  restart state; the accumulator is an analysis product. Storing one in the
  other would make the checkpoint format depend on which reduction a run chose
  — a restart file loadable only by a run asking for the same means — and let a
  restart corrupt an analysis. So the carry is checkpointed as usual, a resumed
  run starts a fresh accumulator and covers only what it integrates, and it
  **warns** when it does — a first accumulated run, which has lost nothing yet,
  is told the same fact at INFO. A mean across a restart boundary is built by
  finalizing each call's accumulator and combining them, or by running the span
  in one call.

`monthly_mean` takes only the coupler: the accumulator's shapes come from
`jax.eval_shape` of one coupled step, and a record's month is read directly off
its own **midpoint** — the same instant `TimeAxis` labels the record with —
which is what makes `monthly.finalize(...)` equal
`to_xarray(...).groupby("time.month").mean()` of the same run **by
construction**, for every calendar, rather than only for a run whose labels
happen to agree with a fixed table (2026-09 jax-gcm-878 migration review, item
A). This was **not** always true: before that review, this reduction bound a
record by its interval's *end*, which was the pre-878 label too, but jax-gcm
PR 878 moved `TimeAxis`'s own label to the midpoint without this reduction's
bin rule following — so for one migration round the two genuinely disagreed at
every month boundary, which the review caught and fixed rather than leaving as
a documented gotcha. Two implementations share the underlying arithmetic:
`jem.accumulate._midpoint_month_rule` for the fixed-length calendar
(`365_day` — the only fixed-length calendar a `Coupler` accepts; `month_lengths`
can still build a 360-day table, but there is no `"360_day"` calendar name
anywhere in jem — a static table of month boundaries, compared in **days**
— not seconds, since a sequential accumulator's own boundary-seconds period
can itself run well past `2**31` for a multi-decade run — against a record's
own midpoint via `jem.base.calendar.gregorian_instant`'s int32-safe limb
(schoolbook) multiply-then-divide) and `_gregorian_month_rule` for `gregorian`
(below), which uses
`gregorian_instant` directly. (This replaced a genuinely different,
smaller-range decomposition in a 2026-09 fix — see `gregorian_instant`'s own
docstring's **History** note and `max_safe_record` for the current, tested
bound.) Without `accumulate`, the generated function is what it always was.

**`gregorian` now works in-scan, exactly, real leap years included** — no
fixed table, no "coupling step divides the year" restriction. This closes what
used to be this reduction's sharpest limit: a `NotImplementedError` on *any*
`gregorian`-calendar coupler, which jax-gcm PR 878 made the common case rather
than a corner one (jax-gcm's own atmosphere clock is unconditionally Gregorian,
so every `Coupler` built with a real `jcm.model.Model` must itself use
`calendar="gregorian"`, and it is now `Coupler`'s own default too). The fix
(the migration review's item A) was to stop trying to build a *table* of month
lengths for a calendar whose year is not a fixed number of days, and instead
read a record's real Gregorian `(year, month)` directly off its own midpoint,
via `jem.base.calendar.gregorian_instant` (an int32-safe
reduce-before-multiply, vendored from jax-gcm's own `jcm.date` algorithm so
`jem.base` and `jem.accumulate` need no jax-gcm import to do it) and
`gregorian_ymd_from_days` (the Fliegel & Van Flandern (1968) integer
algorithm). The twelve-bin climatology needs nothing further — every record's
own real calendar month *is* its bin, 0-indexed January first. The sequential
form's bin **count**, for `total_time=`, is computed on the **host**, with
Python's own `datetime` (exact, and a bin count is static, so there is no
reason to do calendar arithmetic in jit for it): the calendar month of the
run's first and last records' own midpoints,
`(last.year - first.year) * 12 + (last.month - first.month) + 1`.

Separately, on the fixed calendar (`365_day`) only, the *bins'*
calendar can still disagree with the *labels'*: the labels are always
proleptic Gregorian (above, and jax-gcm#449), while this calendar's bins
are its own fixed table's months. On a `365_day` run started on 1 January
2000 — where the shipped examples start — the record whose midpoint is the
real Gregorian leap day, `2000-02-29T12:00`, is one the model's own fixed
calendar (no 29 February) calls 1 March and bins into March. `groupby
("time.month")` of the written output and this reduction's own bins therefore
part company only at February (29 records under the real labels' calendar
against the model's 28) and, if the run is exactly one model year long,
December (short one real day, since the real year is 366 days and the model's
is 365) — a much narrower residual than before the midpoint rebinding, which
used to cascade the mismatch through every month from March on. Reproducing
`finalize` from the written output across a leap day still means binning by
model day-of-year rather than by `time.month`. `gregorian` has no such
mismatch at all: the bins and the labels are the same real calendar. Nothing
here depends on whether #118 (calendar-consistent labels on JEM's own fixed
calendars) or jax-gcm#449 (the same inconsistency in JCM's own output, tracked
there only) is ever taken up.

**Twelve bins or one per month of the run.** `monthly_mean(coupler)` bins into
the twelve calendar months, so a ten-year run composites its ten Januaries into
bin 0 — a climatology, and what a fixed `(12, …)` accumulator is for.
`monthly_mean(coupler, total_time="3650 days")` (or `n_months=`) instead gives
the months the run passes through, in order, each with a bin of its own,
starting with the month of the run's own first record (its midpoint, to be
precise — see `monthly_mean`'s **Sequential-form bin 0**). (`"3650 days"`, not
the more readable `"10 years"`: `total_time` must be a whole number of
coupling steps — `monthly_mean` refuses one that is not, exactly as
`run_chunked` refuses the same duration for its own `total_time`/`chunk` — and
on `"gregorian"`, JEM's own duration parser's fixed-average year makes `"10
years"` `3652.425` days, never a whole number of daily coupling steps.) It is
sized by counting the calendar months the run's record midpoints touch, which
is why 3650 days gives exactly **120** bins, not 121: the run's last record's
own midpoint is `total_time - dt/2`, half a coupling step *short* of the
3650-day boundary, so it never spills into an eleventh year the way the pre-migration
end-of-interval convention's boundary record used to. A run longer than the
accumulator wraps at the **span** of its bins, exactly as a windowed mean wraps
at the span of its windows — so a wrapped bin lines up with a calendar month
only when `n_months` is a multiple of twelve, and otherwise holds parts of two
(six bins from 1 January span 181 days, and the second August of the run
splits 28 records into the February bin and 3 into March's). `total_time`,
which sizes the accumulator so that it is never wrapped into, is the form to
prefer. On the fixed calendars, that span is rounded up to the next whole
coupled step, because the record counter is reduced modulo it and a whole
number of calendar months need not be a whole number of steps (a 5-day
coupling divides the 365-day year but not 59 days of January and February);
the wrap moves by less than one step, every bin boundary stays exact, and
nothing that is meaningful in the first place can see it. `gregorian`'s
sequential form needs no such rounding: a wrapped bin there is `jnp.mod` of a
month *count*, not of an elapsed-time span.

**Any fixed set of bins, not only the months.** A calendar month is one binning
of a run; a sub-seasonal forecast is scored on another — 5-day and 7-day means.
Both are the same reduction with a different step-to-bin rule, so
`jem.accumulate` is one private `_build_binned_mean(coupler, bin_of_record, n_bins)`
under two public builders, returning the same `BinnedMean` named tuple with the
same `finalize`:

```python
from jem.accumulate import month_lengths, monthly_mean, windowed_mean

monthly = monthly_mean(coupler)                                  # 12 bins
months  = monthly_mean(coupler, total_time="3650 days")          # 120: every month
pentads = windowed_mean(coupler, "5 days", n_windows=73)         # a year of them
weeks   = windowed_mean(coupler, "7 days", total_time="1 year")  # 53: the last is short
leads   = windowed_mean(coupler, [1, 1, 1, 1, 1, 1, 1, 5, 5],    # a pattern, cycled
                        total_time="30 days")
```

`window` is a `jem.base.component.parse_duration_days` string or a number of days, parsed
on the coupler's calendar, and must be a whole number of coupling steps — a
window ending part-way through a step could only be filled by splitting that
step between two windows. The accumulator's size is `n_windows`, given directly
or counted from `total_time` (rounding *up*, so a run that does not divide into
whole windows still has a bin for the one it ends inside; that bin is divided by
its own count, so it is the mean of what fell in it).

**The windows need not be equal.** `window` may be a *sequence* of lengths,
which the accumulator's windows cycle through, repeating for as long as
`n_windows` (or the count from `total_time`) asks and wrapping only at the sum
of all of them — daily leads for a forecast's first week and pentads
thereafter, say. A sequence is the one case in which giving neither
`n_windows` nor `total_time` is answerable, and it means one cycle of the
pattern.

**A window is not a calendar month, whatever its length.** Every window is
measured from the run's own start date, with no phase and no reference to the
calendar, so the month lengths of `month_lengths()` — which are always
January-first — are calendar months only for a run starting at 00:00 on 1
January; from 1 July they would bin the first 31 days together, then 28. That
is why the per-month reduction is `monthly_mean(coupler, total_time=…)` and not
a pattern handed to `windowed_mean`, and why `windowed_mean` has no `offset=`
knob to fix it with: the builder that knows where in the calendar a run starts
is the one that should own the phase.

**The two bin rules are genuinely different arithmetic, on purpose, since the
2026-09 migration review.** `windowed_mean` bins a record against its
interval's **end**, in elapsed run-time — unrelated to, and independent of,
whatever instant `TimeAxis` happens to write the record's label as (see that
class's docstring: since jax-gcm v3, PR 878, that label is the interval's
midpoint, not its end). `monthly_mean` bins a record by its own **midpoint**
instead, because a monthly mean is required to equal `groupby("time.month")`
of the *same* written output, which moved to the midpoint too — keeping
`monthly_mean` on the old end-of-interval rule after `TimeAxis` moved would
have made the two disagree at every month boundary rather than only across a
Gregorian 29 February. A useful consequence: at any boundary both a window
and a calendar month actually land on (a month-length window pattern from a 1
January start, say), the two now agree exactly — a record ending precisely on
the boundary has its midpoint half a record *before* it and the next record's
midpoint half a record *after*, so "ends at or before" and "midpoint is
before" give the same answer on both sides. Before this migration the two
rules shared one convention (both bound by the interval's end) but closed a
shared boundary in *opposite* directions, so a 31-day window and January
differed by exactly the record on their shared boundary; that historical
difference is what `windowed_mean`'s own `test_a_month_long_window_and_a_month
_now_agree_at_their_shared_boundary` test used to pin, under its old name.

`windowed_mean` is the private
`_variable_window_rule(boundaries_seconds, offset_seconds, inclusive)`: bins
laid end to end as a cumulative sum of lengths, a phase (0, since a window is
measured from the run's own start) and which side a boundary closes on
(`"right"`, the only mode this function is exercised with in production now).
Inside the scan it is a `searchsorted` in a static table, after the record
counter is reduced modulo the records in one period of the bins — which is
both what wraps a long run and what keeps the arithmetic inside int32. The
boundaries themselves are converted from seconds to record counts on the host,
in int64, so nothing in the traced code multiplies a counter that grows with
the run: a table of seconds would pass 2³¹ after 68 simulated years and wrap
to nonsense. `monthly_mean` used this same function (`inclusive="left"`)
before the 2026-09 migration review; it now uses two different rules instead
(below), because a record's midpoint is a *half*-record shift from its end,
which the record-count conversion above cannot express (it only supports a
whole-record shift).

**`monthly_mean`'s own bin rules.** `_midpoint_month_rule` (the fixed
calendar, `365_day`) compares a record's midpoint in **days**, not
seconds or record counts (2026-09 migration review, round 2: comparing in
seconds, as this rule first did, overflows int32 for a sequential
accumulator's own boundary-seconds period past about 68 simulated years,
regardless of the coupling step) — `jem.base.calendar.gregorian_instant`
gives the record's own (day, second) exactly, and a `searchsorted` against
the pattern's own boundaries, rounded up to the day (so a pattern's
occasionally fractional-day last boundary can never place a record past the
last bin), decides the bin; the second is not needed, since every month
boundary but that possibly-fractional last one falls exactly at midnight.
`_gregorian_month_rule` (`gregorian`) needs no period or table at all:
`jem.base.calendar.gregorian_instant` — the same limb (schoolbook)
multiply-then-divide, exact for any traced record counter an int32 can hold,
not merely one reduced modulo some period first — gets the record's midpoint
as an exact (days, seconds) pair, and `gregorian_ymd_from_days` reads its
real calendar month directly off that.

**A coupled step is not always one record.** A component the workflow runs
*n* times per coupled step emits *n* records, each with its own sub-interval,
and a nested coupler's inner steps are records in the same way — so a coupled
step's records need not all fall in the same bin. The 24 hourly records of the
daily step covering 31 January have midpoints half an hour past each hour: 23
of them (through `22:00-23:00`) fall before midnight and one
(`23:00-00:00`) after, exactly as `to_xarray` labels them (their own
midpoints). Each record is therefore binned by **its own** interval (its
midpoint, for `monthly_mean`; its end, for `windowed_mean`), from the
coupler's own sub-step clock (`coupling_time_at_substep`), and the sub-step
axis is kept rather than folded: that component accumulates into `(n_bins, n,
…)`, bin *b* slot *j* holding the records of call *j* that fell in *b* — a
monthly-mean diurnal cycle, which folding would destroy and which
`fold_records(means["atm"], counts["atm"])` recovers by weighting each slot
with its own count (a straight mean over the slots is the same number only
where every slot holds the same number of records, which is exactly what a
month boundary breaks). Because components recording at
different rates fill different bins as one step is folded in, the accumulator's
counts are then one array per component (`(n_bins, *sub-step axes)`) instead of
the single `(n_bins,)` array a model whose components all record once per
coupled step keeps.

A run longer than the accumulator **wraps**: window *w* also collects windows
*w + n_windows*, *w + 2·n_windows*, … exactly as the twelve-month table wraps
years and gives a three-year run a January climatology, and as `n_months` bins
wrap at their own span (above). That is the price of a fixed-size accumulator, which is
the whole point of reducing inside the scan — size it to the run if each bin is
to stand on its own.

**Calibrating against a binned mean.** The accumulator is an ordinary pytree in
the scan carry, so nothing about the reduction is special to differentiate:
`jax.grad` of a monthly mean reaches a component parameter through the
reduction exactly as it reaches one through the trajectory. A July SST
calibration against a target, with a process parameter varied in the carry the
way *Parameters: process and initial-condition* describes:

```python
monthly = monthly_mean(coupled)
trajectory = coupled.generate_trajectory_function(365, accumulate=monthly)
JULY = 6              # finalize()'s leading axis is January first

def loss(relaxation_time, target_july_sst):
    # `ocn` is the SlabOceanModel registered in `coupled`; replacing the
    # parameters it is initialized with is what makes them differentiable.
    params = ocn.params.replace(relaxation_time=relaxation_time)
    _, accumulator = trajectory(coupled.initialize({"ocn": params}))
    july = monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature[JULY]
    return jnp.mean((july - target_july_sst) ** 2)

gradient = jax.grad(loss)(relaxation_time, target_july_sst)
relaxation_time = relaxation_time - learning_rate * gradient   # one plain step
```

`tests/unit/test_accumulate.py` runs exactly this — the gradient of an
accumulated July mean equals the gradient of the same quantity computed from
the stacked diagnostics, and one descent step reduces the loss — so the snippet
cannot rot. Two practical notes it makes. `relaxation_time` is of order 1e6
seconds while the loss is a few K², so `learning_rate` has to be scaled to the
parameter (a real calibration hands the gradient to an optimizer, which does
that for it); and for a long calibration `remat=True` on the trajectory trades
recomputation for the memory the backward pass would otherwise need.

Which parameter is varied *how* is the distinction in *Parameters: process and
initial-condition* above: `relaxation_time` is read every step out of
`carry["params"]`, so it could equally be replaced in the carry, while an
initial-condition parameter has to go through `initialize` as it does here.
Building the model inside `jax.grad` is not an alternative — a constructor
validates its parameters, which needs concrete floats.

## Configuration

Python is the primary interface. The configuration layer is a thin wiring layer
over it, and `python -m jem.main` (or the `jem` console script) is one command
for a coupled run:

```bash
python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run
```

**Exit status.** `0` if the run reached `total_time`, `1` if the health gate
stopped it early — `jem.main` raises `SystemExit(1)` after logging the last
report. A run stopped by the gate keeps everything it wrote, but it did not do
what it was asked to, and the exit status is the only thing a queue system, a
CI job or a shell `&&` can see.

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
`coupled_run=short_run` selects an option, `coupled_run.total_time="90 days"` sets
one key — and `test_jcm_run_group_is_not_shadowed` pins it down. The *options*
are named apart for the same reason: `short_run` and `long_run` against jcm's
`smoke` and `longrun`, so no override reads as though it might be configuring
the atmosphere.

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

**A grid goes only to a component that asks for one.** `_accepts_grid` resolves
the node's `_target_` (a class or a classmethod — both spellings occur) and
looks for an explicit `grid` parameter in its signature; a `**kwargs` catch-all
does not count, because that is the signature that swallows the keyword and
fails elsewhere. `VerosComponent.from_setup` forwards every keyword it does not
recognise to the Veros setup factory, so an injected `grid=` would have been
rejected *there*, with a message about the factory. A component that takes no
grid does not get one **built** either: it brings its own bathymetry and
land-sea mask, and a `SlabGrid` made from the atmosphere's geometry would
describe a grid nothing runs on.

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
it calls, the one private name it still depends on (`cf_metadata._COORD_ATTRS`,
whose values the slab components copy so their axes describe themselves exactly
as the atmosphere's do), the physics diagnostics fields the surface exchange is
read out of, the package data it resolves, and the constructors its documented
workflow asks a user to call. Each entry says what it is used for,
which is what makes it possible to decide whether an entry may be deleted.

`tests/unit/test_jcm_contract.py` walks that list against the installed `jcm`,
so a jax-gcm rename fails as "jax-gcm renamed or removed X, which JAX-ESM used
for Y, at revision Z" — at the cheapest possible moment, rather than mid-run.
The pin is a `dev` sha because no tagged jax-gcm release carries the changes
JAX-ESM is written against. It is the `dev` commit that merged jax-gcm PR 878
(`808412a5`) — the merge commit itself rather than whatever `dev` was at bump
time, since later unrelated `dev` commits have not been checked against this
code. That revision carries #750's one run schema and `configuration` group,
#763's input-resolution engine, #819's removal of jax-gcm's own logging
configuration and #824's public resumable state and date conversion; PR 877's
two changes, jax-gcm#754's package-independent `SurfaceExchange` struct (see
"The JCM adapter" below) and #884's declared forcing-alignment rule; and PR
878's exact Gregorian `RunState` clock, `Model(start_time=)` and midpoint
output labels (see "The JCM adapter" and the output-time conventions). Under that rule, `jcm.forcing.resolve_align`'s `auto` no longer infers
climatology-vs-transient from a file's time axis, and raises for any file it
cannot resolve from jax-gcm's own data-mirror manifest — see the
`forcing.align` comments in `jem/config/configuration/{earth-slab,
veros-double-drake,veros-earth}.yaml`. `pyproject.toml`'s `jcm>=3.0.0rc1` is the
loosest true statement of the version, since jax-gcm bumps its version string
only at release. Every required CI job checks that revision out through a
workflow-level `JCM_REV`, which the test asserts equals `JCM_SUPPORTED_REV`,
and a non-blocking `canary-jcm-dev` job keeps tracking `dev` so drift stays
visible without blocking a pull request. `contract.py`'s docstring is the
procedure for bumping the pin.

## The JCM adapter

`jem/components/jcm/component.py` wraps a `jcm.model.Model` (the spectral
atmosphere from jax-gcm) as `JCMComponent`. It is a wrapper object, not an
in-place adaptation: the atmosphere JEM drives is the same object the user
configured, and nothing in JCM has to know JEM exists. Its carry is:

```python
{
    "state":   <jcm modal (spectral) dycore state>,
    "physics": <jcm's cross-step physics carry, threaded, opaque>,
    "time":    <jax_datetime.Datetime; jcm's exact RunState.time, threaded>,
    "step":    <jax.Array int32; jcm's exact RunState.step, threaded>,
    "forcing": <jcm ForcingData; holds sea_surface_temperature, sice_am, ...>,
    "derived": JCMDerived(physics, total_heat_flux, total_freshwater_flux,
                          evaporation, precipitation, u0, v0),
}
```

`initialize()` builds those pytrees from the `(dycore_state, physics_carry)`
pair `Model.bootstrap_state()` returns (seeding `"time"`/`"step"` at
`model.start_time` / `0`), plus a structural template of the diagnostics dict;
it does **not** integrate. Each `step` calls `model.run_from_state_with_carry()`
with the coupling interval as both `save_interval` and `total_time` and the
carry's `"time"`/`"step"` as `initial_time`/`initial_step` — required
explicitly since jax-gcm PR 878 (the exact `RunState` clock; earlier revisions
inferred them from the incoming dycore state) — so JCM sub-steps internally at
its own timestep and returns exactly one saved record per coupling step, along
with the `RunState` to carry into the next call. `step` then reads the surface
exchange out of the returned physics diagnostics. Threading `"time"`/`"step"`
rather than recomputing them from the coupler's own step counter each call is
deliberate: JCM's `RunState` is the authoritative clock since jax-gcm v3 (PR
878), and jax-gcm's own migration guide (`docs/source/v2_to_v3.rst`, "One real
datetime clock") is explicit that a caller should keep threading it rather
than deriving it elsewhere — not because recomputing it would overflow (an
earlier draft of this note said so; the CHANGELOG retracts it, since
`jem.base.calendar.gregorian_instant`'s own int32-safe reduce-before-multiply
decomposition computes exactly this instant from the coupler's own step count
for `JCMComponent._report_authoritative_clock_drift`'s drift check below, so
recomputing was never the obstacle) — but because threading is what
guarantees JCM's clock and the coupler's can never disagree about what instant
a step is at, which recomputing one from the coupler's own step count cannot
once a checkpoint or a differently-configured coupler is involved — see
`jem.components.jcm.component`'s module docstring.

That read is isolated in `jem/components/jcm/exchange_fields.py`, which since
jax-gcm#754 (PR 877) is a single reader,
`from_diagnostics()`, off jax-gcm's own package-independent
`diagnostics["surface_exchange"]` struct — published identically by every
physics package that resolves a surface (SPEEDY, ECHAM; Held-Suarez opts out).
It replaces the pre-#754 `speedy()`/`echam()`/`detect()` readers (git history,
commit `756cc2c`), which reached into each package's own private diagnostics
by hand and could not build a struct for ECHAM at all — its reader always
raised `NotImplementedError`. Translating jax-gcm's contract to JEM's
conventions is now a two-line sign flip and nothing else: heat flux
**positive upward** (jax-gcm's `net_heat_flux` is positive *down*, so it is
negated exactly here), while `evaporation`/`precipitation` and wind need no
unit conversion any more — the published contract is already `kg m-2 s-1` and
already the convective+large-scale/stratiform total, computed once by the
publisher rather than assembled here from two package-specific diagnostics
entries. The module's own docstring has the full field-by-field derivation and
the evidence for it (verified against a real SPEEDY step, not just read off
the source).

One package-specific read remains, and is not expected to go away with a
mechanical jax-gcm update: jax-gcm's contract publishes only the *scalar*
`wind_speed`, never a near-surface wind *vector*, so
`jem.fluxes.bulk_wind_stress` (the independent bulk-drag law
`jem.fluxes.VerosExchange` applies for a Veros ocean) still reads SPEEDY's
private `_surface_flux.u0`/`.v0` directly. ECHAM has no wind vector anywhere
in its own diagnostics either (its boundary layer scheme diagnoses only a
speed), so this is not a regression from the #754 collapse — it predates it,
and is the reason the pre-#754 `echam()` reader could never have supplied a
wind vector either, even if it had had a heat/water struct to read.

The consequence is wider than the Veros exchanger, though. `from_diagnostics`
reads the wind *eagerly*, and `JCMComponent.step()` calls it on every coupled
step to fill `JCMDerived.u0`/`.v0`. So **no ECHAM-composed coupled model can
complete a step, whatever it is coupled to** — a slab ocean as much as Veros.
ECHAM *publishes* the grid-mean heat and water fluxes; it is the wind read
that fails. Every shipped JAX-ESM configuration composes SPEEDY, so nothing
shipped is affected. Making the wind optional — through `JCMDerived`, the
coupled carry and the output — is tracked in jax-esm#129.

Every JCM *attribute* the wrapper touches is public at the pinned revision,
apart from that one underscore-prefixed diagnostics key
(`_surface_flux.u0`/`.v0`, above): the initial state and physics carry are the
pair `Model.bootstrap_state()` returns, and a stacked `ModelPredictions` is
repaired with `ModelPredictions.with_context(model)`. jax-gcm#824 is what made
both public, and each is a `JCM_INTEGRATION_POINTS` entry, so a JCM refactor
that moved one fails the contract test by name instead of inside somebody's
run.
The same state and carry are also installed on the model as
`Model.dycore_state` / `Model.physics_carry`; the wrapper takes them from
`bootstrap_state`'s return value and never reads those attributes, so they are
not watched. `with_context` also stamps the atmosphere dataset's
`jcm_prov_params` attribute with JCM's `parameters_rederived_from_live_context`
note, which is the truthful record for a coupled run: the trajectory is traced
once and scanned, so those parameter values are read from the live physics
afterwards rather than captured at trace time.

The atmosphere's output keeps JCM's own `time` labelling (`JCMComponent.to_xarray`
hands the stacked predictions straight to jax-gcm's own `ModelPredictions.to_xarray`),
and `TimeAxis.datetimes()` — which labels every *other* component's output, so
it merges with the atmosphere's — now **calls** jax-gcm's own labelling
conversion instead of reimplementing it. Through jax-gcm PR 877 this was a
decision rather than a gap: the only public conversion at the time,
`Model.date_from_sim_time` (jax-gcm#824), was the **model clock** conversion —
exact integer day/second arithmetic, returning a `DateData` for forcing and
physics — not what JCM's own output files were labelled with (a float64
days-since-epoch product, inexact past a 128 ns ulp, internal to
`ModelPredictions._trajectory_dataset`), so adopting it would have merged with
JCM's own output only when the coupling step happened to be a power-of-two
fraction of a day. jax-gcm PR 878 published the labelling conversion itself,
`jcm.predictions.output_time_labels` (closing jax-gcm#862), and moved JCM's own
averaged output from an end-of-interval label to a midpoint-of-interval one (see
the time-label bullet above); `TimeAxis.datetimes()` uses that function
directly, which is exact for *any* coupling step and picks up the midpoint
convention automatically. `Model.date_from_sim_time` itself is now a
compatibility adapter only — jax-gcm's own exact clock is threaded
incrementally (`time = time + dt`) rather than recomputed from elapsed
seconds, so nothing in JCM's own integration path calls it any more; a
perpetual-season (frozen seasonal cycle) override hook built on it, once
tracked as jax-esm#120, would be a silent no-op today and needs a different
mechanism if it is still wanted.

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

See {doc}`../adding_a_component` for the narrative version of this checklist,
worked through end to end for JCM; keep the two in sync when either changes.

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
   time)` if it produces output, and `save_carry`/`load_carry` if its carry
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
