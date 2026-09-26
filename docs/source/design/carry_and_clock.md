# The carry, parameters and the clock

How a component's state travels between coupling steps, how its tunables are
varied and differentiated, and how the coupled model tells the time.

## The carry

Every component owns a **carry**: a pytree (by convention a plain `dict`)
holding everything that must survive from one coupling step to the next. The
coupler never inspects it — `Component.step(carry, time) -> (carry,
diagnostics)` is the whole contract — but every packaged component follows
the same convention, because it is what keeps an exchanger readable:

```python
{
    "params":  <the component's tunable parameters>,   # slab models
    "state":   <the component's prognostic state>,
    "forcing": <what other components send in>,
    "derived": <diagnostics other components read out>,
}
```

An exchanger only ever moves a `derived` (or `state`) field of one component
into a `forcing` field of another; `jem.exchangers` addresses fields by that
convention (`"atm.derived.total_heat_flux"`).

The carry holds more than the mathematical state — anything that must
participate in differentiability, and anything cheap to keep but expensive to
rediagnose:

- The slab models put their `flax.struct` parameters in `carry["params"]`
  rather than closing over them, so `jax.grad` of a coupled run with respect
  to, say, `SlabOceanParameters.relaxation_time` works with no special casing
  in the coupler (see *Parameters* below).
- `JCMComponent`'s carry has two keys beyond the shared convention:
  `"physics"`, JCM's own cross-step physics carry (sub-cycled radiation,
  prior-step TKE, tendencies handed from one term to the next), threaded
  straight back into `Model.run_from_state_with_carry` because dropping it
  between coupling steps would reset that memory once per coupling interval;
  and `"time"`/`"step"`, jax-gcm's own clock (a `jax_datetime.Datetime` and a
  step counter), fed back in as `initial_time=`/`initial_step=` on the next
  call so JCM's internal sub-stepping continues exactly where it left off.
  `JCMComponent.step` logs an error (via `jax.debug.callback`, since the
  check runs inside the traced scan) if that carried clock ever disagrees
  with the coupler's own — which can only happen if the carry came from
  somewhere else, such as a checkpoint restored under the wrong start date.
- `JCMComponent`'s `carry["forcing"]` is a whole `jcm.forcing.ForcingData`, so
  an exchanger addresses a boundary condition under JCM's own field name
  (`atm.forcing.sea_surface_temperature`). It is the one section that is
  both *read from a file* and *overwritten every step*, and the two want
  different shapes: with `forcing@atmosphere.forcing=from_file`, JCM builds
  each time-varying boundary condition as a `jcm.forcing.TimeSeries` (values,
  time axis, alignment mode) and slices it by date every internal timestep,
  while an exchanger writes one plain `(ix, il)` array — a field that is a
  time series before the exchange and an array after changes the carry's
  pytree structure, which `lax.scan` cannot carry (see *Exchangers* in
  {doc}`exchange`). `JCMComponent.set_exchanged_forcing(names)` declares
  which fields the coupling supplies, and `initialize()` collapses exactly
  those to the climatology at the run's start date; every other field keeps
  its time series, so a land surface in a run built with `land=none` still
  follows the seasonal cycle. The names are a property of the *coupled
  model*, not the atmosphere — `jem.runners.build_coupler` reads them off
  the built coupling table with `jem.exchangers.exchanged_fields(exchangers,
  "atm")` — so a hand-written exchanger has to declare the same set
  explicitly.

## Parameters: process and initial-condition

A component's parameters divide into two kinds, and which one a given
parameter is decides how a parameter study varies it. The framework does not
enforce the distinction — it follows from *when* the parameter is read:

| | Read by | Varied by | Example |
|---|---|---|---|
| **Process parameter** | `step`, out of `carry["params"]`, every step | replacing that leaf in the carry | `SlabOceanParameters.relaxation_time` |
| **Initial-condition parameter** | `initialize`, once | passing parameters to `initialize` | `SlabOceanParameters.initial_sst` |

A process parameter is varied in the carry, because that is where `step`
reads it from:

```python
carry = model.initialize()
carry["params"] = carry["params"].replace(relaxation_time=tau)   # differentiable
```

An initial-condition parameter **cannot** be varied that way: by the time a
carry exists its value has already been copied into the state, and `step`
never looks at it again, so replacing the leaf changes nothing and a gradient
with respect to it is zero. It is varied by handing the parameters to
`initialize`, which builds the initial state from them *and* puts them in
`carry["params"]`, so the state and the process parameters come from one
object:

```python
def loss(initial_sst):
    params = ocn.params.replace(initial_sst=initial_sst)
    _, diagnostics = trajectory(coupled.initialize({"ocn": params}))
    return jnp.mean(diagnostics["ocn"]["state"].sea_surface_temperature)

jax.grad(loss)(jnp.float32(288.15))     # non-zero
```

`Coupler.initialize(params)` takes `{component name: that component's
parameters}` and routes each one to that component's `initialize(params=...)`;
a component the mapping does not name initializes exactly as it would with
no argument, and a name the coupler has no component for is a `ValueError`.
Not every component accepts parameters — a wrapper around an external model
initializes from that model's own state — so naming one that cannot is a
`TypeError`. For a nested coupler the value is itself a mapping over its
components (`{"atm_lnd": {"atm": params}}`), because that is what its own
`initialize` takes.

Building the model inside `jax.grad` is not an alternative route to the same
gradient: a constructor validates its parameters as concrete Python floats,
which cannot be done to a traced value. Validation stays at construction,
where the values are concrete, and `initialize(params)` is the
differentiable entry point.

## The component contract

`jem.base.component.Component` is a runtime-checkable `typing.Protocol`, so
"implementing" it means having the right attributes — there is no base class
to inherit from, and nothing is monkey-patched onto a wrapped model:

| Member | Signature | Purpose |
|---|---|---|
| `name` | `str` | The component's name in the workflow, carry and output |
| `initialize()` | `() -> Carry` | Build the initial carry. Must not integrate |
| `step(carry, time)` | `(Carry, CouplingTime) -> (Carry, Diagnostics)` | Advance one coupling step |

`Coupler.add_component(name, component)` checks `isinstance(component,
Component)` and raises `TypeError` naming the missing members. The object
itself is stored, so `coupler.components[name] is component`.

`initialize()` must be callable with no arguments — that is all the protocol
asks. A component may additionally accept `initialize(params=...)`, which is
how an initial-condition parameter is varied (above); the slab models and
`Coupler` itself do, `JCMComponent` and `VerosComponent` do not, since they
initialize from the wrapped model's own state.

Three capabilities are **optional**, and are tested for with `isinstance`
against their own protocols at the one place that uses them — never with
`hasattr` at a random call site:

| Protocol | Member | Who implements it |
|---|---|---|
| `SupportsXarray` | `to_xarray(diagnostics, time) -> xr.Dataset \| Mapping[str, xr.Dataset]` | slab models, `JCMComponent`, `VerosComponent`, `Coupler` |
| `SupportsBind` | `bind(*, coupling_timestep, start_date)` | `JCMComponent`, `VerosComponent`, the slab models |
| `SupportsCheckpoint` | `save_carry(carry, directory)` / `load_carry(directory)` | `VerosComponent`, `Coupler` |

`bind` is called by the coupler once per component, from `add_component`
(hence from the constructor for everything passed to it), and it is the only
way a component learns about the clock outside a step. A component with an
internal timestep uses it to derive the coupling interval — `JCMComponent`
converts it to the number of days it passes to JCM as
`save_interval`/`total_time`, `VerosComponent` to a count of tracer
timesteps — and it is where a disagreement about the clock is refused: both
wrappers raise `ValueError` if the coupling timestep is not a whole multiple
of the model's own, and `JCMComponent` additionally refuses a `start_date`
that differs from the model's own `start_time`. The slab models use it only
for the start date: `initialize()` takes no argument, so `bind` is how a run
starting on 1 July samples the July record of its climatology rather than
January's, through `SlabModelBase.start_year_fraction` — computed by the
same `jcm.date.fraction_of_year_elapsed` function that `CouplingTime.year_fraction`
calls on the coupled clock, so a climatology sampled in `initialize()` and
one sampled in `step()` cannot disagree about where the run starts. A model
that was never registered with a coupler reads 1 January, which is what a
bare `model.initialize()` in a test or notebook gets.

`step` must be a pure function of `(carry, time)` and must return a carry
with exactly the pytree structure, shapes and dtypes it received, or
`lax.scan` rejects it.

`to_xarray` normally returns one dataset, keyed in the coupler's output by
the name the component is registered under. A component that is itself a
coupled model — a `Coupler` nested in a slower one — has no single dataset
to return, so it may return a **mapping** of name to dataset instead, which
the outer coupler flattens into its own result under those names (a
collision with a name already there is a `ValueError`). See {doc}`nesting`.

## The clock

The coupler owns the only clock. Components hold no start date, no timestep
and no calendar of their own, so two of them cannot disagree about the date.
The coupled state, `CoupledCarry`, carries it directly:

```python
@struct.dataclass
class CoupledCarry:
    components: dict[str, Carry]   # one carry per component, keyed by name
    time: jdt.Datetime             # the coupled clock; jax_datetime's
                                    #   proleptic Gregorian -- there is no
                                    #   other calendar to choose
    step: jax.Array                # int32; coupled steps completed
```

`time` is a carried instant, advanced by the coupling timestep every step —
never a step count multiplied out — so it survives a chunked run or a
checkpoint restart exactly (a `lax.scan` index restarts at zero on every
call; the carry does not). `step` rides along for the sub-step indexing that
workflow multiplicity and a nested coupler need, and as the authoritative
count of how far a run has got; nothing is derived from a chunk index or a
file name.

Each `step` is handed a `CouplingTime` built from that carried clock:

```python
@struct.dataclass
class CouplingTime:
    step: jax.Array          # int32, coupled steps completed before this one
    time: jdt.Datetime       # the coupled clock at the start of this step
    sim_time: jax.Array      # seconds since start_date; equals step * dt
    dt: float                # static: coupling timestep in seconds
```

- `time.year_fraction` is the position in the annual cycle in `[0, 1)` at the
  *start* of the step; it is what a monthly climatology is interpolated with
  (`jem.utils.cycles.evaluate_cyclic_linear`), computed by calling
  `jcm.date.fraction_of_year_elapsed` on `time.time` — the same function the
  atmosphere itself uses, so JEM does not vendor any date arithmetic of its
  own.
- `time.end_of_step()` returns the clock one step later, advancing `step`,
  `time` and `sim_time` together. A model that needs a boundary condition at
  both ends of a step (the slab models measure an anomaly against the
  climatology at the start and add it back at the end) must use it rather
  than advancing `sim_time` by hand, because `year_fraction` is derived from
  `time`.

`coupling_timestep` itself is a `jdt.Timedelta`, which holds whole seconds —
the shortest step expressible today is one second, which is no limit for a
geoscience configuration but does force a non-geoscience component onto a
coarser step than it might otherwise want (jax-esm#110 tracks lifting it to
accept a float number of seconds).

## The scan loop

`Coupler.generate_step_function()` returns the pure function `CoupledCarry ->
(CoupledCarry, dict[str, Diagnostics])` that runs one workflow pass and
returns the carry with `step` (and `time`) advanced. It snapshots the
components and exchangers as they stand when it is called, so registering a
component afterwards cannot silently change an already-compiled step.

`Coupler.generate_trajectory_function(iterations, *, remat=False, jit=True)`
drives that step with `jax.lax.scan` over `iterations` steps and no `xs`
(the steps are identical; the only per-step input, the clock, comes from the
carry). It returns `carry -> (final_carry, diagnostics)`, where every
diagnostics leaf gains a leading axis of length `iterations`. `remat=True`
wraps the step in `jax.checkpoint`, trading recomputation for memory when
differentiating through a long trajectory; `jit=False` leaves the scan
unjitted.

Because the clock lives in the carry, calling the trajectory function again
on the returned carry continues the run:

```python
run = coupler.generate_trajectory_function(30)
carry = coupler.initialize()
for chunk in range(12):
    carry, diagnostics = run(carry)
    datasets = coupler.to_xarray(diagnostics, first_step=chunk * 30)
```

{doc}`running` covers the loop that does this for a real run — chunking,
output, checkpointing and the health gate.
