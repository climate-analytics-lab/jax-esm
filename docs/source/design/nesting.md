# Nesting and multiplicity: running components at different rates

A coupled Earth-system model rarely runs every component on one clock — a
GFDL-style fast atmosphere/land loop inside a daily ocean coupling is the
standard example. JEM gives that pattern two equivalent forms: repeating a
name in one `Coupler`'s `workflow` (multiplicity), or nesting a `Coupler`
inside a slower one (a `Coupler` satisfies `Component`). They produce
bit-identical runs, and which one reads better is a matter of whether the
fast loop is a thing in its own right.

## Workflow nesting and multiplicity

A **workflow** is an ordered tuple of names driving one coupling timestep.
Each entry is either a component name — run that component's `step` on its
carry and record its diagnostics — or an exchanger name — call it on the
whole mapping. Components and exchangers share one namespace, so one name
may not be both; a name may, however, appear in the workflow more than once.
The default is every exchanger (in insertion order) followed by every
component (in insertion order):

```python
coupler.workflow  # ("atm_ocn_exchange", "atm", "ocn")
```

An explicit `workflow=` may be an arbitrarily **nested** sequence of names.
The nesting is notation only — it is flattened at construction, and
`Coupler.workflow` is always the flat tuple actually executed — but it lets
a coupling scheme be written the way it is described:

```python
Coupler(components, exchangers,
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=start_date,
        workflow=[["atm_lnd_exchange", "atm", "lnd"] * 24,
                  "atm_ocn_exchange", "ocn"])
```

Strings are the leaves; any other leaf is a `TypeError`, because the
alternative — iterating it — would silently turn a stray object into a
sequence of characters. An unknown name is still a `ValueError` at
construction.

**Multiplicity.** An element listed *n* times runs *n* times per coupled
step, on a clock *n* times faster. In the example above the atmosphere, the
land and the exchanger between them run hourly inside a daily ocean
coupling, with no second `Coupler` and no component-side sub-stepping code.

- **The sub-timestep is `coupling_timestep / n`, and must be a whole number
  of seconds.** `jdt.Timedelta` is integer-backed, so anything else would
  have to be rounded, and a rounded sub-step would desynchronise the
  sub-stepped component from the coupled clock a little more every step. It
  is refused at construction, with a `ValueError` naming the element and the
  count.
- **A bindable component is bound with its own sub-timestep**, once — the
  step it actually advances by, not the coupled one, so a component that
  sub-cycles an internal timestep (JCM, Veros) sub-cycles the right number
  of times. A component an explicit workflow never names has multiplicity
  0: it is neither bound nor run. A component registered *after*
  construction with `add_component` is bound with the multiplicity the
  current workflow gives it, or the full coupling timestep when nothing
  names it — which is the case for the default workflow, since that is
  derived from the components and the new one is not registered yet.
  Binding still happens before registering, so a component that rejects the
  clock never enters the coupler.
- **Each call gets its own clock.** The loop over the workflow is ordinary
  Python, run once at trace time, so which call this is — *k* of *n* — is a
  static number: call *k* of coupled step *s* is handed
  `Coupler.coupling_time_at_substep(s, time, k, n)`, whose `step` is the
  sub-step `s * n + k` (exact integer arithmetic on the int32 counter),
  whose `dt` is the sub-timestep, whose `sim_time` is `(s * n + k) * dt` and
  whose `time` is the carried `jax_datetime.Datetime` advanced to that
  sub-step. `year_fraction` is `jcm.date.fraction_of_year_elapsed(time)` at
  the sub-rate too, so an hourly sub-step's own `time` is a real instant on
  the real calendar and the seasonal cycle does not quantise away in a long
  run. Exchangers may be repeated as well and see the same clock.
- **`CoupledCarry.step` still counts coupled steps.** The sub-step count is
  derived from it, never stored, so checkpoints, resume and chunked runs are
  untouched: a checkpoint of a run with multiplicity restores the coupled
  counter and both clocks continue.
- **Diagnostics of a repeated component are stacked** along a new leading
  axis of length *n*, in the order the calls were made, so a trajectory
  returns `(steps, n, ...)` for it; `to_xarray` folds those two axes into
  one and labels the records at the sub-rate (see {doc}`output`).

Everything about `n == 1` — the clock a component sees, the shape of its
diagnostics, its time axis, the traced operations — is exactly what it is in
a coupler with no multiplicity at all, so adding a fast loop to one part of a
model cannot perturb the rest of it.

## Nesting couplers

A `Coupler` satisfies `Component`: it has a `name` (the keyword-only
`name="coupled"` argument; the *registered* key in an outer coupler is what
the outer workflow uses), an `initialize()` returning its `CoupledCarry`, and
a `step(carry, time)`. It also implements `SupportsBind` and `SupportsXarray`.
So a coupled model can be a component of a slower coupled model with no
wrapper class:

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
  one, and the start date to be equal; anything else is a `ValueError`, as
  it is for any other component with an internal timestep. It records the
  ratio *r* (24 here). Binding again to the same clock is a no-op, to a
  different one a `ValueError`: one instance belongs to one coupled model.
- **`step`** runs *r* of the inner coupler's own coupled steps, through an
  unjitted trajectory (`lax.scan`, so the inner step appears once in the
  outer jaxpr rather than *r* times unrolled). The inner clock comes from
  the inner carry's own `time`/`step` exactly as in a standalone run, so it
  is continuous across outer steps and survives a checkpoint; the outer
  `time` is only checked against it. Calling `step` before `bind` is a
  `RuntimeError`. For `r == 1` the inner step is run directly and the
  diagnostics gain no extra axis, mirroring multiplicity 1.
- **The carry** of the inner coupler is a `CoupledCarry` living inside the
  outer one's `components`, so there are two step counters: the outer
  counts outer steps, the inner counts its own. `outer.save_carry(carry,
  directory)` writes the inner model into `directory / <its registered
  name>` through the inner coupler's own `save_carry`, and
  `outer.load_carry(directory)` reads it back the same way, so a resume
  continues both clocks — and a component inside the inner model that needs
  its own format (Veros) still gets it.
- **Exchangers in the outer coupler** see the inner `CoupledCarry` under its
  registered name and reach inner components through `.components`.
  `jem.nested_carry(carries, outer_name, inner_name)` and
  `jem.with_nested_carry(carries, outer_name, inner_name, new_inner_carry)`
  are that read and that immutable write (`dataclasses.replace` on the
  inner `CoupledCarry`), written once:

  ```python
  def srf_ocn_exchange(components, time):
      del time
      land = nested_carry(components, "atm_lnd", "lnd")
      ocn = dict(components["ocn"], forcing=land["derived"].total_heat_flux)
      return dict(components, ocn=ocn)
  ```

- **Output.** The inner coupler's `to_xarray` returns one dataset per *its*
  components, and the outer coupler flattens them into its result under
  those names — the nested coupler's own registered name does not appear.
  The inner datasets carry the inner, faster time axis: `Coupler.to_xarray`
  supports both the run form `to_xarray(diagnostics, first_step=0)` and the
  component form `to_xarray(diagnostics, time)`, and in the second it takes
  `time.steps[0] * r` as its own first step and folds the outer coupler's
  leading axis of length *r* into the records first.

## The same model, written flat

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

The two are **equivalent** — the same elements in the same order, on the
same clocks, producing bit-identical carries and datasets
(`tests/unit/test_nested_coupler.py::test_nested_and_flat_forms_are_the_same_run`
is that check). They differ only in bookkeeping:

| | Nested | Flat |
|---|---|---|
| Carry | two levels, two step counters | one level, one counter |
| Exchangers | outer ones go through `nested_carry` | all at one level |
| Fast loop | exists on its own: buildable, testable and runnable alone | is a rate, not an object |

Prefer the **nested** form when the fast loop is a thing in its own right —
an already-assembled surface model, something you also run standalone, or a
piece another model will reuse — and the **flat** form when it is only a
rate: one coupler, one carry and one workflow to read.
