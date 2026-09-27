# Exchanging information between components

## Exchangers

An **exchanger** is the only mechanism for exchanging information between
components:

```python
Exchanger = Callable[[dict[str, Carry], CouplingTime], dict[str, Carry]]
```

It receives the mapping of every component's carry and the clock, and
returns the mapping to continue with. The clock is passed so a
time-dependent coupling (lagged exchange, ramped forcing) needs no state of
its own:

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

1. **Do not mutate in place.** The carries handed to an exchanger are the
   ones the scan is carrying. Build new structs (`.replace(...)` on a
   `tree_math` or `flax.struct` struct, `dataclasses.replace`, a new `dict`)
   and return them. The coupler passes a *fresh* dict, so adding or
   replacing entries cannot reach the caller's carry, but the structs inside
   it are shared, so mutating one of those in place still corrupts the
   caller's state.

   The risk is easy to miss because it only shows up outside `jit`. Inside
   `generate_trajectory_function` the carries are tracers, so an in-place
   assignment cannot reach the caller's arrays. But run one step eagerly
   (`model.generate_step_function()(carry0)` — the natural thing to do when
   debugging, checking a gradient, or comparing two workflows from one
   initial condition) and the struct being assigned into *is*
   `carry0.components["seaice"]["forcing"]`: the initial carry is silently
   overwritten, and the next run from `carry0` starts somewhere else.
   `tests/unit/test_coupler.py` pins this asymmetry
   (`test_in_place_exchange_corrupts_the_initial_carry_eagerly`), so the rule
   is not just advice.
2. **Do not change the pytree structure.** After every workflow element the
   coupler compares the structure of the carries dict with the structure it
   had on entry and raises `RuntimeError` naming the element responsible.
   The check is at trace time, so it costs nothing per step and turns an
   opaque `lax.scan` error into a located one.

An exchange that only *moves* fields — which is most of a coupled model —
does not need to be written as a function at all: see *The declarative
exchange* below.

## The declarative exchange

Almost every exchange in a coupled Earth-system model is the same shape:
field X of component A becomes field Y of component B, optionally regridded
on the way. `jem.exchangers` writes that as a table instead of a function:

```python
from jem import Coupler, default_exchangers

components = {"atm": atm, "ocn": ocn, "seaice": seaice}
coupler = Coupler(components, default_exchangers(components),
                  coupling_timestep=jdt.to_timedelta(1, "day"),
                  start_date=start_date)
```

`ExchangeSpec(src, dst, regrid=None)` is one row, addressing a field as
`"component.section.field"` with `section` one of `state`, `derived`,
`forcing` — the carry layout every packaged component shares. `Exchange(specs,
regridders)` executes a list of rows and *is* an ordinary `Exchanger`, so
nothing in the coupler knows the difference. A component whose carry is
shaped differently can still be coupled, with a hand-written exchanger.

`default_exchanges(components)` is the standard wiring, in one place
(`STANDARD_EXCHANGES`):

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

A row survives only if **both** its components are present, so an
aquaplanet with no land model gets the four rows that do not mention `lnd`,
and an atmosphere/ocean pair gets two. The wiring is by *name* — `("atm",
"ocn", "lnd", "seaice")` — and a component registered under a name that just
misses one of those (`ice`, `ocean`, `land`) is left unconnected with a
warning, because the failure it would otherwise cause is silent: the sea ice
simply never receives anything.

**One table per carry layout.** Component wrappers do not all name the same
physical field the same way, nor keep it in the same section: the Veros
ocean takes its surface heat flux as `forcing.heat_flux` and publishes its
sea surface temperature from `derived` (its `state` is Veros' own
`VerosState` object, not a struct of exchangeable fields), where a slab has
`forcing.total_heat_flux` and `state.sea_surface_temperature`. So a
`VerosComponent` registered as `"ocn"` selects `VEROS_OCEAN_EXCHANGES`
instead — the same wiring in Veros' names, plus the freshwater flux Veros
also takes, and with no `ocn` → `seaice` row because Veros publishes no
freeze/melt potential (that combination is warned about). The choice is made
by *type*, which needs real components: called with a list of names,
`default_exchanges` cannot tell one ocean from another and gives the slab
table. The check looks the wrapper's module up in `sys.modules` rather than
importing it, so a JAX-ESM without the optional Veros dependency never
imports Veros to find out that it has no Veros ocean.

What the Veros table deliberately does **not** carry is the **wind stress**:
Veros integrates `forcing.surface_taux`/`tauy` and the atmosphere publishes a
near-surface *wind*, so getting from one to the other is a bulk drag law
(and, on a rotated grid, a rotation into its local frame) — a computation,
not a copy, and therefore a hand-written exchanger,
`coupling.exchanger: jem.fluxes.VerosExchange`, in the shipped `veros-*`
configurations. The declarative table above carries the rest of the
coupling.

Three properties are worth stating, because a hand-written exchanger has
them only by accident:

- **An exchange is simultaneous, not sequential.** Every source is read from
  the mapping as it arrives, before any destination is written, so
  reordering the table cannot change a run. Two rows writing the same
  destination is a `ValueError` at construction for the same reason.
- **It is checkable before the run.** `Exchange.validate(carries)` — which
  `jem.runners` calls with `coupler.initialize().components` — turns a
  mistyped component, section, field or regridder into an error naming the
  spec, before a model is integrated. The same lookups fail the same way at
  trace time for a caller that skips it, which is still far earlier than a
  wrong number.
- **It never mutates.** New section structs with `.replace(...)`, new
  carries with `dict(carry, ...)`, and a new mapping — the exchanger
  contract's own rule, enforced here once for every table.

**Regridding.** A row that crosses the atmosphere/ocean grid boundary may
name a regridder, and `default_exchanges` names one from a mapping keyed by
*direction and kind*: `a2o`/`o2a` for the direction, `flux`/`state` for the
kind. The kind is written on each row of the table rather than inferred from
the carry section the field is read from, because the two do not agree — the
same intensive sea surface temperature comes from `state` on a slab and from
`derived` on a Veros ocean. Extensive quantities (heat fluxes, the
freeze/melt energy, an areal ice fraction) are mapped conservatively so their
budgets survive the interface, while an intensive state variable such as SST
is interpolated bilinearly, which does not leave a conservative map's
staircase in a smooth field. Rows that stay on one grid never get a
regridder.

```python
default_exchangers(components, regrid={
    "a2o_flux":  ESMFRegridder(a2o_conservative_weights),
    "o2a_flux":  ESMFRegridder(o2a_conservative_weights),
    "o2a_state": ESMFRegridder(o2a_bilinear_weights),
})
```

The maps themselves are `jem.regrid.ESMFRegridders`, an immutable named
collection built from ESMF weight files generated offline by
`ESMF_RegridWeightGen` — JEM applies weights, it does not compute them,
because they depend only on the two grids and never on the run.

### Coupling is lagged

None of this changes *when* fields move, and with the default workflow the
exchange is lagged by one coupling step. With `["exchange", "atm", "ocn"]`:

- `exchange` runs **first**, on the carries as they were left at the end of
  step *n−1*. So at step *n* the ocean is driven by the atmosphere's fluxes
  from step *n−1*, and the atmosphere sees the SST the ocean reached at the
  end of step *n−1*.
- On the **first** step there is no previous step, so each component
  receives whatever its `initialize()` put in its forcing section — zeros,
  for every packaged surface component, and for `JCMComponent` the boundary
  conditions it was built with, taken at the start date. A run therefore
  begins with one step of uncoupled spin-up: the ocean's first step sees no
  heat flux at all. (Under this workflow the atmosphere's own initial
  forcing is overwritten before it ever steps, since `exchange` runs first;
  under `["atm", "exchange", ...]` it is what the atmosphere integrates its
  first step on.)
- The lag is a property of the *workflow*, not of the exchanger. An
  `["atm", "exchange", "ocn"]` workflow hands the ocean the atmosphere's
  fluxes from the same step, at the cost of giving the atmosphere a
  two-step-old SST. Neither order gives every component same-step
  information; that needs an iterated (implicit) exchange or a partitioned
  workflow, which is a follow-up.
