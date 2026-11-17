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
        total_heat_flux=atm["derived"].sea_total_heat_flux))
    atm = dict(atm, forcing=atm["forcing"].replace(
        sea_surface_temperature=ocn["state"].sea_surface_temperature))
    return dict(components, atm=atm, ocn=ocn)
```

The ocean takes `sea_total_heat_flux`, the flux over the sea part of the
cell, rather than `total_heat_flux`, the grid-cell mean (see *Each surface
takes the flux over its own part of the cell* below).

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

`ExchangeSpec(src, dst, regrid=None, weight=None)` is one row, addressing a field as
`"component.section.field"` with `section` one of `state`, `derived`,
`forcing` — the carry layout every packaged component shares. `Exchange(specs,
regridders)` executes a list of rows and *is* an ordinary `Exchanger`, so
nothing in the coupler knows the difference. A component whose carry is
shaped differently can still be coupled, with a hand-written exchanger.

`default_exchanges(components)` is the standard wiring, in one place
(`STANDARD_EXCHANGES`):

| source | destination |
| --- | --- |
| `atm.derived.sea_total_heat_flux` | `ocn.forcing.total_heat_flux` |
| `ocn.state.sea_surface_temperature` | `seaice.forcing.sea_surface_temperature` |
| `ocn.state.sea_surface_temperature` | `atm.forcing.sea_surface_temperature` |
| `seaice.derived.ice_fraction` | `atm.forcing.sice_am` |
| `atm.derived.precipitation` | `lnd.forcing.precipitation` |
| `atm.derived.land_evaporation` | `lnd.forcing.evaporation` |
| `lnd.derived.soil_water_availability` | `atm.forcing.soilw_am` |

A row survives only if **both** its components are present, so an
atmosphere/ocean pair gets two. A Winton sea ice ({doc}`winton_seaice`)
replaces the first row with `WINTON_SEAICE_EXCHANGES`: it takes the
atmosphere's state and its flux over the sea part, passes the ocean what it
does not keep (`seaice.derived.ocean_heat_flux_up`), and hands the atmosphere
its fraction and skin temperature. The land rows are the bucket's soil water
({doc}`bucket_land`): the land tile's evaporation, never the cell mean, so
`default_exchanges` refuses them under an atmosphere with no land tile. The wiring is by *name* — `("atm", "ocn",
"lnd", "seaice")` — and a component registered under a name that just
misses one of those (`ice`, `ocean`, `land`) is left unconnected with a
warning, because the failure it would otherwise cause is silent: the sea ice
simply never receives anything.

**Each surface takes the flux over its own part of the cell.** A slab ocean
integrates every cell with any sea in it (`SlabGrid.ocean_mask`,
`land_fraction < 1`), so on a coastal cell it shares the cell with land. It
is forced by the atmosphere's `sea_*` fields — the fluxes over the sea part
of the cell — not by the grid-cell mean, which also contains the land part's
flux. The adapter reads them from the tiles jax-gcm's `SurfaceExchange`
publishes, by tile name: SPEEDY's `sea` tile, or the fraction-weighted mean
of a `water` and a `sea_ice` tile
(`jem.components.jcm.exchange_fields.sea_tile`). The land part stays with
the atmosphere: SPEEDY integrates its own slab land (`SpeedySurfaceFlux`'s
default `land_model="slab"`, jax-gcm's `step_slab_land` after
`land_model.f90`), forced by its own land tile, so no row carries the land's
temperature; a land component (`land=bucket`) evolves only the soil water
SPEEDY prescribes.
**SPEEDY.** Its grid mean is `(1 - land_fraction) * sea + land_fraction *
land`, so on one grid the ocean's share and the land's are disjoint and
together are the cell's flux: the ocean takes the sea part only and the land
part stays with SPEEDY's slab land. What the land does with its share is
SPEEDY's: it holds its anomaly at zero on cells with less than `flandmin` of
land, and its relaxation to the climatology is itself a source and a sink.

**ECHAM.** It publishes no flux tiles (its delivered fluxes are untiled), so
`sea_tile` finds none and the `sea_*` fields are equal to the grid-cell mean.
An ocean on a coastal cell therefore also receives the land part's flux, and
the double count on coastal cells is not removed for ECHAM. The adapter says
so with a warning at `initialize`
(`JCMComponent.publishes_sea_tiles` is `False`), and the couplings that need a
tile refuse ECHAM outright: the bucket land (`publishes_land_evaporation`) and
the Winton sea ice (`publishes_sea_tiles`) raise in `default_exchanges`.

**On another grid the sea fluxes are weighted by the sea part.** A plain
conservative map averages every atmosphere cell an ocean cell overlaps by
area alone, so a coastal ocean cell would take the "sea" flux of an all-land
atmosphere cell, which is computed against whatever surface temperature was
written there, with no sea under it. A row's optional `weight` names the
fraction of each source cell its field is over, and the field is then
regridded as `R(weight * value) / R(weight)` (`regrid_weighted`, the
fraction-normalised mapping CESM's coupler applies to the fluxes it computes
over the ocean fraction of a cell): each atmosphere cell counts by the area of
its sea part, and an ocean cell that overlaps no sea at all takes
`R(value)`, the only estimate there is. `default_exchanges` gives every
regridded row reading an `atm.derived.sea_*` field the weight
`atm.derived.sea_fraction` (`1 - fmask` of the atmosphere's terrain), and
`VerosExchange` weights its stress and fluxes the same way. On one grid a row
is a copy and takes no weight: each ocean cell is its own atmosphere cell.
The rows back onto the atmosphere are weighted by the sea part of the
ocean-grid cells (`<component>.derived.sea_fraction`: `1 - fractional_mask`
for a slab ocean or sea ice, the wet mask for Veros), so an atmosphere cell
takes the SST of the sea under it rather than a land cell's fill value, and
`sice_am` stays the ice fraction of the sea part, which is how SPEEDY reads
it. A YAML `coupling.exchangers` table is taken as written: give its rows a
`weight` to get the same.

**The two masks are one coastline, by construction.** Veros decides its wet
and dry columns from its own land-sea mask (land where the ERA5 land fraction
is at least 0.5, `jem.components.veros.setups.earth`), and a wet column is
wholly water, while the atmosphere's sea part is `1 - fmask` of its terrain.
Two independent files disagree: on the packaged `terrain_JCM_T31.nc` against
the rotated Gaussian ocean grid, 0.375 sr of wet columns lay under what the
atmosphere counted as land and took a sea flux it never exchanged (723 of 3018
wet columns had a mapped sea fraction below 0.99), and 0.359 sr of the
atmosphere's sea lay over dry columns (681 of them), against 8.98 sr of ocean:
about 4 % each way. Worse, 110 of the 3404 sea-containing atmosphere cells
(sea fraction 10-57 %, 15 of them tropical) had no wet column under the SST
map at all, so `regrid_weighted` took the unweighted map of the ocean's SST,
which over land holds Veros' fill value, and gave them 273.15 K for the whole
run.

`veros-earth` therefore makes the ocean's mask authoritative, the way CESM
derives its surface fractions (`lfrac = 1 - ofrac`):
`jem.tools.veros_earth_terrain.veros_earth_terrain` reads the wet mask the way
the ocean does (same file, same threshold), maps it onto the T31 grid with the
packaged *conservative* ocean-to-atmosphere weights (the ones `regrid=esmf`
names `o2a_conserve`), and writes `lsm = 1 - ofrac` with the original file's
orography. Each atmosphere cell's sea area is then exactly the wet Veros area
under it (tested to round-off), and the file is packaged as
`terrain_veros_earth_T31.nc`. Three consequences follow, and `VerosExchange`
acts on them:

- A cell with sea in it always overlaps a wet column, so the conservative map
  covers it. The SST map is still bilinear (so it leaves no staircase), and a
  bilinear stencil can miss a sliver of wet ocean (10 cells of 3324 on this
  grid pair, none tropical): where the bilinear map reaches no wet column the
  conservative, wet-weighted map takes over, so no sea cell is ever handed the
  fill value (110 before, none after; `regrid_weighted` states what a cell
  with no weight takes).
- `VerosExchange.check_masks` raises if a cell has sea but no wet ocean
  mapped under it, on the first coupled step (a host callback under a
  `lax.cond` on `step == 0`, since the masks live in the carries and are
  tracers inside the step), naming the count and the remedy: run the
  atmosphere on a terrain file derived from the ocean's mask. A single-grid
  exchange, with identity maps, has one shared mask and is exact:
  `veros-double-drake` is unaffected.
- jax-gcm reads a terrain file **by coordinate** (it interpolates the file
  onto the model's Gaussian latitudes even when the file is on that grid), so
  the new file's `lat` labels are JCM's own latitudes. The original
  `terrain_JCM_T31.nc` labels its first and last rows -/+87.66 where JCM's are
  -/+87.159 (a 0.5 degree error, so jax-gcm mixes 12 % of the neighbouring row
  into each polar row: the orography of the southernmost row changes by up to
  297 m, the northernmost by 56 m) and its other rows are off by up to 0.005
  degrees; its values are on the T31 grid by index. Read through jax-gcm, the
  new file's orography is unchanged to float32 round-off.

What the agreement does not remove:

- **jax-gcm rounds the land fraction.** `jcm.terrain.get_terrain` sets an
  `lsm` within 0.1 of 0 or 1 to exactly 0 or 1. The atmosphere's sea area is
  therefore the wet area to 0.1 per cell, not to round-off: 0.014 sr of
  atmosphere sea is dropped (cells with under 10 % sea) and 0.015 sr added
  (cells with over 90 %), against 8.98 sr of ocean, and they nearly cancel
  (the sea areas differ by 1.6e-4 of the ocean's).
- **The flux map conserves to about half a percent, not exactly.** The
  fluxes are normalised by the sea fraction mapped onto each ocean column
  (`R(s F) / R(s)`), which keeps a coastal column from being diluted by the
  land beside it but cannot match the heat the atmosphere gave up column by
  column: the flux handed to the dry columns beside the coast (5 % of the
  total on day 1 of a `veros-earth` run) is nearly, not exactly, what the
  normalisation adds back. The sea-tile heat the atmosphere gives up and the
  heat the ocean receives differed over a 30-day run by -1.13 W m-2 of ocean
  on average before (always the same sign, up to 1.8) and -0.16 W m-2 after
  (alternating, at most 0.38; 0.27 in mean absolute value), against a mean
  flux of 60-90 W m-2. An exactly conservative alternative exists now that the
  masks agree: hand a wet column `wet * R(s F / ofrac)`, where `ofrac` is the
  conservative map of the wet mask, which closes the heat budget to 2e-16 on a
  day's real fluxes but dilutes columns beside the cells JCM rounds to land;
  it is not what the exchange does.
- **The atmosphere's land data describe the old coastline.** `veros-earth`
  takes its albedo, snow, soil water and land temperature from jax-gcm's
  `bc/t30/clim/forcing.nc`, defined for the original coast. 417 cells gain land
  fraction (0.205 sr of new land, against 3.6 sr of land, so 5.7 %; a mean of
  0.20 of a cell; 8 cells by more than 0.5), and 410 lose it (0.222 sr). The
  new land takes the file's mixed coastal values, an albedo of 0.13 where its
  nearest land neighbours have 0.21 and a soil water availability of 0.52
  where they have 0.73: about 0.001 of global-mean albedo, or 0.5 W m-2 of
  absorbed shortwave if all of it were wrong, for the whole new coast. Sea
  ice, which the atmosphere reads from the same file, is 0.38 in the cells
  that turn to sea poleward of 55 degrees where the nearest sea has 0.48. The
  orography of a cell that turns from mostly land to mostly sea is kept (the
  worst is 1341 m, Andes coast at 31.5 S, land fraction 0.58 to 0.21), so a
  small area of sea sits under a raised surface. A matching forcing file, with
  land fields extended into the new land from their nearest land neighbours
  and sea ice into the new sea, would remove the first and the third; it is
  not part of this change.

**The freezing point is an energy-budget decision.** The slab sea ice
diagnoses its fraction from the SST and has no energy budget, so it takes no
freeze/melt potential, and the standard table has no row for
`ocn.derived.ice_frazil_melt_energy`. The ocean beside it therefore keeps
its deficit below freezing (`clamp_at_freezing=False`): clamping it there
would create the heat it removed the deficit with. `default_exchanges`
raises for an ocean that clamps (`clamp_at_freezing=True`) when no row takes
that deficit, since nothing else in a run would say the energy is not
conserved.

**One table per carry layout.** Component wrappers do not all name the same
physical field the same way, nor keep it in the same section: the Veros
ocean takes its surface heat flux as `forcing.heat_flux` and publishes its
sea surface temperature from `derived` (its `state` is Veros' own
`VerosState` object, not a struct of exchangeable fields), where a slab has
`forcing.total_heat_flux` and `state.sea_surface_temperature`. So a
`VerosComponent` registered as `"ocn"` selects `VEROS_OCEAN_EXCHANGES`
instead — the same wiring in Veros' names, plus the freshwater flux Veros
also takes; the `ocn` → `seaice` row reads the sea surface temperature from
`derived` like the atmosphere's does. The choice is made
by *type*, which needs real components: called with a list of names,
`default_exchanges` cannot tell one ocean from another and gives the slab
table. The check looks the wrapper's module up in `sys.modules` rather than
importing it, so a JAX-ESM without the optional Veros dependency never
imports Veros to find out that it has no Veros ocean.

What the Veros table deliberately does **not** carry is the **wind stress**.
The atmosphere publishes the stress its own surface closure delivered
(`derived.eastward_wind_stress`/`northward_wind_stress`) and Veros integrates
`forcing.surface_taux`/`tauy`, but a stress is a vector in true east/north
components, and a rotated ocean grid needs it in its own local frame. That
rotation mixes the two components, which a row moving one field at a time
cannot do, and two plain copy rows would be silently wrong on a rotated grid.
It therefore goes through a hand-written exchanger,
`coupling.exchanger: jem.fluxes.VerosExchange`, in the shipped `veros-*`
configurations — which also applies the sea-ice mask, a condition rather
than a copy. The declarative table above carries the rest of the coupling.

The ocean takes the atmosphere's stress rather than applying a drag law of
its own to the near-surface wind (jax-esm#132). The published stress is the
one the atmosphere column received, stability-corrected and averaged over the
coupling interval, so momentum is conserved across the interface up to the
conservative regrid. A JEM-side bulk law would be a second stress law, blind
to stability, applied to a wind whose reference height differs between
physics packages (SPEEDY's lowest model level, ECHAM's 10 m), so the ocean's
forcing would change with the atmosphere package for a reason unrelated to
the ocean. It would also be applied to the interval-mean wind, and the drag
of a mean wind is smaller than the mean drag. Measured on the
`veros-double-drake` configuration after 30 days, the published stress over
open ocean is about 1.6 times that of a constant-drag law with `Cd = 1e-3`
(RMS), with the same sign everywhere and a pattern correlation of 0.92.

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
`derived` on a Veros ocean. Extensive quantities (heat fluxes, an areal ice
fraction) are mapped conservatively so their
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
because they depend only on the two grids and never on the run. The grids
themselves (and the weight files built from them, shipped under `jem/data/`)
are generated by [EarthSystemGrids.py](https://github.com/meteorologytoday/EarthSystemGrids.py),
a separate offline tool this project does not depend on at runtime.

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
  workflow, and no packaged workflow does either.
