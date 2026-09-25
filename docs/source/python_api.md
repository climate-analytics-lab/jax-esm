# The Python API

Python is JAX-ESM's primary interface. The configuration layer
(`python -m jem.main`, {doc}`getting_started`) is a thin wiring layer over
exactly the objects on this page — `Coupler`, the exchangers, `run_chunked`
and the component classes — and every default lives on the Python class or
function that owns it, never on a YAML file. This page is the complete,
direct construction; {doc}`getting_started` shows the same run composed from
config groups, and {doc}`adding_a_component` shows how to wrap a new model so
it can join one.

## A complete coupled run

A runnable aquaplanet: the JCM atmosphere coupled to JEM's slab ocean. It
takes a couple of minutes on a laptop CPU, mostly XLA compilation. This block
is identical to the README's Quick Start — copy either one — and is executed
by `tests/unit/test_readme_quickstart.py`, so the two cannot drift apart.

```python
import jax_datetime as jdt
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords

from jem import Coupler, default_exchangers, run_chunked
from jem.components import JCMComponent, SlabOceanModel
from jem.components.slab import SlabGrid

start_date = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")

# The JCM atmosphere: a plain jcm.model.Model, wrapped as a component. jax-gcm
# v3's clock is unconditionally proleptic Gregorian, so the coupler below must
# share that calendar.
atm_model = jcm.model.Model(coords=get_speedy_coords(), start_time=start_date)
atm = JCMComponent(atm_model)

# Aquaplanet: the slab grid is built from the atmosphere's own horizontal grid,
# and with no fractional mask every cell is ocean.
grid = SlabGrid.from_coords(atm_model.coords.horizontal)

# An exchanger is the only place where components exchange information.
# `default_exchangers` is the standard wiring written down once — here, the
# atmosphere's surface heat flux drives the ocean and the ocean's SST comes
# back as the atmosphere's boundary condition — filtered to whichever of the
# standard components (`atm`, `ocn`, `lnd`, `seaice`) are present.
components = {"atm": atm, "ocn": SlabOceanModel(grid)}
coupler = Coupler(
    components,
    default_exchangers(components),
    coupling_timestep=coupling_timestep,
    start_date=start_date,
    calendar="gregorian",
)
print(repr(coupler))

# One run loop for every coupled run: integrate a chunk, write one file per
# component, check the atmosphere is still healthy, repeat. Every run default
# lives on `run_chunked` itself. Each file is named after the coupled step its
# chunk starts at, so this writes `atm-00000000.nc` and `atm-00000005.nc`
# (and the ocean's two) into `output/`.
result = run_chunked(
    coupler, total_time="10 days", chunk="5 days", output_dir="output"
)
print(result.steps_completed, "coupled steps;", len(result.paths), "files")
```

## The pieces, in the order they appear

- **The wrapper** `JCMComponent` adapts a stock `jcm.model.Model` without
  touching it — no methods are attached to the model. The coupler calls its
  `bind()` when it is registered, which is where the model's start time and
  timestep are checked against the coupler's, and where the coupler's own
  calendar is checked against jax-gcm's — `"gregorian"` is the only value
  that agrees with jax-gcm's own (unconditional) clock.
- **The grid** comes from the atmosphere's own `coords.horizontal`, so the
  ocean cannot end up on a grid that merely resembles the atmosphere's. Pass
  `fractional_mask=` (e.g. `jcm.terrain.TerrainData.from_file(...).fmask`)
  for a land-sea mask; without one every cell is ocean.
- **The exchanger** is the only place where components exchange anything.
  {func}`~jem.default_exchangers` builds the standard wiring for
  whichever of the standard components (`atm`, `ocn`, `lnd`, `seaice`) are
  present. An exchange it cannot express — one that regrids, computes a flux,
  converts units or blends two fields — is a plain function
  `(dict[str, carry], CouplingTime) -> dict[str, carry]`; {doc}`adding_a_component`
  writes one out. Coupling is **lagged**: with the default workflow the
  exchanger at step *n* moves what each component produced during step *n-1*.
- **The coupler** owns the clock: the coupling timestep, the start date and
  the calendar live here and nowhere else, and every component's `step` is
  handed the same `CouplingTime`.
- **The workflow** — printed by `repr(coupler)` — is the coupling scheme. It
  defaults to every exchanger followed by every component; pass
  `workflow=["atm", "exchange", "ocn"]` to reorder it. It may be nested, and a
  name may appear more than once: an element listed *n* times runs *n* times
  per coupled step, on a clock *n* times faster. So
  `workflow=[["atm_lnd_exchange", "atm", "lnd"] * 24, "atm_ocn_exchange",
  "ocn"]` couples the atmosphere and the land hourly inside a daily ocean
  coupling, and the hourly components write 24 output records per coupled
  step. The same model can be written as an hourly `Coupler` registered as a
  component of the daily one — a `Coupler` satisfies the component contract.
  See {doc}`design/architecture` for both forms.
- **The run loop** {func}`~jem.run_chunked` integrates in chunks: per
  chunk it writes one file per component, checkpoints if it was given a path,
  and runs a health check on the result, stopping the run if the atmosphere
  has gone unstable. Every run default lives on its signature. `total_time`
  and `chunk` must both be whole multiples of the coupling timestep, and
  `total_time` a whole multiple of `chunk`.

## Exchanges a table can express

Most of what a coupled model does is move a field from one component's carry
into another's, and `jem.exchangers` writes that as a table instead of a
function. The same aquaplanet, with a sea-ice component added and the wiring
spelled out explicitly beside the one-line form that builds the same thing:

```python
from jem import Coupler, Exchange, ExchangeSpec
from jem.components import JCMComponent, SlabOceanModel, SlabSeaiceModel

components = {
    "atm": atm,
    "ocn": SlabOceanModel(grid),
    "seaice": SlabSeaiceModel(grid, name="seaice"),
}

# Written out by hand, this is exactly what default_exchangers(components)
# below builds: the standard rows filtered to the components present.
table = Exchange([
    ExchangeSpec("atm.derived.total_heat_flux", "ocn.forcing.total_heat_flux"),
    ExchangeSpec("ocn.derived.ice_frazil_melt_energy",
                 "seaice.forcing.ice_frazil_melt_energy"),
    ExchangeSpec("ocn.state.sea_surface_temperature",
                 "atm.forcing.sea_surface_temperature"),
    ExchangeSpec("seaice.derived.ice_fraction", "atm.forcing.sice_am"),
])

coupler = Coupler(
    components,
    {"exchange": table},                 # or: default_exchangers(components)
    coupling_timestep=coupling_timestep,
    start_date=start_date,
    calendar="gregorian",
)
```

A spec addresses a field as `"component.section.field"`, `section` being one
of `state`, `derived`, `forcing` — the carry layout every packaged component
shares. `Exchange` enforces two rules that a hand-written exchanger only has
by accident: **one destination field has one source** (two rows writing the
same destination raise at construction), and **every source is read before
anything is written**, so an exchange is simultaneous rather than sequential
and reordering the table cannot change a run.

Call `Exchange.validate(carry.components)` — where `carry =
coupler.initialize()` — before building the coupler's trajectory function; it
names a mistyped component, section, field or regridder rather than letting
it fail inside a traced step. And the wiring is by *name*:
`SlabSeaiceModel`'s `name` must be `"seaice"` (its default) for
`default_exchangers` to route the ocean's freeze/melt potential to it — a
sea-ice model registered under any other name is left unconnected, silently
unless the name is one of the near-misses (`ice`, `sea_ice`, `sic`) the
default wiring recognises and warns about.

See {doc}`design/architecture` for the full table (including the Veros
variant), the regridding keys a mixed-grid run uses, and the lag in full.

## Parameters: process and initial condition

A component's `flax.struct` parameters travel in the carry, so `jax.grad`
reaches them with no special casing — but *how* to vary one depends on when
the component reads it:

```python
ocn = SlabOceanModel(grid)

# A PROCESS parameter is read by step() out of the carry, every step, so it
# is varied by replacing that leaf:
carry = ocn.initialize()
carry["params"] = carry["params"].replace(mixed_layer_depth_max=depth)

# An INITIAL-CONDITION parameter is read once, by initialize(), and never
# again -- replacing it in an existing carry does nothing, so it is given to
# initialize instead:
carry = ocn.initialize(ocn.params.replace(initial_sst=sst0))

# In a coupled model, through the coupler, which routes by component name:
coupled_carry = coupler.initialize({"ocn": ocn.params.replace(initial_sst=sst0)})
```

Each `*Parameters` docstring says which of its fields are initial conditions.
See {doc}`design/architecture`'s *Parameters* section for the pattern in
full, including why the distinction is not one the framework enforces.

## Long runs, checkpoints and reductions

`run_chunked` checkpoints by default and resumes from the same call; a
reduction that must not cost memory proportional to the run length —
`jem.accumulate.monthly_mean`, `windowed_mean` — runs *inside* the scan
instead of writing every step to disk, and is differentiable like everything
else in the carry. Both are long enough that they are not duplicated here:
see the README's *Long runs* section for the worked examples and
{doc}`design/architecture` for the checkpoint format and the accumulator's
binning rules.

## The same run from the command line

```bash
python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run
```

See {doc}`getting_started` for the override spellings, the `@atmosphere`
group re-rooting and the exit status.
