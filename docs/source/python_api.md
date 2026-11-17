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

# The JCM atmosphere: a plain jcm.model.Model, wrapped as a component.
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
  `bind()` when it is registered, which is where the model's start date and
  timestep are checked against the coupler's.
- **The atmosphere's own timestep.** `Model(...)` above passes no
  `time_step`, so it picks the active physics' *stable* step — 30 minutes for
  SPEEDY at T31L8 — rather than any particular jax-gcm configuration's own
  choice. `python -m jem.main`'s atmosphere instead reads `atmosphere.run.time_step`
  from its composed config (12 minutes by jax-gcm's own `run/default.yaml`
  default), so a script that means to match a specific `+configuration=<name>`
  bit for bit — not just build *an* aquaplanet — should pass that recipe's own
  `time_step` explicitly, in minutes, rather than rely on this page's
  no-configuration default; see `jem.configurations` below for loading the
  recipe itself instead of reproducing it by hand.
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
- **The coupler** owns the clock: a carried `jax_datetime.Datetime`, advanced
  by the coupling timestep every step, lives here and nowhere else, and every
  component's `step` is handed the same `CouplingTime` built from it.
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
  See {doc}`design/nesting` for both forms.
- **The run loop** {func}`~jem.run_chunked` integrates in chunks: per
  chunk it writes one file per component, checkpoints if it was given a path,
  and runs a health check on the result, stopping the run if the atmosphere
  has gone unstable. Every run default lives on its signature. How far to
  integrate is exactly one of `total_time` (a fixed duration) and `end_time`
  (an ISO date). The run length and `chunk` must be whole multiples of the
  coupling timestep; a run length that is not a whole number of chunks ends
  with a shorter chunk, and a run that starts off the chunk grid (extending a
  run that ended on a short chunk, or resuming under a different `chunk`) opens
  with a short one that lands it back on it, so its checkpoints stay on
  `checkpoint_interval`.

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
    ExchangeSpec("atm.derived.sea_total_heat_flux", "ocn.forcing.total_heat_flux"),
    ExchangeSpec("ocn.state.sea_surface_temperature",
                 "seaice.forcing.sea_surface_temperature"),
    ExchangeSpec("ocn.state.sea_surface_temperature",
                 "atm.forcing.sea_surface_temperature"),
    ExchangeSpec("seaice.derived.ice_fraction", "atm.forcing.sice_am"),
])

coupler = Coupler(
    components,
    {"exchange": table},                 # or: default_exchangers(components)
    coupling_timestep=coupling_timestep,
    start_date=start_date,
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
`default_exchangers` to route the ocean's sea surface temperature to it — a
sea-ice model registered under any other name is left unconnected, silently
unless the name is one of the near-misses (`ice`, `sea_ice`, `sic`) the
default wiring recognises and warns about.

See {doc}`design/exchange` for the full table (including the Veros
variant), the regridding keys a mixed-grid run uses, and the lag in full.

**Sea ice and the freezing point.** `SlabSeaiceModel` diagnoses its ice
fraction from the ocean's sea surface temperature, ramping from open water at
`SlabSeaiceParameters.ice_free_sst_excess` (1.8 K) above freezing to full
cover at freezing; it has no energy budget. So the slab ocean beside it
keeps its own deficit when it cools below freezing
(`SlabOceanParameters.clamp_at_freezing=False`, the default) rather than
being clamped at freezing with nothing to account for the heat the clamp
would add. `clamp_at_freezing=True` is for a sea ice that turns the ocean's
`derived.ice_frazil_melt_energy` into ice, and `default_exchanges` refuses it
when no exchange takes that field. `WintonSeaiceModel` is that ice: a
thermodynamic snow-and-two-layer ice with an energy budget, which
`default_exchanges` forces from the atmosphere and routes the ocean's heat
flux through, and which it refuses over an ocean that does not clamp
({doc}`design/winton_seaice`).

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
See {doc}`design/carry_and_clock`'s *Parameters* section for the pattern in
full, including why the distinction is not one the framework enforces.

**Differentiating a whole coupled run with respect to a parameter.** The
carry is built from the parameter, so `jax.grad` of a function of the final
carry reaches it through every coupled step. Differentiate with respect to the
parameter (a float), not the carry: the carry holds integer leaves (the step
counter) that `jax.grad` refuses. The coupler and its trajectory function are
built once, outside the differentiated function (`components` and `coupler`
as built above):

```python
import jax

ocn = components["ocn"]                       # the SlabOceanModel built above
trajectory = coupler.generate_trajectory_function(2, remat=True)

def mean_final_sst(mixed_layer_depth_max):
    params = ocn.params.replace(mixed_layer_depth_max=mixed_layer_depth_max)
    final, _ = trajectory(coupler.initialize({"ocn": params}))
    return final.components["ocn"]["state"].sea_surface_temperature.mean()

sensitivity = jax.grad(mean_final_sst)(60.0)  # K per metre of mixed-layer depth
```

`jax.jvp` of a function of the initial sea surface temperature gives the
tangent-linear response to an initial-condition perturbation instead; the
notebook `examples/01_basic/03_aquaplanet_response_to_SST_perturbation_using_gradient.ipynb`
does that for the whole coupled aquaplanet.

## Ocean, land and sea-ice variants

The slab components beyond the first block's ocean are built the same way,
each from the model's `SlabGrid` and an optional parameters object. A realistic
land-sea mask is what makes a land or a sea ice worth coupling, so this block
reads jax-gcm's packaged T30 terrain and climatology (the files
`+configuration=earth-slab` names); `run_two_steps` is the usual
`default_exchangers` coupling, run for two steps.

```python
import jax_datetime as jdt
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

from jem import Coupler, default_exchangers, exchanged_fields
from jem.components import (
    BucketLandModel, JCMComponent, SlabOceanModel, WintonSeaiceModel,
)
from jem.components.slab import SlabGrid, SlabOceanParameters, WintonSeaiceParameters
from jem.config import package_data_path

start_date = jdt.to_datetime("2000-01-01")
coords = get_speedy_coords()
climatology = package_data_path("jcm.data", "bc/t30/clim/forcing.nc")

terrain = TerrainData.from_file(package_data_path("jcm.data", "bc/t30/clim/terrain.nc"), coords)
atm_model = jcm.model.Model(coords=coords, terrain=terrain, start_time=start_date)
grid = SlabGrid.from_coords(atm_model.coords.horizontal, fractional_mask=terrain.fmask)


def run_two_steps(components):
    exchangers = default_exchangers(components)
    # The fields an exchanger writes into the atmosphere's forcing have to be
    # declared to the atmosphere, so that its carry keeps one structure.
    components["atm"].set_exchanged_forcing(exchanged_fields(exchangers, "atm"))
    coupler = Coupler(components, exchangers,
                      coupling_timestep=jdt.to_timedelta(1, "day"), start_date=start_date)
    return coupler.generate_trajectory_function(2)(coupler.initialize())


# The bucket land evolves the soil water SPEEDY's own land reads. It starts half
# full unless `soil_water_clim_file` names a monthly `soilw_am`.
land = BucketLandModel(grid, soil_water_clim_file=climatology)
carry, _ = run_two_steps({"atm": JCMComponent(atm_model), "ocn": SlabOceanModel(grid), "lnd": land})

# The Winton sea ice has an energy budget, so the ocean beside it has to clamp at
# freezing and hand it the deficit. `ice_clim_file` is the initial cover only;
# `transport=True` also advects and diffuses the ice.
ocean = SlabOceanModel(grid, SlabOceanParameters(clamp_at_freezing=True))
ice = WintonSeaiceModel(
    grid, WintonSeaiceParameters(initial_ice_thickness=1.5), ice_clim_file=climatology,
)
carry, _ = run_two_steps({"atm": JCMComponent(atm_model), "ocn": ocean, "seaice": ice})
```

What a user has to know about each, beyond the constructor:

- **`BucketLandModel(grid, params=None, *, name="lnd", soil_water_clim_file=None)`**
  needs a grid with cell areas, takes the atmosphere's precipitation and its
  *land-tile* evaporation, and hands back `soilw_am`. SPEEDY keeps integrating
  the land's temperature itself. It is refused under an atmosphere with no land
  tile (ECHAM). CLI: `land=bucket +land.soil_water_clim_file=...`;
  {doc}`design/bucket_land` has the budget.
- **`WintonSeaiceModel(grid, params=None, *, name="seaice", transport=False,
  frozen_precipitation_latent_heat_release_mode=True, ice_clim_file=None)`**.
  `default_exchangers` forces it from the atmosphere's sea tile and routes the
  ocean's heat flux through it, and refuses three pairings: an ocean that does
  not clamp at freezing (`SlabOceanParameters(clamp_at_freezing=True)`, else
  the freezing would be counted twice), a Veros ocean, and an atmosphere that
  publishes no sea tile (ECHAM). Leave
  `frozen_precipitation_latent_heat_release_mode` at `True` for SPEEDY, whose
  snowfall arrives without its latent heat released; pass `False` for an
  atmosphere that already released it. CLI: `seaice=winton
  +ocean.params.clamp_at_freezing=true`, or `+configuration=earth-slab-winton`;
  {doc}`design/winton_seaice` has the coupling.
- **`SlabOceanModel(grid, params=None, *, name="ocn", sst_clim_file=None,
  q_flux_file=None)`** forces its mixed layer in one of three ways,
  `SlabOceanParameters.forcing_method`: `"none"` (the default, the coupled flux
  alone), `"relaxation"` toward the `sst` of `sst_clim_file` (required;
  CLI `ocean=slab_relax ocean.sst_clim_file=...`), or `"qflux"`, a prescribed
  heat flux from the `qflux` of `q_flux_file` (CLI `ocean=slab_qflux
  ocean.q_flux_file=...`, where the file is required but may be given as
  `null`, which applies a zero Q-flux, the starting point when the Q-flux is
  itself being trained).

```python
ocean = SlabOceanModel(grid, SlabOceanParameters(forcing_method="qflux"), q_flux_file=None)
```

## A Veros ocean

`VerosComponent` wraps an already-built Veros model, a live Python object no
configuration can name; `VerosComponent.from_setup` is the door that takes an
importable setup instead. It needs the optional jittable Veros fork
({doc}`getting_started`), and importing Veros makes the whole process double
precision, so this block is not executed here; its signatures are those of
`jem.components.VerosComponent.from_setup`, `jem.fluxes.VerosExchange` and
`double_drake_setup`:

```python
from jem.components import VerosComponent
from jem.fluxes import VerosExchange

ocean = VerosComponent.from_setup(
    "jem.components.veros.setups.double_drake.double_drake_setup",
    land_sea_mask_file=package_data_path("jem.data", "terrain_double_drake_T31.nc"),
)
# `default_exchangers` supplies the table in Veros' field names, but not the wind
# stress, which crosses through `VerosExchange` (rotated into the ocean's frame
# when the grid is rotated, `rotation_grid_file=`; regridded when `regrid=` is given).
coupler = Coupler(
    {"atm": atm, "ocn": ocean}, {"exchange": VerosExchange()},
    coupling_timestep=coupling_timestep, start_date=start_date,
)
atm.set_exchanged_forcing(["sea_surface_temperature"])
```

`atm_model` has to be built on the terrain file the ocean shares
(`atmosphere.terrain.file` in the configuration) and with a 30-minute
`time_step`, as `+configuration=veros-double-drake` does; the configuration
is the validated way to get all of that right, and its comments say why each
setting is what it is. There is no sea ice to couple to a Veros ocean yet
({doc}`issues`).

## Surface constants

`jem.constants` holds the surface-slab properties `jcm.constants` does not
define (seawater's density, heat capacity and freezing point, and the idealized
slab atmosphere's bulk coefficients); everything the atmosphere shares comes from
`jcm.constants`, so the two cannot disagree. Override either before building
any component, since a component reads the constants while building its
carry and while tracing its step:

```python
import jem.constants

jem.constants.set_constants(ocean_density=1030.0)   # one field of the live set
```

From the command line `+atmosphere.constants.grav=9.7` overrides a
`jcm.constants` value for every component.

## The atmosphere's initial condition

`JCMComponent` starts the atmosphere from jax-gcm's default, an isothermal
atmosphere at rest, unless it is given an `initial_state` -- anything
`jcm.model.Model.bootstrap_state` takes: a gridpoint `PhysicsState` such as an
ERA5 analysis, or a dycore-native warm-start state (with its physics carry as
`initial_physics_state`):

```python
from jcm.initial_states import era5_state, jw_state

atm = JCMComponent(atm_model, initial_state=era5_state(atm_model.coords, "2022-12-24"))
atm = JCMComponent(atm_model, initial_state=jw_state(atm_model, rh=0.6))
```

From the command line this is jax-gcm's own `init` group, re-rooted:
`init@atmosphere.init=era5` (see {doc}`getting_started`).

## Gradients of long, high-resolution runs

`jax.grad` of `generate_trajectory_function(n, remat=True)` keeps one coupled
carry per step on the device. When that does not fit -- a T255 atmosphere
coupled hourly to a one-degree ocean carries 2.6 GB, and a twelve-day
sensitivity has 288 steps -- {func}`jem.adjoint.checkpointed_value_and_grad`
computes the same gradient with the block-start carries held in host memory:

```python
from jem.adjoint import checkpointed_value_and_grad

weights = np.zeros(n_steps)
weights[-120:] = 1 / 120                       # J = mean over the last 120 steps

def box_precipitation(new_carry, diagnostics):
    return jnp.sum(box_weight * new_carry.components["atm"]["derived"].precipitation)

J, gradient = checkpointed_value_and_grad(
    coupler.generate_step_function(), box_precipitation, coupler.initialize(), weights,
    block_size=6,                               # carries held on the device at once
    on_block_gradient=lambda k, g: ...,         # dJ/d(carry at step k), every block
)
gradient.components["ocn"]["state"]             # dJ/d(initial ocean state)
```

At high resolution the atmosphere's time-varying forcing is the other big
leaf of the carry (jax-gcm's daily T255 climatology is 4.3 GB);
`jem.components.jcm.forcing_window.restrict_forcing_to_window(forcing, start,
end)` keeps only the records a run reads, without changing any value it
sees. {doc}`design/differentiating_long_runs` explains the choices, with a
worked T255 case, including what the gradient of a chaotic atmosphere a week
ahead does and does not tell you.

## Long runs, checkpoints and reductions

`run_chunked` checkpoints by default and resumes from the same call; a
reduction that must not cost memory proportional to the run length —
`jem.accumulate.monthly_mean`, `windowed_mean` — runs *inside* the scan
instead of writing every step to disk, and is differentiable like everything
else in the carry. Both are long enough that they are not duplicated here:
see {doc}`design/running` for the worked examples, the checkpoint format and
the accumulator's binning rules.

A spun-up state can be the initial condition of a run with a **different start
date**: build the coupler for the new run, put the carry's clock at *its*
start date with `Coupler.restart_clock`, and pass the result as
`initial_carry`. `run_chunked` refuses a carry whose clock is not its
coupler's, and the message names this method:

```python
carry = july_coupler.restart_clock(spun_up_carry)     # or load_carry(..., check_clock=False)
run_chunked(july_coupler, total_time="30 days", initial_carry=carry)
```

The seasonal cycle, output labels and file names then count from the new start
date. See {doc}`design/carry_and_clock` for what each component resets.

## Validated configurations from Python

The construction above builds a coupled model by hand -- the thing to do
when writing a new experiment. To instead run one of the *validated*
configurations in `jem/config/configuration/*.yaml` -- `aquaplanet-slab`,
`aquaplanet-slab-mixed-grid`, `earth-slab`, `earth-slab-winton`,
`veros-double-drake` and `veros-earth`, the same six
`configurations.available()` returns, each with its command in
{doc}`examples` -- without composing Hydra,
`jem.configurations` is the recipe door onto them (issue #131):

```python
from jem import configurations, run_chunked

configurations.available()          # {name: one-line summary}, read off the yaml

exp = configurations.load("aquaplanet-slab")
exp.coupler                         # the built jem.base.coupler.Coupler
exp.config["ocean"]                 # a plain dict -- what the recipe resolved to
run_chunked(exp.coupler, **exp.run_kwargs)   # reproduces the CLI's build and run settings
```

`load` composes the named yaml through Hydra internally, builds it through
the SAME `jem.runners` builders `python -m jem.main` uses
(`jem.runners.build_coupler`, `jem.runners.build_run_kwargs`), and hands back
a Hydra-free `LoadedConfiguration` -- `omegaconf`/`hydra` never appear in a
caller's own code. `**overrides` is the escape hatch onto both a dotted
value (`load("earth-slab", **{"coupled_run.total_time": "60 days"})`; the run
length need not be a multiple of `chunk`, a last short chunk finishes it) and a
config-group selection (`load("aquaplanet-slab",
seaice="none")`, the Python spelling of the CLI's `seaice=none`). A string
override is carried literally except for a `${...}` interpolation, which
resolves just as it does on the CLI, so
`**{"ocean.sst_clim_file": "${jcm_data:bc/t30/clim/forcing.nc}"}` names the
packaged file (`jem.config.package_data_path("jcm.data", ...)` is the
interpolation-free alternative). `\${` escapes a literal `${` only in a
value read straight off the config, such as `coupled_run.*`: Hydra
re-resolves a component's node when it instantiates it, so a component
field cannot hold a literal `${` through config, on the CLI or here. `load`'s own
docstring lists the small, specific things `run_chunked(exp.coupler,
**exp.run_kwargs)` does NOT reproduce of the CLI (a fresh `output_dir` of the
door's own rather than the CLI's Hydra-managed one, no logger-level change)
and the process-global persistence of a `+atmosphere.constants.*` override.
The docstring is rendered in the API reference ({doc}`api_superset`); in short,
the door builds the same coupler and the same run settings as the command
line, and differs only in where the run writes by default and in leaving the
logger alone.

A notebook that *runs* a validated configuration to demonstrate something
else -- perturbing an initial condition, differentiating through a
trajectory -- loads it through this door rather than copying its settings
into Python by hand, which is exactly the drift a parallel Python
description of a YAML recipe invites (`examples/01_basic/02_...ipynb` and
`03_...ipynb`, `examples/02_experimental/01_earth.ipynb`). A notebook that
*teaches* how a coupled model is put together, like the construction at the
top of this page, still builds directly.

## The same run from the command line

```bash
python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run
```

See {doc}`getting_started` for the override spellings, the `@atmosphere`
group re-rooting and the exit status.
