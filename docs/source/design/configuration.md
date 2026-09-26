# Configuration

Python is the primary interface. The configuration layer is a thin wiring
layer over it, and `python -m jem.main` (or the `jem` console script) is one
command for a coupled run:

```bash
python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run
```

**Exit status.** `0` if the run reached `total_time`, `1` if the health gate
stopped it early — `jem.main` raises `SystemExit(1)` after logging the last
report. A run stopped by the gate keeps everything it wrote, but it did not
do what it was asked to, and the exit status is the only thing a queue
system or a shell `&&` can see.

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
would therefore shadow jcm's own `run/default.yaml` and `run/longrun.yaml`:
the atmosphere would silently be handed the coupler's run keys, and every
jax-gcm configuration bundle that says `override /run: longrun` would compose
the wrong file. `atmosphere.run` also already exists and means something
else. So the coupled run keeps its own name at both ends — `coupled_run=short_run`
selects an option, `coupled_run.total_time="90 days"` sets one key — and
`test_jcm_run_group_is_not_shadowed` pins it down. The *options* are named
apart for the same reason: `short_run`/`long_run` against jcm's
`smoke`/`longrun`, so no override reads as though it might be configuring the
atmosphere.

**YAML is wiring, and a test enforces it.** A key earns its place in a group
or configuration file only by being (a) `_target_`, (b) a required input
marked `???`, or (c) a value that differs from the Python default *and* is
what the named configuration is about. `test_config_has_no_python_defaults`
instantiates every group option and every configuration and fails if a
supplied value equals the target's own default, so a physics default cannot
acquire a second home in the configuration and drift from the class that owns
it. `jem.runners` never reads a physics parameter either: its one table is
`GROUP_TO_NAME` (which config group becomes which component name — `ocean`
→ `"ocn"`, and so on), and `test_runners_has_no_component_kwargs` fails if
any component parameter's name appears in its source at all. What the runner
*does* supply is what a config file cannot name — a surface component's
`SlabGrid` (from the built atmosphere's `coords.horizontal` and
`terrain.fmask`, or from a SCRIP `grid_file`), the regridders, and the
coupling timestep — injected into `hydra.utils.instantiate` after the keys
that describe them are removed.

**A grid goes only to a component that asks for one.** `_accepts_grid`
resolves the node's `_target_` (a class or a classmethod — both spellings
occur) and looks for an explicit `grid` parameter in its signature; a
`**kwargs` catch-all does not count, because that is the signature that
swallows the keyword and fails elsewhere. `VerosComponent.from_setup`
forwards every keyword it does not recognise to the Veros setup factory, so
an injected `grid=` would have been rejected *there*, with a message about
the factory. A component that takes no grid does not get one **built**
either: it brings its own bathymetry and land-sea mask, and a `SlabGrid` made
from the atmosphere's geometry would describe a grid nothing runs on.

**Packaged data resolvers.** `${jcm_data:bc/t30/clim/forcing.nc}` and
`${jem_data:DisplacedPoleGrid.SCRIP.nc}` resolve to files inside the
installed `jcm.data` and `jem.data` packages; importing `jem.config`
registers them. They exist so the shipped configurations run **offline**:
jax-gcm's own configurations fetch boundary data from an `hf://` mirror,
which needs the network and a warm cache, while everything named through
these is already on disk beside the code. A path that does not exist is
reported while composing, naming the key, rather than much later as a netCDF
open error.

`+atmosphere.constants.grav=9.7` reaches `jcm.runners.apply_constants_overrides`
before the model is built, because the dynamical core reads the live
`jcm.constants` singleton at construction. The override is process-global
and the surface components read the same singleton, so one such setting
moves the whole Earth system, not only the atmosphere.
