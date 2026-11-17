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
`+configuration@atmosphere=speedy-t31`. Leaving it out is at least loud:
`+configuration=speedy-t31` asks for that bundle at the *root*, where there
is no `physics` group for it to override, and Hydra refuses it with "Could
not override 'physics'. Did you mean to override
physics@atmosphere.physics?".
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
composes every group option and every configuration (it builds nothing),
resolves each `_target_` to the class or function it names, and fails if a
key the YAML supplies equals the default in that target's signature, so a
physics default cannot acquire a second home in the configuration and drift
from the class that owns it. `jem.runners` never reads a physics parameter either: its one table is
`GROUP_TO_NAME` (which config group becomes which component name — `ocean`
→ `"ocn"`, and so on), and `test_runners_has_no_component_kwargs` fails if
any component parameter's name appears in its source at all. What the runner
*does* supply is what a config file cannot name — a surface component's
`SlabGrid` (from the built atmosphere's `coords.horizontal` and
`terrain.fmask`, or from a SCRIP `grid_file`), the regridders, and the
coupling timestep — injected into `hydra.utils.instantiate` after the keys
that describe them are removed.

**The coupled run's defaults live once, on `run_chunked`'s signature.**
`coupled_run/default.yaml` is the schema of the run -- every key present, so
`coupled_run.chunk="10 days"` needs no `+` and `--cfg job` lists them all --
but a key that takes `run_chunked`'s default does not repeat it: it says
`chunk: ${run_default:chunk}`, and the `run_default` resolver (registered with
the data resolvers in `jem/config/__init__.py`) reads the default off
`inspect.signature(run_chunked)` when the config is resolved. So
`--cfg job --resolve` shows the real values, an override replaces the
interpolation as it would a literal, and changing a default on the signature
changes the command line with it. Three keys are literals on purpose, and the
file says why: `total_time` (`run_chunked` has no default run length; the
command line's own choice is 30 days), `output_dir` (null means the Hydra run
directory, where `run_chunked`'s own default is a relative `outputs`) and
`log_level` (not an argument of `run_chunked`). `health_check`, `accumulate`
and `initial_carry` are not keys: a YAML cannot name a live function or
carry. `test_run_options_repeat_no_run_chunked_default` and
`test_run_options_cover_every_configurable_run_chunked_keyword` hold both
facts.

**The precision is a configuration key, applied before anything is built.**
`jax_enable_x64` is a root-level key (`null` in `config.yaml`, which means
"leave JAX as the process has it" -- float32 -- so no default value is
repeated in YAML) that `jem.runners.build_coupler` applies with
`jax.config.update` *first*, before the atmosphere is constructed.
The flag is process-global and fixes the dtype of every array JAX allocates
afterwards, so a component that allocates while it is built (the
atmosphere's `Model`, a slab's climatologies) takes the precision of that
moment. `jax_enable_x64=true` on the command line gives a float64 slab run.

`ocean=veros` selects `true` itself: the option's file has a
`# @package _global_` header so it can set the root-level key next to its own
`ocean:` node. It is on the option, not on the two named Veros
configurations, because `ocean=veros` can be chosen without one, and Veros
cannot run in single precision here. Veros flips the flag when its operators
are first imported, which happens inside `VerosComponent.from_setup`, i.e.
after the atmosphere was built, so without the key the atmosphere is built in
float32 and stepped in float64 -- a mix whose composition depends on what
jax-gcm had allocated at the flip. The runner has no `if ocean is veros`; it
only applies the key.

After the components are built the runner compares the flag with what it
expected (the requested value, or what the process had when the key was
`null`) and **raises** `RuntimeError` naming the key if they differ, because a
component built before the flip is already allocated in the other precision and
nothing downstream can repair it. So `ocean=veros jax_enable_x64=false` is
refused rather than honoured. Two limits are worth knowing: the check needs
the flag to move, so in a process where Veros was already imported (its flip
spent) `false` is not detected and Veros' float64 arrays are silently held in
float32; and a Python-API user who builds the atmosphere in float32 and *then*
calls `VerosComponent.from_setup`, with no runner involved, is outside the
runner's reach (they must call `jax.config.update("jax_enable_x64", True)`
first; see `configure_veros_runtime`).

The change outlives `build_coupler`, `jem.runners.run` and
`jem.configurations.load` by design, like `atmosphere.constants`: the
returned coupler is only valid in the precision it was built in. A failed
`load()` restores the flag; tests that build a float64 configuration restore it
themselves. jax-gcm itself builds a few of its forcing leaves (the
solar-geometry scalars and an empty analytic-ozone array) as float32 whatever
the flag, so "float64 throughout" means every field with a spatial extent.

**A missing required input names its full key.** A `???` input (`ocean.sst_clim_file`
under `ocean=slab_relax`, the four `regrid.*` weight files under `regrid=esmf`,
`ocean.setup` under `ocean=veros`) is reported by OmegaConf as the bare leaf
(`Missing mandatory value: sst_clim_file`), which does not say which group it
belongs to. `jem.runners.build_component` re-raises it as
`MissingMandatoryValue` naming each full key and the override to add,
spelled with the packaged-file resolver where the repository's own examples use
one.

**`--help` lists every group's options.** Hydra's `$APP_CONFIG_GROUPS` fills in
JAX-ESM's own groups, but not the groups on the `pkg://jcm.config` search path,
so jax-gcm's re-rooted ones (and its `+configuration@atmosphere` bundles) are
listed by the `${jcm_groups:}` resolver, read from the installed package so
the listing cannot go stale.

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
registers them (and `run_default` and `jcm_groups`, above). They exist so the shipped configurations run **offline**:
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
