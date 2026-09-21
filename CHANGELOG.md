# Changelog

All notable changes to JAX-ESM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and JAX-ESM aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from v1.0.0
onwards. Before v1.0.0 the public API may change in any release; every such
change is listed here.

## [Unreleased] — 1.0.0b0, "the driver and configuration layer"

Phase 2 of the [API hardening plan][plan]. Phase 1 made a coupled model a thing
you could build in Python; this one makes it a thing you can *run*. It adds the
one chunked run loop every example and experiment driver used to re-invent, the
Hydra configuration that turns a coupled run into one command, the declarative
exchange that writes the standard coupling down once, and a checkpoint format
that validates what it loads. It also pins the jax-gcm revision all of that is
built against, so "which jax-gcm does this work with?" has an answer in the
repository.

Breaking changes are marked; everything else is additive.

### Added

- **`jem.driver.run_chunked(coupler, *, total_time, ...) -> RunResult`** — the
  one chunked run loop, and the home of every run default. Per chunk it
  integrates, labels and writes one file per component, checkpoints, and runs
  a health check on the result:

  ```python
  from jem import run_chunked

  result = run_chunked(
      coupler,
      total_time="6 years",        # 2190 days: a whole number of chunks
      chunk="30 days",             # a file, a restart and a health check a chunk
      output_dir="output",
      output_averages=True,        # one record per chunk: its 30-day-window mean
      # checkpoint_path="checkpoint" is the default, relative to output_dir
  )
  ```

  `total_time` and `chunk` are `jcm.date.parse_duration_days` strings or
  numbers of days, parsed on the *coupler's* calendar. Both must be whole
  multiples of the coupling timestep, and `total_time` a whole multiple of
  `chunk`; all three are checked before anything is compiled, with a message
  naming both quantities. A final partial chunk is refused rather than
  silently compiled a second time. The trajectory is compiled once, for a
  chunk's worth of coupled steps.
- **Checkpointing is on by default.** `run_chunked`'s `checkpoint_path`
  defaults to `"checkpoint"` rather than to `None`: a run long enough to be
  worth chunking is a run worth being able to restart, and a default of `None`
  made losing a week of compute the consequence of forgetting an argument.
  `checkpoint_path=None` (`coupled_run.checkpoint_path=null`) disables it.
  A **relative** path — including that default — is now resolved against
  `output_dir` rather than against the working directory, so each run gets its
  own restart directory (Hydra makes a fresh output directory per run) and two
  runs launched from one shell cannot overwrite each other's restart state;
  resuming is pointing a second run at the first's output directory, the same
  action that would otherwise overwrite its files, and the provenance line says
  which happened. An **absolute** path is used exactly as given, for a run that
  checkpoints to scratch while writing output elsewhere.
  `coupled_run/long_run.yaml` no longer repeats the setting.
- **`run_chunked(..., checkpoint_interval=...)`** — how often that checkpoint
  is written, in the same duration forms as `chunk`. `None` (the default, and
  `coupled_run.checkpoint_interval: null`) keeps a save after every chunk the
  health gate accepts; a value must be a whole multiple of `chunk`, since a
  chunk boundary is the only place the run stops, and is counted in coupled
  steps from the **start of the run** rather than of the call, so a resumed run
  checkpoints at the same points an uninterrupted one does. It is for a run
  whose chunks are short for one of the *other* reasons a chunk exists — a
  health check every few days, an output file per day — and which does not want
  its restart state rewritten that often. Two guarantees survive it, so no run
  loses work it would have kept without one: the last chunk of a completed run
  is always checkpointed, and a run the health gate stops checkpoints the last
  chunk that **passed** on the way out (an INFO line names the step it holds).
  What it gives up is a *killed* run, which falls back to the last interval
  boundary and re-integrates the chunks after it, **rewriting** their output
  files — safe because each is named after the coupled step its chunk starts
  at, and reported at INFO (how many existing files the resume will write
  again) before the run starts. That rewrite lands on the same names only
  while the resume keeps the same `chunk` and runs at least as far as the
  killed pass got: one that rechunks writes files at different steps and would
  leave the killed run's beside its own, holding records for the same
  simulated time, and one that stops earlier leaves that run's later files
  stranded past its end. Either is now **refused** — a `ValueError`, before
  anything is compiled, naming the files that would be left behind and which
  of the two they are, the restored step, the chunk and the ways out (resume
  under the chunk those files were written with, remove them, or use another
  `output_dir`) — rather than silently producing a directory with duplicate
  time labels; the driver does not delete a killed run's output on its own
  initiative. Files before the restart point, files whose names this coupler
  would never write, and every file in a directory a run that writes none (an
  accumulated run, or a call with nothing left to integrate) resumes into, are
  untouched and never block a run.
  It is validated with the other durations, before anything is compiled,
  and giving it with `checkpoint_path=None` is a `ValueError` rather than a
  setting silently ignored. A `total_time` that is not a whole number of
  intervals, and a run that starts part-way through a chunk — a resume under a
  different `chunk`, or an `initial_carry` at such a step, where no chunk before
  the last can end on a multiple of the interval — are WARNINGs rather than
  refusals:
  neither loses anything, but both mean the saves do not fall where they were
  asked for.
- **A run says where its starting state came from.** `run_chunked` logs one
  INFO line before anything is compiled — `Starting from coupler.initialize()
  at coupled step 0 (no checkpoint was given).`, `Starting from the
  initial_carry argument at coupled step N.` or `Resumed from checkpoint
  <path> at coupled step N.` — because resuming a run is the same command as
  starting one, so nothing else in the log distinguishes a restart from a cold
  start that repeats simulated time already paid for. A `checkpoint_path` that
  holds no complete checkpoint says so on the line before, in as many words:
  every component starts from its initial state rather than from a restart (or
  from the `initial_carry`, if one was given). A directory left without its
  carry file by an interrupted save is a WARNING; a path with nothing at it is
  INFO, because with checkpointing on by default that is what every first run
  sees. Both name the path. A run that finds it has nothing left to integrate
  also warns rather than noting it, since the caller asked for a run and got
  none. `Coupler.load_carry`
  / `jem.checkpoint.load_coupled_carry` log at INFO which components were
  restored from the shared carry file, which read themselves back through
  their own `load_carry`, and at what step, so every component's source is
  named. Loading is **all-or-nothing** and the docs now say so: the template
  built from `initialize()` supplies the pytree structure only, every leaf
  comes from the checkpoint, and a component the checkpoint does not hold is a
  `ValueError` — never a component quietly left freshly initialized while the
  rest of the model continues from the saved step.
- `jem.driver.RunResult` — a frozen dataclass with `final_carry`,
  `steps_completed` (`int(final_carry.step)`, so it counts from the run's start
  date and includes what a checkpoint restored), `completed` (False if the
  health gate stopped the run), `reports` (one per chunk), `paths` (every
  file written, in order) and `accumulator`.
- **`run_chunked(..., accumulate=(init, update))`** — the in-scan reduction,
  driven. Each chunk's trajectory is built with it and the accumulator is
  threaded from chunk to chunk, so a ten-year run reduces to monthly means
  without ever holding a chunk of diagnostics; the result is on
  `RunResult.accumulator`, for that reduction's own `finalize`. An accumulated
  run has no per-step diagnostics, and the three consequences are chosen, not
  inherited: it writes **no files** (`paths` is empty, and `output_averages` /
  `subsample` reduce the files, so they do nothing and the run says so); it
  **refuses** a `health_check`, because the gate is on by default and a long
  accumulated run of an atmosphere is exactly the run that needs it, so
  `health_check=None` makes going without it a decision rather than a log
  line; and the accumulator is **not checkpointed**, because the checkpoint is
  the model's restart state while the accumulator is an analysis product —
  storing one in the other would make the checkpoint format depend on which
  reduction the run chose. The carry is checkpointed as usual, a resumed run
  starts a fresh accumulator, and it warns when both are given.
- `jem.driver.default_health_check(datasets, chunk_index, elapsed_days)` — it
  runs `jcm.diagnostics.check_health` on `datasets["atm"]`, so a coupled run
  stops on the same evidence an uncoupled atmosphere does. A coupled model with
  no atmosphere reports `{"skipped": "no atmosphere"}` and the run continues:
  an abstention, not a pass. `health_check=None` disables the gate entirely,
  and `bail_on_unhealthy=False` logs the failure and keeps integrating, which
  is what a run studying the instability itself wants.
  `jem` exports `run_chunked`, `RunResult` and `default_health_check`.
- **`jem.runners`** — composed config to built objects, and the only module
  that reads the configuration. `build_atmosphere`, `build_grid`,
  `build_component`, `build_regridders`, `build_exchangers`, `build_coupler`
  and `run(cfg) -> RunResult`. It is deliberately generic: its one table is
  `GROUP_TO_NAME = {"ocean": "ocn", "land": "lnd", "seaice": "seaice"}`, and
  `test_runners_has_no_component_kwargs` fails if any component parameter's
  name appears in its source at all, so a new component is configured by adding
  a group file and never by adding a branch to the runner. What it injects is
  what a config file cannot name: a surface component's `SlabGrid` (from the
  built atmosphere's `coords.horizontal` and `terrain.fmask`, or from a SCRIP
  `grid_file`), the regridders, and the coupling timestep. A grid is injected
  only into a component whose `_target_` actually declares a `grid` parameter
  (the signature is inspected; a `**kwargs` catch-all does not count), and for
  one that does not — a Veros ocean, which brings its own bathymetry and mask —
  no grid is built at all. `+atmosphere.constants.*`
  reaches `jcm.runners.apply_constants_overrides` *before* the model is built,
  and because the surface components read the same process-global singleton,
  one such override moves the whole Earth system.
- **`jem.main`, and a `jem` console script** (`[project.scripts]`). A coupled
  run is now one command:

  ```bash
  python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run
  python -m jem.main --help          # every group, option and override spelling
  python -m jem.main +configuration=earth-slab --cfg job   # compose, don't run
  ```

  It sets the `jem` logger from `coupled_run.log_level`, logs the composed
  config and calls `jem.runners.run`. Nothing prints. It exits `0` when the run
  reached the time it was asked for and `1` when the health gate stopped it
  early, so a scheduler, a CI job or a shell `&&` sees a stopped run as the
  failure it is; the output and the checkpoint written up to that point are
  kept, and the last health report is logged at ERROR.
- **`jem/config/`** — JAX-ESM's Hydra configuration. `config.yaml` puts
  **jax-gcm's own config groups under `atmosphere`** through
  `hydra.searchpath: pkg://jcm.config`, so `cfg.atmosphere` is exactly the
  config `jcm.runners.build_model` expects and jcm's group and option names
  are unchanged; JAX-ESM's own groups (`ocean`, `land`, `seaice`, `coupling`,
  `regrid`, `coupled_run`) sit at the top level, and `configuration` composes a
  named coupled model out of all of them (`aquaplanet-slab`,
  `aquaplanet-slab-mixed-grid`, `earth-slab`, `veros-double-drake`,
  `veros-earth`). The override spellings, all verified:

  | To do this | Write this |
  | --- | --- |
  | Run a named coupled configuration | `+configuration=earth-slab` |
  | Compose a whole jax-gcm bundle as the atmosphere | `+configuration@atmosphere=speedy-t31` |
  | Change one atmosphere group | `physics@atmosphere.physics=held_suarez grid@atmosphere.grid=held_suarez_t31_l8` |
  | Set one atmosphere key | `atmosphere.run.time_step=7` |
  | Choose a surface component | `ocean=slab_relax ocean.sst_clim_file='${jcm_data:bc/t30/clim/forcing.nc}'` |
  | Drop one | `land=none` |
  | Set a component parameter | `+ocean.params.relaxation_time=1e6` |
  | Override a physical constant | `+atmosphere.constants.grav=9.7` |
  | Choose the run settings | `coupled_run=short_run`, `coupled_run.total_time="90 days"` |

  Spell the `@atmosphere`: `+configuration=speedy-t31` without it composes that
  jax-gcm bundle at the *root*, where its `physics`, `terrain` and `run` keys
  are nobody's and nothing reads them.
- The coupled run's group is **`coupled_run`**, not `run`, for two reasons.
  Hydra resolves a group option from the first search-path entry that has it,
  and this package precedes `pkg://jcm.config`, so a `run` group here would
  shadow jcm's own `run/default.yaml` and `run/longrun.yaml` — silently, and in
  every jax-gcm configuration bundle that says `override /run: longrun` (16 of
  the 19 shipped ones). And `atmosphere.run` already exists and means something
  else. `coupled_run/default.yaml` is the group's complete schema — every key
  the driver reads, with the default `run_chunked` gives it — and `short_run`
  and `long_run` inherit it, so any key is overridable without a `+`. Those two
  option names are the group's own, distinct from jax-gcm's `run=smoke` /
  `run=longrun`: those configure the *atmosphere*, and a coupled option spelled
  like an atmosphere option is the same confusion the group's name already
  exists to avoid. `long_run` is ~10 years in 30-day chunks with one averaged
  record per chunk; its `total_time` is spelled in days (3600) because
  `total_time` must be a whole multiple of `chunk` and 30 days does not divide
  a 365-day year, so `coupled_run=long_run coupled_run.total_time="1 year"` is
  refused — override it with a multiple of 30 days ("6 years" = 2190). Those
  averaged records are 30-day *window* means, whose boundaries drift about five
  days a year against the calendar, not calendar-month means: those are
  `jem.accumulate.monthly_mean`, which bins each record by its own label. A
  test composes every shipped option and fails if its `total_time`/`chunk`
  pair does not divide.
- **The YAML is wiring only**, and a test says so: a key earns its place by
  being `_target_`, a required input (`???`) or the non-default value that
  makes a named configuration what it is.
  `test_config_has_no_python_defaults` instantiates every group option and
  every configuration and fails if a supplied value equals the target's own
  default, so a default cannot acquire a second home here and drift.
- Packaged data is named through two OmegaConf resolvers registered by
  importing `jem.config`: `${jcm_data:bc/t30/clim/forcing.nc}` and
  `${jem_data:DisplacedPoleGrid.SCRIP.nc}`. The shipped configurations
  therefore run **offline**, unlike jax-gcm's own, which fetch boundary data
  from an `hf://` mirror. The config groups are package data
  (`[tool.setuptools.package-data]`), so an installed wheel can run them.
- **`jem.exchangers`** — the declarative form of an exchanger, which every
  example in the repository used to write out by hand.
  `ExchangeSpec("component.section.field", "component.section.field",
  regrid=None)` is one row of a coupling table and `Exchange(specs,
  regridders)` executes it as an ordinary `Exchanger`: it reads **every** source
  from the mapping as it arrives before writing anything, so the result does
  not depend on the order of the table, and it builds new carries with
  `.replace(...)` rather than writing in place. `Exchange.validate(carries)` is
  the eager pre-flight — `jem.runners` calls it with `coupler.initialize()` —
  which turns a mistyped component, section, field or regridder into an error
  naming the spec, before a model is integrated.
- `jem.exchangers.default_exchanges(components, regrid=None)` and
  `default_exchangers(...)` — the standard atmosphere/ocean/land/sea-ice
  coupling, in one place:

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

  A row survives only if both its components are present, so an aquaplanet with
  no land model gets the four rows that do not mention `lnd`. `regrid` is keyed
  by direction and kind (`a2o_flux`, `o2a_state`, or a bare `a2o`/`o2a`), the
  kind written on each row — which is exactly the split the mixed-grid
  example makes by hand: conservative maps for fluxes and the areal ice
  fraction, bilinear for the sea surface temperature.
  There is **one table per carry layout**: the rows above (`STANDARD_EXCHANGES`)
  for JAX-ESM's own slab components, and `VEROS_OCEAN_EXCHANGES` when the
  component registered as `"ocn"` is a `VerosComponent` — the same wiring in
  Veros' own field names (`ocn.forcing.heat_flux`,
  `ocn.forcing.freshwater_flux`, `ocn.derived.sea_surface_temperature`), so the
  shipped `veros-double-drake` and `veros-earth` configurations get a coupling
  that validates instead of one written for a slab carry. The wind stress is
  not in it and cannot be — Veros integrates a stress, the atmosphere publishes
  a wind, and a drag law is not a copy — so those configurations run
  thermodynamically forced and mechanically at rest until `coupling.exchanger`
  names a hand-written one. A Veros ocean coupled to a sea-ice component is
  warned about, because Veros publishes no freeze/melt potential to drive it.
  The Veros wrapper's module is looked up in `sys.modules` rather than
  imported, so nothing here depends on the optional Veros install.
  `default_workflow` is the order a `Coupler` runs by default. The module
  docstring is where the
  **one-step coupling lag** the default workflow implies is finally written
  down: with `["exchange", "atm", "ocn"]` the ocean at step *n* is driven by the
  atmosphere's fluxes from step *n−1*, and the first step of a run exchanges
  whatever `initialize()` left in each forcing section — zeros, for every
  packaged component. `jem` exports `Exchange`, `ExchangeSpec`,
  `default_exchanges`, `default_exchangers` and `default_workflow`.
- **`jem.regrid.ESMFRegridders`** — the named collection of
  `ESMFRegridder` objects a mixed-grid run exchanges through, built from
  weight files by the `regrid` config group (`a2o_conserve`, `a2o_bilinear`,
  `o2a_conserve`, `o2a_bilinear`). It is an immutable `Mapping`, because the
  maps a run uses are fixed when the model is built, and an empty one is
  refused with a pointer at `regrid=same_grid` — which is how a single-grid run
  says it needs none. `jem.runners` also exposes each map under the exchange
  role it plays, so a coupling table may name either vocabulary.
- **`jem.output`** — the last step of a chunk, which every driver re-invented:
  `chunk_datasets(coupler, diagnostics, *, first_step)` (label a chunk's
  records, reduce nothing), `postprocess(dataset, *, output_averages,
  subsample)` and `postprocess_datasets` (the same reduction over a whole
  chunk), `write_chunk(datasets, output_dir, first_step)`, and
  `datasets_for_chunk`, which is the labelling and the reduction in one call.
  The labelling and the reduction are separate because a chunk has two
  consumers: what is *written* is reduced, while what the health gate
  *inspects* must not be. `output_averages` is defined against jcm's meaning
  of the same word: jcm replaces each saved record with the mean over its save
  interval, labelled at the interval's end, and the coupler's records are
  already one per coupling step — so the coupler's output interval is the
  **chunk**, and the flag replaces a chunk's records with their mean, labelled
  with the chunk's last time and carrying `cell_methods = "time: mean"`. Files
  are `<component>-<first step:08d>.nc`: the component first so a listing groups
  a component's files, and the **coupled step the chunk starts at** second,
  zero-padded so the listing sorts in run order. The step rather than a chunk
  index, because the chunk length belongs to the run and not to the checkpoint:
  a run resumed with a different `chunk` gives the same simulated time a
  different index, and would write over a file the earlier run had already
  written. An existing file is overwritten with a WARNING — a resumed run never
  collides, so it means a rerun into the same directory. `jem` exports all
  three.
- **`jem_role`**, a variable attribute on every packaged component's output
  saying which section of the carry a variable came from — `state`, `derived`
  or `forcing` — built by `jem.base.component.role_attrs(role)`. The
  `forcing_` name prefix stays, because it is what stops an `xr.merge`
  collision between a field one component computed and the lagged copy another
  received; this is the metadata that makes
  `ds.filter_by_attrs(jem_role="forcing")` the whole query, instead of parsing
  names. A variable that is none of the three — a grid mask, a layer thickness
  — is deliberately left untagged, and the JCM wrapper tags only the surface
  boundary conditions an exchanger writes into jcm's own dataset, because the
  rest of those names are jcm's and their roles are not JEM's to assert.
- **`psi`, the ocean's barotropic streamfunction**, in the Veros component's
  output (`["time", "lon", "lat"]`, `m^3/s`, `jem_role="derived"`). Veros
  carries a real streamfunction only when the setup solves the external mode
  for one (`settings.enable_streamfunction`); under the linear free surface
  every Veros setup shipped with JEM chooses, the same `variables.psi` array
  holds the *surface pressure* instead (`m^2/s^2`, on the T grid), so
  publishing it unconditionally would have published a different quantity
  under the streamfunction's name. `psi` is therefore Veros' own field when
  the run solves for one, and is otherwise diagnosed from the
  depth-integrated zonal transport by the discrete relation Veros inverts
  *when it does solve for a streamfunction* —
  `sum_k u dzt maskU = -(psi[i,j] - psi[i,j-1]) / dyt[j]`, the relation behind
  its barotropic-mode update — integrated northwards from a southern boundary
  where it vanishes. A free-surface run never reaches that code: it solves for
  a surface pressure, and the barotropic mode enters the momentum equation as
  a pressure gradient. What carries over is therefore the *definition*,
  applied to the transports that run did produce; that it is the right one,
  with the sign and the metric Veros uses, is pinned by a test that makes the
  diagnosis reproduce Veros' own `psi` where Veros has one. Which of the two
  the run used is recorded in the variable's `comment` attribute, with the
  caveats that a free-surface barotropic flow is not exactly non-divergent
  (so the diagnosis is the standard "meridionally integrated zonal transport"
  rather than an exact streamfunction) and that, like `u` and `v`, the field
  sits on Veros' staggered grid — the zeta points — while wearing the T-grid
  `lon`/`lat` labels the dataset uses throughout. A free-surface run also
  publishes **`ssh`, the sea surface height** (`["time", "lon", "lat"]`, `m`,
  `jem_role="derived"`) — the sea surface height of that surface-pressure
  solve, `ssh = psi / grav`, the relation Veros itself applies when it sets
  `variables.ssh`, on the T grid the dataset's `lon`/`lat` already label. The
  relation is applied to the `psi` of the record's own time level rather than
  `variables.ssh` being read back, because Veros writes that field before
  permuting its time indices, leaving it one Veros timestep behind the `psi`,
  `u`, `v` and tracers of the same record. A streamfunction run carries no sea
  surface height, so the variable is absent there; the key set is fixed per
  component at construction, not per step.
- **`mask_U`, `mask_surface_U` and `mask_surface_Z`** beside `mask_T` in that
  same output. `psi`'s values over land are carried through the integration
  rather than computed, so a reader needs the zeta-point mask to blank them,
  and the u-grid mask is the one its depth integral ran over — with `dzt`,
  that makes the diagnosis reproducible from the file alone. Grid
  configuration rather than carry, so like `mask_T` they carry no `jem_role`.
- **`jem.checkpoint`** — `save(carry, path)` / `load(template, path)` for any
  pytree, and `save_coupled_carry` / `load_coupled_carry` for a whole
  `CoupledCarry`, in
  the format jax-gcm already uses: the leaves flattened to typed arrays and
  serialised with flax's MessagePack (`msgpack`) codec — a compact binary
  serialization format, a binary cousin of JSON, reached through
  `flax.serialization.msgpack_serialize`, hence the `carry.msgpack` file name —
  the tree they came from *rebuilt* from
  a template at load time and recorded beside them only as a manifest to check
  that template against. Every leaf is checked against the template's path,
  shape and dtype, so a checkpoint from another grid or another component
  composition is a `ValueError` naming the leaf instead of a wrong resume, and
  the int and bool flags in a physics carry keep their dtypes. The manifest
  also holds the repr of the whole `PyTreeDef`, checked after the leaves,
  which is what catches the mismatches no leaf can show: a component whose
  carry holds no arrays (`{}` or `None`) contributes no leaf, so renaming one
  — or renaming a component that checkpoints itself, whose carry never reaches
  the shared file at all — would otherwise load cleanly and resume a different
  model at the saved step. A delegated component's *name* is stored in the
  shared carry file as an empty marker entry for exactly that reason. Static
  (`pytree_node=False`) parameters live in the `PyTreeDef` too, so resuming
  with one of them edited — `forcing_method` from `none` to `qflux` — is
  refused, deliberately: it selects a code path at trace time, so the resumed
  run would be a different model, and the message says so and shows the
  difference. A *differentiable* parameter is a leaf and is restored from the
  checkpoint as before.
  `latest_complete_checkpoint(root,
  pattern="step_*")` and `remaining_batches(steps_done, total_steps,
  steps_per_batch)` moved here from `jem.utils.checkpoints` unchanged.
- `jem.accumulate.monthly_mean(coupler, total_time=… | n_months=…)`,
  `jem.accumulate.windowed_mean(coupler, window, n_windows=…)` and
  `Coupler.generate_trajectory_function(iterations, accumulate=(init, update))`
  — an **in-scan reduction** of the per-step diagnostics. `update(accumulator,
  diagnostics, time)` runs inside the `lax.scan` body, with the same
  `CouplingTime` that step's components were handed, and the scan returns
  nothing per step, so the memory a call needs no longer grows with
  `iterations`; the call becomes `(carry, accumulator=None) -> (carry,
  accumulator)` and a chunked run threads the accumulator across chunk
  boundaries. `monthly_mean` needs only the coupler — the diagnostics' shapes
  come from `jax.eval_shape` of one step, and a record's month is a
  `searchsorted` in a static table of month boundaries, reached from the record
  counter reduced modulo the records in a year — so a whole year is one
  compiled trajectory instead of the three that chunking by calendar month
  would need. Which month a record counts in follows the end of the interval it
  covers — the instant JEM labels it with — read on the **model** calendar, so
  `monthly.finalize(acc)` and `to_xarray(...).groupby("time.month").mean()` of
  the same run agree for a run whose output labels cross no Gregorian 29
  February (and, for a sub-stepped component, after `fold_records`, below).
  That condition is the labels' calendar, not the binning: labels are proleptic
  Gregorian whatever the model calendar is (JCM's convention, jax-gcm#449;
  calendar-consistent labels are tracked as #118), so a `365_day` run started
  on 1 January 2000 labels the record the model calls 1 March 00:00 as
  `2000-02-29` and accumulates it into March. A `groupby("time.month")` of the
  written output therefore moves the first record of every month from March on
  into the month before it, gives February the record the model calls 1 March,
  and hands December the year's wrap record — the one the model calls 1 January
  of the next year — that the twelve-bin form counts in January; the
  accumulated bin stays the model's month, which is the month the forcing and
  the seasonal cycle follow. Reproduce it from the written output
  by binning on model day-of-year (each label's offset from the start date in
  whole days) rather than on `time.month`. A calendar with no fixed month table
  (gregorian) and a coupling step that does not divide the year are refused
  with a message saying why. **Without `accumulate` the generated
  function is exactly what it was.**

  `monthly_mean(coupler)` bins into the **twelve** calendar months, so a
  multi-year run composites its Januaries into one bin — a climatology.
  `monthly_mean(coupler, total_time="10 years")` (or `n_months=121`) bins into
  the months the run **passes through**, in order, each with a bin of its own:
  the same month table rotated to the month the run starts in and phased to the
  start date, so it is calendar months whatever day the run begins on, and
  nothing drifts the way a fixed 30-day window does. The size counted from
  `total_time` is the months the run's labels touch, which is why ten years is
  121 bins and not 120 — the last record is labelled 00:00 on 1 January of the
  eleventh year, which is that January's, and without a bin for it the
  accumulator would wrap it into the first January. A run longer than the
  accumulator wraps at the **span** of its bins, as a windowed mean wraps at
  the span of its windows, so a wrapped bin is a calendar month only when
  `n_months` is a multiple of twelve and otherwise holds parts of two —
  `total_time`, which is never wrapped into, is the spelling to prefer. (That
  span is rounded up to the next whole coupled step, because the record
  counter is reduced modulo it and a whole number of calendar months need not
  be a whole number of steps: a 5-day coupling divides the 365-day year but
  not the 59 days of January and February. The wrap moves by less than one
  step, every bin boundary stays exact, and neither a `total_time` accumulator
  nor a calendar-aligned wrap can see it.) The two forms are mutually
  exclusive; giving neither is the climatology.

  `windowed_mean` is the same reduction over `n_windows` windows — of one
  fixed length, which is what a sub-seasonal forecast is scored on, or of a
  repeating **pattern** of lengths, which the windows cycle through:

  ```python
  pentads = windowed_mean(coupler, "5 days", n_windows=73)          # a year
  weeks   = windowed_mean(coupler, "7 days", total_time="1 year")   # 53 of them
  leads   = windowed_mean(coupler, [1, 1, 1, 1, 1, 1, 1, 5, 5],     # cycled
                          total_time="30 days")
  ```

  Each length is a `jcm.date.parse_duration_days` string or a number of days
  and must be a whole number of coupling steps; the accumulator's size is
  given directly as `n_windows` or counted from `total_time` (rounded up, so a
  run that does not divide into whole windows still has a bin for the one it
  ends inside), and a sequence given neither is used once through. Both
  builders return the same `BinnedMean` named tuple from one private
  `_build_binned_mean(coupler, bin_of_record, n_bins)` with one private
  `_variable_window_rule(boundaries, offset, inclusive)` — a calendar month is
  that rule with the month boundaries, the run's phase in the calendar and
  bins closed at their start — and both bin every record by its own **label**;
  a run longer than the accumulator wraps, so window *w* composites every
  *w*-th window exactly as the twelve monthly bins composite years.

  A window is **not** a calendar month, whatever its length, and neither
  builder pretends otherwise: a window is measured from the run's own start
  date with no calendar phase and closes at its **end** (JEM labels a record at
  the end of the interval it covers, and a window is one such interval), while
  a calendar month is phased to the calendar and closes at its **start**
  (which is what `groupby("time.month")` does). So a 31-day window started on 1
  January takes the record labelled 00:00 on 1 February, which is February's
  month, and from a 1 July start a pattern of month lengths is not months at
  all. There is deliberately no `inclusive=` or `offset=` knob to mix the two:
  each convention is what makes its own builder agree with the thing it must
  agree with.

  **`jem.accumulate.month_lengths(calendar_or_coupler)`** is public so that an
  analysis can weight or label months without rebuilding the table: the twelve
  month lengths in days of a fixed-length calendar, January first, taken from a
  coupler, a calendar name or a year length, refusing `gregorian` (whose leap
  years change the table from year to year).
  **`jem.accumulate.fold_records(means, counts)`** is the count-weighted fold
  of a component's kept sub-step axes — what makes a sub-stepped component's
  binned mean equal a `groupby` of its output (on the bins' own terms; for a
  monthly mean, under the leap-year condition above), and a no-op on a
  component that records once per coupled step. A coupled step is not always
  one record: a component the workflow runs *n* times per step emits *n*, and
  a nested coupler's inner steps are records in the same way, so the 24 hourly
  records of the daily step covering 31 January are binned 23 in January and
  one (at 00:00 on 1 February) in February, which is where `to_xarray` writes
  them too whenever label and model calendar agree. That component's
  sub-step axis is kept — `(n_bins, n, ...)`, a monthly-mean diurnal cycle —
  and the accumulator's counts are one array per component when components
  record at different rates, instead of the single `(n_bins,)` array a model
  whose components all record once per coupled step keeps. The accumulator is an ordinary pytree in the scan
  carry, so a binned mean is **differentiable** — `jax.grad` of a loss on
  `finalize(...)` reaches a component parameter through the reduction, which
  is what calibrating against monthly observations needs; there is a worked
  example in `docs/source/design/architecture.md` and a test of it in
  `tests/unit/test_accumulate.py`.
- **`VerosComponent.from_setup(setup, ...)`** — the config-shaped
  door to Veros. The constructor takes an already-built Veros model, which is a
  live Python object no YAML can describe, so the `ocean=veros` group had no way
  to select it; `from_setup` imports a dotted path (never a file path), builds
  the setup with the keys the config supplies, runs the setup's own `setup()`
  and wraps it. Both a `VerosSetup` subclass and a factory returning one are
  accepted, decided on what calling it returns.
- **`jem/components/jcm/contract.py`** — the jax-gcm revision JAX-ESM is
  supported against, and every jax-gcm name it reaches for.
  `JCM_SUPPORTED_REV` is `9e399ab2`, and `JCM_SUPPORTED_VERSION` the
  `3.0.0rc1` that revision reports. It is a `dev` sha rather than a tag
  because no tagged jax-gcm release carries the four changes JAX-ESM is
  written against — #750's one run schema and `configuration` group, #763's
  input-resolution engine, #819's removal of jax-gcm's own logging
  configuration, and #824's public resumable state (`Model.bootstrap_state()`
  returns the `(dycore_state, physics_carry)` pair;
  `ModelPredictions.with_context(model)` repairs a prediction object a
  `lax.scan` round trip stripped) and public date conversion
  (`Model.date_from_sim_time`, with `Model._date_from_sim_time` kept only as
  a delegating alias, which is why the season-freeze helper in
  `examples/02_experimental/03_jcm_veros_earth` overrides the public name:
  overriding the alias would leave jax-gcm's own callers on the unpatched
  method and let the season go on advancing silently). The `jcm>=3.0.0rc1`
  floor in `pyproject.toml` is the loosest true statement of the same pin,
  since jax-gcm bumps its version only at release.

  `JCM_INTEGRATION_POINTS` records each name with what it is used for: the
  calls, the constructors JAX-ESM's documented workflow asks a user to make,
  the package data it resolves, the underscore-prefixed SPEEDY diagnostics
  keys the surface exchange reads (jax-gcm#754 is the issue that will replace
  them with a published struct), and the one private attribute JAX-ESM still
  depends on — `cf_metadata._COORD_ATTRS`, whose *values* the slab components
  copy so their axes describe themselves exactly as the atmosphere's do.
  Every other jax-gcm attribute the JCM wrapper touches is public at the
  pinned revision. `tests/unit/test_jcm_contract.py` walks the list against
  the installed `jcm`, so a jax-gcm rename fails as "jax-gcm renamed or
  removed X, which JAX-ESM used for Y" instead of as an `AttributeError` in
  the middle of a run. Every required CI job checks that revision out through
  a workflow-level `JCM_REV`, which the test asserts equals
  `JCM_SUPPORTED_REV`; a non-blocking `canary-jcm-dev` job keeps tracking
  `dev` so drift stays visible without blocking a pull request.

- **`jem.replace_field(carry, path, value)` / `jem.read_field(carry, path)`**
  — write or read one `"component.section.field"` of a coupled carry, the
  same address `jem.exchangers.Exchange` already uses. Every example that
  customised a single initial condition rebuilt three nested containers by
  hand to do it (`dict(carry, components=dict(carry.components, ocn=dict(
  ocean_carry, state=ocean_carry["state"].replace(...))))`), and each did it
  slightly differently; `replace_field` is that rebuild written once, and
  works equally on a whole `CoupledCarry` or the bare `dict[str, Carry]`
  mapping an exchanger is handed. Both live in `jem/exchangers.py`, which
  already owns the path vocabulary and its error messages, rather than in a
  new module.
- **`jem.plot`** — the plotting the example notebooks share:
  `open_output` glues a chunked run's files for one component back into one
  dataset, `area_mean` is the cos(latitude)-weighted horizontal mean,
  `map_plot` draws one 2-D field (handling both a separable lon/lat grid and
  a curvilinear one, and the `(..., lon, lat)` transpose every JEM field
  needs), and `animate_map` steps it through time. The four notebooks that
  produced a map each carried ~90 lines of their own cartopy animation code;
  this is that written once. Behind the `plot` extra, with matplotlib and
  cartopy imported inside the functions that need them, so `import jem` (and
  `import jem.plot`) never requires either.

### Changed

- **jax-gcm configures no logging of its own, and `jcm.model.Model` takes no
  `log_level` keyword** (jax-gcm#819, at the pinned revision). Nothing under
  `jem/` ever passed it — only the test fixtures did, to silence the
  `logging.basicConfig` that `import jcm` used to run — and it is gone from
  them. A JAX-ESM process now configures its own logging and nothing else's,
  which is what `jem.main` already assumed when it set the level of the `jem`
  logger alone.
- **A coupled run's atmosphere dataset now carries jax-gcm's
  `parameters_rederived_from_live_context` note** inside the
  `jcm_prov_params` global attribute (and so a different
  `jcm_prov_params_sha`), because the stacked predictions are repaired with
  `ModelPredictions.with_context(model)` before they are serialized. The note
  is accurate — a coupled trajectory is traced once and scanned, so the
  parameters in that record are read from the live physics after the fact,
  not captured at trace time — and saying so is the point of it.
- **`TimeAxis.datetimes()` still reproduces JCM's *output* arithmetic** and
  deliberately does not adopt the now-public `Model.date_from_sim_time`: that
  is JCM's model-clock conversion, not the float64 arithmetic its output
  files are labelled with, and for a coupling step that is not a
  power-of-two fraction of a day (10 or 20 minutes, say) the two differ by up
  to 128 ns — enough to take JEM's labels off the atmosphere's time axis.
  Sharing one computation needs JCM to publish its *output* labelling, which
  is jax-gcm#862; the reasoning is recorded on `TimeAxis`.
- **`test_installed_jcm_matches_contract` checks `jcm.__version__`**, not the
  distribution metadata. An editable install records its version when it is
  installed, so a jax-gcm checkout moved to another revision keeps advertising
  the old one — exactly the situation a pin bump creates, and a check that
  passes on stale metadata is worse than none.
- **`SlabSeaiceModel`'s constructor default is `name="seaice"`, not
  `name="ice"`** (breaking). The standard coupling wires the sea ice under
  `seaice`, which is what every example registers it as, so the old default
  produced a model that silently received nothing and published nothing.
  `SlabSeaiceModel(grid, name="seaice")` still does exactly what it did, so no
  existing code has to change; code that *relied* on the carry, workflow and
  output key being `ice` must now pass `name="ice"` explicitly.
  `default_exchanges` still warns when a component is registered under a name
  that just misses the standard wiring (`ice`, `ocean`, `land`, `atmosphere`),
  because the failure it causes is otherwise silent.
- **The checkpoint format is MessagePack, and old checkpoints cannot be read**
  (breaking). `jem/utils/checkpoints.py` wrote one pickle per component plus a
  pickled step counter; a pickle ties a checkpoint to the classes that wrote it
  and validates nothing. A checkpoint directory now holds one
  `carry.msgpack` — every component that does not write itself, plus
  `CoupledCarry.step`, plus the *name* of every component that does — beside
  one subdirectory per `SupportsCheckpoint` component. **`carry.msgpack` is written last**, through a flushed and fsynced
  temporary renamed into place, and *is* the completion marker: a resume cannot
  do without the clock, which lives in that file, so its presence is exactly the
  condition "this checkpoint is loadable". The separate `coupled_step.pkl`
  marker is gone with the pickles. Re-run from the beginning, or from an
  initial carry built in Python.
- **`save_state` / `load_state` are now `save_carry` / `load_carry`**
  (breaking), on the `SupportsCheckpoint` capability and on both its
  implementations, `Coupler` and `VerosComponent`. What a checkpoint holds is
  a component's *carry* — its state and its parameters together — so "state"
  named only half of it, and the pair now reads the same as the
  `jem.checkpoint.save_coupled_carry` / `load_coupled_carry` it is built on.
  Rename the calls — `model.save_carry(carry, directory)` /
  `model.load_carry(directory)`; no alias is kept under the old names, and
  because the capability is matched by method name, a component that still
  defines only the old pair is no longer recognised as checkpoint-capable:
  its carry goes into the shared `carry.msgpack` like any other component's
  rather than through its own writer.
- `Coupler.load_carry` takes the template it needs from the components' own
  `initialize()`, so a checkpoint is loaded into the model meant to continue it
  and a component added, removed or rebuilt on another grid since it was written
  is reported — naming the leaf — rather than papered over.
- `jem.driver.run_chunked`'s checkpoint is **one directory, rewritten after
  every chunk**, not a directory of dated restart points. That is what makes
  resuming the same command as starting: point at the path, and the run either
  starts fresh or continues from the coupled step the checkpoint holds. A
  directory with no carry file is what an interrupted save leaves behind, so it
  is logged and stepped over rather than resumed from. A run that wants a
  history of restart points keeps its own directory of them and passes each in
  turn; `latest_complete_checkpoint` is still there for that.
- The README quick start builds its coupling with `default_exchangers(components)`
  and runs it with `run_chunked`; the hand-written exchanger it used to show is
  now the worked example in `docs/source/tutorial.rst` and
  `docs/source/design/architecture.md`, where the contract it illustrates is
  described.
- `pyproject.toml` ships `config/**/*.yaml` as package data and declares the
  `jem` console script; its `jcm>=3.0.0rc1` floor now points at `contract.py`
  for the actual pin.
- **The six example notebooks are rewritten against the configurations Phase
  2 shipped.** The four ordinary ones (aquaplanet, mixed-grid aquaplanet,
  Earth-like, and the long-aquaplanet driver retired below) are now one
  `python -m jem.main +configuration=...` run plus a short plotting section
  built on `jem.plot`; the two bespoke ones (a customized initial sea surface
  temperature, the `jax.jvp` response to an SST bump) build their coupler
  with `jem.runners.build_coupler(compose(...))` and customise only the one
  thing that is theirs, through `jem.replace_field`. No notebook builds its
  components, its exchanger or its coupler by hand any more, and none writes
  netCDF or an animation by hand either. `examples/README.md` is the new
  index of which command or notebook runs which example.
- `tests/examples/test_examples.py` runs notebooks only, one test per
  notebook (`@pytest.mark.parametrize`, so a failure names the notebook that
  caused it) rather than one test per example group; the `run.sh` driver it
  used to also execute is gone (see *Removed*).

### Removed

- **`jem.utils.checkpoints`** (whole module, breaking): `save_carry`,
  `load_carry`, `save_component_carries`, `load_component_carries`,
  `save_coupled_carry`, `load_coupled_carry`, `save_veros_carry` and
  `load_veros_carry`. The first six are `jem.checkpoint.save` / `load` /
  `save_coupled_carry` / `load_coupled_carry` under their existing names;
  `latest_complete_checkpoint` and
  `remaining_batches` moved to `jem.checkpoint` unchanged; the two Veros
  functions are now `VerosComponent.save_carry` / `load_carry`, where they
  belong — the HDF5 restart is Veros' business, not the coupler's. Call
  `Coupler.save_carry` / `load_carry` rather than any of them.
- **`examples/02_experimental/03_long_aquaplanet.py`**, a hand-rolled
  chunked driver (a T106 aquaplanet, 100 model years in 30-day batches, with
  its own per-batch netCDF write, time mean and NaN check). Every one of
  those is now a feature of `run_chunked`/`coupled_run=long_run`; the
  command that replaces it is the "Long aquaplanet at T106" row of
  `examples/README.md`.

### Fixed

Defects found by the local review of this change before it was pushed, all in
code this release adds:

- `jax` and `jaxlib` are capped below 0.11.2 for as long as no released flax
  survives it: jax 0.11.2 removed `jax.experimental.hijax.HiPrimitive`, which
  flax 0.12.9 subclasses at import time, so an environment resolving the two
  latest releases could not import `jcm` at all (#117 tracks lifting it).
- `run_chunked` validates `subsample` before compiling a trajectory, instead of
  after the first chunk has been integrated.
- `jem.runners` no longer reads a broken `_target_` lookup as "this component
  takes no grid"; `hydra-core>=1.3` is now a stated requirement.
- `python -m jem.main --help` prints the `${jcm_data:}` / `${jem_data:}`
  resolver syntax instead of resolving it to the local machine's paths.
- A `grid_file` / `land_fraction_file` given to a component that takes no grid
  is refused rather than silently dropped.
- `VerosComponent.load_carry` restores Veros' process-global `force_overwrite`
  even when the restart read fails.
- The health check is given each chunk **unreduced**, and only the copy that is
  written is thinned or averaged. `output_averages=True` (the shipped
  `coupled_run=long_run`) replaced a chunk with its time mean before the gate
  saw it: xarray's mean skips NaNs and dilutes a finite extreme, and
  `subsample>1` could drop the chunk's last record — which is the record
  `jcm.diagnostics.check_health` judges, so a model that went bad near the end
  of a chunk was reported healthy and checkpointed.
- `subsample` counts coupled steps of the **run**, so the cadence it writes
  survives chunking and resume. The stride was applied to each chunk's records
  from that chunk's first one, so with three-step chunks and `subsample=2` a
  six-step run kept global steps 0, 2, 3 and 5 instead of 0, 2 and 4 — an
  irregular cadence, and more output than was asked for, changing with a
  `chunk` that is meant to be free to choose for memory and restart
  granularity. `jem.output.postprocess` and `postprocess_datasets` now take
  the chunk's `first_step` and its number of coupled `steps` (`run_chunked`
  passes both), keep the step `s` when `s % subsample == 0` counting from the
  start of the run, and keep or drop **all** of the records a step produced —
  so a component the workflow runs `n` times per coupled step is thinned on
  the same cadence as everyone else. A record count that is not a whole
  multiple of the chunk's steps is a `ValueError`, since the step a record
  belongs to is then undefined. A chunk containing no coupled step on the
  stride — which a `subsample` longer than `chunk` gives, and so does the
  short final batch a resume under a different chunk length ends with — writes
  **no file**, and removes a file an earlier pass left at that name, since
  this pass's output for that chunk is nothing; both are reported at INFO, and
  `RunResult.paths` is then one file per component per chunk except for the
  chunks that kept nothing. A thinned run's output directory therefore reads
  back with `xr.open_mfdataset(sorted(paths))` at its default settings, which
  a zero-record file would make fail.
  `postprocess(dataset, subsample=k)` on its own is unchanged: no offset means
  the start of a run.
- A chunk mean written with `subsample` set is labelled with the **chunk's**
  last time rather than with the last record the stride kept, so the series of
  chunk means is one record per chunk, evenly spaced with the chunks, as
  `output_averages` has always promised (with a phase-aware stride the last
  kept record falls at a different point in each chunk — measured label
  spacings of 3, 3, 6, 3 days for `chunk="4 days"`, `subsample=3`). The mean is
  still over the kept records only, so successive means can average different
  numbers of records; the two options remain ones a run normally sets one of.
- A chunk the health gate rejects is no longer checkpointed. The gate now runs
  before the save, so a run stopped by it leaves its single restart directory
  holding the last chunk that *passed*, instead of overwriting it with the
  state that failed — which a resume would have started from, failed on again,
  with the last healthy state already gone. `bail_on_unhealthy=False` still
  checkpoints, because that run is carrying on and has to stay resumable.
- The two shipped Veros configurations state that, with `land=none` and the
  default atmospheric forcing, the atmosphere runs over land at a constant
  288.15 K with zero snow and soil water, and name the overrides that change it.
- **`Exchange.__call__` casts a source value to its destination field's own
  dtype instead of writing it through unchanged.** Importing Veros sets
  `jax_enable_x64` process-wide, so a Veros ocean's carry is float64 while
  parts of the atmosphere's carry stay float32; `ocean=veros`
  (`VEROS_OCEAN_EXCHANGES`) then failed on its first coupled step with
  `lax.scan`'s "carry input and carry output must have equal types ...
  float32[96,48] vs float64[96,48]", naming neither the exchange nor the
  field. A shape mismatch is not touched by this and still fails the same
  way it always did -- only dtype, never shape, is silently reconciled here.

## [Unreleased] — 1.0.0a0, "the core API contract"

Phase 1 of the [API hardening plan][plan]. It replaces the duck-typed component
interface with an explicit protocol, moves the clock out of the components and
into the coupler, makes component parameters differentiable, and settles the
output conventions. **Every entry below is a breaking change unless it says
otherwise**; the code that has to change is named in each one.

[plan]: https://github.com/climate-analytics-lab/jax-esm/blob/claude/jax-esm-api-review-jv7j7u/docs/source/design/api_hardening_plan.md

### Added

- **`jem.base.component`** — the whole contract, in one module:
  - `Component`, a runtime-checkable `typing.Protocol` requiring `name`,
    `initialize() -> carry` and `step(carry, time) -> (carry, diagnostics)`.
    There is still no base class to inherit from, but a component is now
    *checked*: `Coupler.add_component` raises `TypeError` naming the missing
    member.
  - The optional capabilities `SupportsXarray` (`to_xarray(diagnostics,
    time)`), `SupportsCheckpoint` (`save_state`/`load_state`) and
    `SupportsBind` (`bind(*, coupling_timestep, start_date, calendar)`), each
    detected with `isinstance` at the one place that uses it.
  - `CoupledCarry`, the scanned state of the coupled model: `components`
    (one carry per component) plus `step`, the authoritative coupled step
    counter.
  - `CouplingTime`, what every `step` receives: `step`, `sim_time`, and the
    static `dt`, `year_offset_seconds` and `days_per_year`, with a
    `year_fraction` property and `end_of_step()`.
  - `TimeAxis`, the description of a run's output records that every component
    labels its dataset from.
  - `Exchanger`, the type of the functions that move information between
    components.
- `Coupler.generate_step_function()` — the pure one-step function
  `step(carry) -> (carry, diagnostics)`, previously only available inside
  `Coupler.run`. (The name is kept from the old API; the signature is new, see
  *Removed*.)
- `Coupler.coupling_time(step)`, `Coupler.time_axis(first_step, n)` and the
  `coupling_timestep` / `start_date` / `calendar` / `dt_seconds` /
  `year_offset_seconds` / `days_per_year` properties: the clock, readable.
- `Coupler.__repr__`, which names the components, exchangers, workflow and
  clock. It replaces `get_info()` + `tree_tools.print_tree` in the examples.
- `jem.components.jcm` — the JCM adapter as a package:
  - `JCMComponent(model, *, forcing=None)`, name `"atm"`, a wrapper *object*.
  - `jem.components.jcm.exchange_fields`, the single place JCM's
    package-specific diagnostics are translated into JEM's conventions:
    `SurfaceExchange`, the `speedy()` reader, an `echam()` reader that raises
    `NotImplementedError` naming jax-gcm#754, and `detect()`.
- `jem.components.veros_component.VerosComponent(model)`, name `"ocn"`, the
  same treatment for the Veros ocean.
- `jem.components.clock` — `clock_tolerance_seconds(sim_time)` and the two
  constants behind it, the one definition of how far a wrapped model's own
  clock may drift from the coupler's before the wrapper reports it. Both the
  JCM and the Veros wrapper make that comparison, and they must not answer it
  differently.
- A `flax.struct` parameters dataclass per slab model, whose numeric fields are
  pytree leaves and which travels in `carry["params"]`, so `jax.grad` of a
  coupled run with respect to a physical parameter needs no special casing:
  - `SlabOceanParameters`: `relaxation_time=60*86400.0`,
    `mixed_layer_depth_min=40.0`, `mixed_layer_depth_max=60.0`,
    `initial_sst=288.15`, static `forcing_method="none"` and
    `ocean_mask_value=0.0`.
  - `SlabLandParameters`: `depth_soil=1.0`, `depth_lice=5.0`,
    `soil_volumetric_heat_capacity=2.50e6`,
    `land_ice_volumetric_heat_capacity=1.93e6`, `tdland=40*86400.0`,
    `flandmin=1/3`, `land_threshold=0.1`, `snow_depth_to_cover_scale=60.0`,
    `land_ice_albedo_threshold=0.4`, `surface_albedo=0.2`.
  - `SlabSeaiceParameters`: `initial_ice_thickness=0.0`,
    `min_ice_thickness=1e-3`, `ice_fraction_thickness_scale=0.5`, static
    `ocean_mask_value=0.0`.
  - `SlabAtmosphereParameters`: `initial_temperature_base=273.15`,
    `initial_temperature_amplitude=17.0`, `initial_zonal_wind=10.0`,
    `initial_meridional_wind=0.0`.
  Each has a `.default()` classmethod, and each model's `params=` argument
  defaults to it.
- **Parameterized initialization**: `SlabModelBase.initialize(params=None)` and
  `Coupler.initialize(params=None)`. A parameter a component reads only in
  `initialize` — the slab ocean's `initial_sst`, the sea ice's
  `initial_ice_thickness`, every field of `SlabAtmosphereParameters` — has
  already been copied into the state by the time a carry exists, so replacing
  that leaf in `carry["params"]` did nothing and `jax.grad` with respect to it
  was zero: the field was advertised as a differentiable leaf but could not be
  varied at all (constructing the model inside `jax.grad` does not help, since
  the constructor validates these values as Python floats). Passing parameters
  to `initialize` builds the initial state from them *and* carries them, and
  they reach the state untouched, so a gradient with respect to an initial
  condition flows through the trajectory:

  ```python
  jax.grad(lambda p: loss(trajectory(model.initialize({"ice": p}))))
  ```

  `Coupler.initialize(params)` takes `{component name: that component's
  parameters}`, routes each to that component's `initialize(params=…)` and
  initializes the rest as before; a name it has no component for is a
  `ValueError`, and a component whose `initialize` takes no parameters is a
  `TypeError` rather than a silently ignored request. For a nested `Coupler`
  the value is itself a mapping over its components.
  **`initialize()` with no argument is exactly as it was**, in every component
  and in the coupler. Constructor validation of these parameters also stays as
  it was — it runs on the concrete construction-time values, which is why it
  can read them as floats; `initialize(params)` is the differentiable entry
  point and takes traced values.
- `SlabGrid.from_coords(horizontal, fractional_mask=None, threshold=0.5)`,
  which builds a slab grid from the dinosaur horizontal grid the atmosphere is
  discretized on, and `SlabGrid.from_scrip(...)`.
- `jem.constants.SurfaceConstants` and `jem.constants.set_constants(...)`,
  mirroring `jcm.constants`: a frozen dataclass, a live singleton and a module
  `__getattr__`, so `jem.constants.ocean_density` honours an override made
  after import.
- `TimeAxis.datetimes()` and `TimeAxis.attrs`, the single definition of the
  output time labels (JCM's arithmetic, not just JCM's answer); every
  component's `to_xarray` calls them directly for the `(values, attrs)` pair
  xarray wants.
- `jem.base.component.seconds_since_new_year(start_date)` and
  `start_year_fraction(start_date, calendar)`, the shared arithmetic behind
  both `CouplingTime.year_fraction` and the date a slab model samples its
  climatology at in `initialize()`, so the two cannot disagree; and
  `NANOSECONDS_PER_DAY`, the factor JCM's time labels go through.
- `SlabModelBase.bind(...)` (the slab models are `SupportsBind` too) and the
  `SlabModelBase.start_year_fraction` property it sets. `initialize()` samples
  a monthly climatology but receives no clock — the clock lives in the carry,
  which does not exist yet — so the coupler's start date reaches the model the
  same way it reaches JCM and Veros. A model that was never registered with a
  coupler reads 1 January, which is what a bare `model.initialize()` in a test
  or a notebook gets.
- Complete exports: `jem` adds the `Carry` and `Diagnostics` aliases a user
  needs to annotate their own component, and `jem.components` adds the four
  `Slab*Parameters`, `SlabGrid`, `SlabModelBase` and `load_monthly_climatology`
  — a documented run can now be assembled from the packages' own exports.
  `jem` still does not re-export the components: importing the JCM wrapper
  pulls in the whole atmosphere, which should not be the cost of `import jem`.
- `docs/source/design/architecture.md` rewritten against the new API, and a
  `tests/examples` note in `docs/source/developers.rst`.
- `jem.utils.checkpoints.latest_complete_checkpoint(checkpoint_root,
  pattern="step_*")` — the newest checkpoint directory that actually holds a
  completion marker, or `None`. A run killed part-way through a save leaves a
  marker-less directory that sorts newest and that `load_coupled_carry`
  refuses; a driver that resumed from `sorted(root.glob(...))[-1]` could not
  restart at all. The helper skips such directories, warning once per skipped
  one, and both experimental Veros drivers now resume through it.
- `jem.utils.checkpoints.remaining_batches(steps_done, total_steps,
  steps_per_batch)` — the lengths of the batches a chunked run still has to
  integrate, the last one short when the total is not a whole number of
  batches. **The experimental Veros drivers now name a checkpoint directory
  after the coupled step it was written at (`step_00000005`), not after a
  batch index (`batch_00001`), and derive the remaining work from the restored
  `carry.step` rather than from that name.** A batch index means nothing
  across two runs that chose different `--simulation-interval-days`: resuming
  such a run used to overshoot or exit immediately. An existing `batch_*`
  checkpoint directory is no longer found by a resume; rename it to
  `step_<the coupled step it holds>` (zero-padded to eight digits) to keep
  using it.
- **Nested workflows and workflow multiplicity.** `Coupler(workflow=...)` now
  accepts an arbitrarily nested sequence of names — it is flattened at
  construction, and `Coupler.workflow` is still the flat tuple — and a name may
  appear more than once. An element listed *n* times runs *n* times per coupled
  step on a clock `coupling_timestep / n` (which must be a whole number of
  seconds, or construction raises `ValueError` naming the element and the
  count):

  ```python
  workflow=[["atm_lnd_exchange", "atm", "lnd"] * 24, "atm_ocn_exchange", "ocn"]
  ```

  A bindable component is bound with its own sub-timestep, once; a component an
  explicit workflow never names is neither bound nor run. Call *k* of coupled
  step *s* is handed `Coupler.coupling_time_at_substep(s, k, n)`, whose `step`
  is the sub-step `s * n + k` and whose `dt` is the sub-timestep, so
  `year_fraction` stays exact at the faster rate; exchangers may be repeated
  too. The repeated component's diagnostics come back stacked on a new leading
  axis of length *n* — `(steps, n, ...)` from a trajectory — and
  `Coupler.to_xarray` folds those into `steps * n` records labelled at the
  sub-rate, `first_step` still being counted in coupled steps.
  `CoupledCarry.step` still counts coupled steps, so checkpoints and resume are
  unchanged. **Everything about `n == 1` is exactly as it was**, including the
  traced operations, the diagnostics shapes and the time axis.
- `Coupler.multiplicities()` — how many times each element runs per coupled
  step — and `Coupler.time_axis(first_step, n, *, multiplicity=1)`, whose new
  keyword builds the sub-rate output axis.
- **A `Coupler` is a `Component`**, so a coupled model can be a component of a
  slower coupled model with no wrapper class — a fast atmosphere/land loop
  inside a daily ocean coupling:

  ```python
  fast = Coupler({"atm": atm, "lnd": lnd}, {"atm_lnd_exchange": ...},
                 coupling_timestep=jdt.to_timedelta(1, "hour"),
                 start_date=start_date, name="atm_lnd")
  model = Coupler({"atm_lnd": fast, "ocn": ocn}, {"srf_ocn_exchange": ...},
                  coupling_timestep=jdt.to_timedelta(1, "day"),
                  start_date=start_date,
                  workflow=["srf_ocn_exchange", "atm_lnd", "ocn"])
  ```

  - `Coupler.__init__` takes a keyword-only `name="coupled"` (the `Component`
    protocol requires the attribute; the key it is registered under is still
    what a workflow names).
  - `Coupler.bind(*, coupling_timestep, start_date, calendar)` requires the
    outer timestep to be a whole multiple of this coupler's own and the start
    date and calendar to be equal, and records the ratio *r*
    (`Coupler.outer_ratio`). Rebinding to the same clock is a no-op, to a
    different one a `ValueError`.
  - `Coupler.step(carry, time)` runs *r* of this coupler's own coupled steps,
    driven by the inner carry's own step counter, and returns its usual
    per-component diagnostics stacked on a leading axis of length *r* (no extra
    axis for `r == 1`). Calling it before `bind` is a `RuntimeError`.
  - `jem.nested_carry(carries, outer_name, inner_name)` and
    `jem.with_nested_carry(carries, outer_name, inner_name, new_inner_carry)`
    — how an exchanger in the outer coupler reads and immutably replaces a
    component inside a nested one.
  - `Coupler.save_state(carry, directory)` / `Coupler.load_state(directory)`
    checkpoint the coupled model, so a `Coupler` implements
    `SupportsCheckpoint` as well. The savers and loaders are derived from the
    components — `{name: component.save_state for … if isinstance(component,
    SupportsCheckpoint)}` — so a driver no longer builds them by hand:

    ```python
    model.save_state(final_carry, checkpoint_dir / f"step_{int(final_carry.step):08d}")
    carry = model.load_state(saved)
    ```

    Because a `Coupler` is itself such a component this recurses: a nested
    coupled model is written into `directory / <its registered name>`, with its
    own components and its own `coupled_step.pkl`, and read back the same way,
    so a resume continues both clocks. Previously the outer save pickled the
    inner `CoupledCarry` as a plain pytree, which bypassed the HDF5 restart
    path a component like `VerosComponent` requires, and no explicit
    `component_savers` mapping could reach into it without a bespoke recursive
    saver. `save_coupled_carry` / `load_coupled_carry` still take
    `component_savers` / `component_loaders` explicitly, for a caller
    overriding one or supplying a saver that is not a component capability.

  Writing the same model as one coupler with a repeated workflow (above) gives
  bit-identical carries and datasets; the design doc says which to prefer when.
- **`SupportsXarray.to_xarray` may return a mapping of datasets**, not only one
  dataset: a component that is itself a coupled model has one per *its*
  components. `Coupler.to_xarray` flattens such a mapping into its result under
  those names — the nested coupler's own registered name does not appear in the
  output — and raises `ValueError` on a name collision. It also accepts the
  component-protocol call `to_xarray(diagnostics, time)` alongside the existing
  `to_xarray(diagnostics, *, first_step=0)`; in the first form the inner
  datasets are labelled on the inner, faster axis, starting at
  `time.steps[0] * r`.

### Changed

- **"Mapper" is now "exchanger".** `Coupler(components, exchangers=...)`,
  `add_exchanger`, `remove_exchanger`, `Coupler.exchangers`. The rename was
  decided in the #108 review: "mapper" reads as a regridding operation,
  whereas one of these functions may regrid, compute a flux, convert units or
  simply copy a field.
- **An exchanger's signature is `(components, time) -> components`**, not
  `coupled_carry -> coupled_carry`. It receives the mapping of component
  carries (a fresh dict) plus the clock, and must build new carries rather
  than assign into the ones it was handed — the coupler hands out the carries
  of a `lax.scan`. The coupler now compares the pytree structure of the
  carries after every workflow element and raises `RuntimeError` naming the
  element that changed it.
- **`Coupler.__init__` takes the coupled model's clock**:
  `Coupler(components, exchangers=None, *, coupling_timestep, start_date,
  calendar="365_day", workflow=None)`. `coupling_timestep` and `start_date`
  are required. `workflow` moved here from `run()`/`generate_*` and defaults
  to every exchanger (in insertion order) followed by every component, so the
  usual coupling scheme need not be spelled out. **The "each name at most once"
  restriction is gone**: a repeated name is now a multiplicity (see *Added*),
  not a `ValueError`.
- **`Coupler.initialize()` returns a `CoupledCarry`**, not a plain
  `dict[str, carry]`. The per-component carries are under
  `.components`; rebuild one with `carry.replace(components=...)`.
- **`VerosComponent.to_xarray` prefixes the fields the ocean was forced with**,
  as the slab models already did: `heat_flux`, `freshwater_flux`,
  `surface_taux`, `surface_tauy` and `surface_air_temperature` are written as
  `forcing_*`. Without the prefix a Veros dataset could not be merged with the
  dataset of the component that produced those fields. Anything reading them
  out of a saved run (`ds_ocn["heat_flux"]`) reads `ds_ocn["forcing_heat_flux"]`
  instead.
- **`forcing_variable` and `FORCING_VARIABLE_PREFIX` moved to
  `jem.base.component`**, since the convention belongs to the output contract
  every component follows rather than to the slab family. They are still
  importable from `jem.components.slab.base`.
- **`jem.utils.checkpoints.save_coupled_carry`/`load_coupled_carry` take and
  return a `CoupledCarry`**, not a plain `dict[str, carry]`, and the
  checkpoint directory gains one file, `coupled_step.pkl`, holding the coupled
  step counter. A caller that passed `final_carry.components` passes
  `final_carry`; a caller that rebuilt the carry around the loaded dict uses
  the returned `CoupledCarry` directly. The mapping-only helpers are still
  available as `save_component_carries` / `load_component_carries`, which is
  what the Veros restart writer uses for the picklable half of its carry.
- **`Coupler.generate_trajectory_function(iterations, *, remat=False,
  jit=True)`** takes neither `workflow` nor `checkpoint`/`show_progress`/
  `tqdm_kwargs`, and returns `carry -> (final_carry, diagnostics)`.
  Because the step counter lives in the carry rather than in the `lax.scan`
  index, calling it again on the carry it returned *continues* the run; this
  is what makes a chunked or restarted run keep the right date and season.
- **`coupler.components[name]` is the component object itself.** There is no
  `JEMComponent` wrapper and no `.raw_component` — code that reached through
  the wrapper now uses the object directly (`component.model` for
  `JCMComponent` / `VerosComponent`).
- **Components implement `step(carry, time)`**, not
  `generate_step_function() -> step(carry, step_index)`, and **`to_xarray(
  diagnostics, time)`**, not `predictions_to_xarray(predictions)`.
  `step` receives a `CouplingTime`, not a bare index or a float.
- **Slab constructors lost the clock and gained parameters.** They are now
  - `SlabOceanModel(grid, params=SlabOceanParameters(), *, name="ocn",
    sst_clim_file=None, q_flux_file=None)`
  - `SlabLandModel(grid, params=SlabLandParameters(), *, name="lnd",
    land_clim_file=None, surface_albedo=None)`
  - `SlabSeaiceModel(grid, params=SlabSeaiceParameters(), *, name="ice")`
  - `SlabAtmosphereModel(grid, params=SlabAtmosphereParameters(), *,
    name="atm")`

  `start_datetime`, `timestep` and `calendar` are gone from all four (the
  coupler owns them); every physical tunable moved into the parameters object;
  `SST_clim_file`/`Q_flux_file` are now `sst_clim_file`/`q_flux_file`;
  `mask_value` is now `params.ocean_mask_value`;
  `initialization_sea_surface_temperature` is now `params.initial_sst` and
  `initialization_ice_thickness` is `params.initial_ice_thickness`. The
  component's name is a constructor argument rather than the class name.
- **`sim_time` is gone from every state struct** (`OceanState`,
  `LandState`, `SeaiceState`, `AtmosphereState`): a component holds no clock.
  Every slab carry gains a `"params"` entry alongside `state`/`forcing`/
  `derived`.
- **`OceanState` holds only `sea_surface_temperature`.** The mixed-layer
  depth is a prescribed profile of `mixed_layer_depth_min`/`_max`, recomputed
  from the carried parameters every step (so they are live, differentiable
  tunables) and written to the output as `mixed_layer_depth` from
  `OceanDerived`.
- **`forcing_method` values are lowercase** `"none" | "qflux" | "relaxation"`
  (`jem.components.slab.slab_ocean_model.params.FORCING_METHODS`). `None` and
  `"None"` are no longer accepted; an unknown value raises `ValueError`.
- **`SlabOceanModel` refuses `forcing_method="relaxation"` without an
  `sst_clim_file`.** The previous behaviour — silently setting the relaxation
  timescale to infinity and then dereferencing a climatology that was never
  loaded — could not work.
- **Science-visible default change**: without an SST climatology the ocean's
  idealized initial profile is now built on `params.initial_sst = 288.15 K`,
  giving 288–298 K instead of the previous 273–283 K. The constructor has
  always accepted this value but never used it — the base of the profile was
  hard-wired to the freezing point — so wiring it up also moves the default
  aquaplanet start to a sensible one.
- **`SlabLandModel` loads its climatologies by name and requires exactly 12
  monthly records.** The branch that accepted daily data is gone; a file with
  a different record count raises `ValueError` naming the file and the check.
  Fields the file does not carry still fall back to idealized ones.
- **The JCM carry gained `"physics"`**: JCM's cross-step physics carry
  (sub-cycled radiation, prior-step TKE, term-to-term tendencies) is threaded
  through `run_from_state_with_carry` instead of being dropped and rebuilt
  once per coupling interval, which was a silent, systematic error in every
  coupled run. It holds integer and boolean leaves, so it must never be cast
  wholesale to a float dtype.
- **`JCMDerived` gained `evaporation`, `precipitation`, `u0` and `v0`**
  alongside `total_heat_flux`, `total_freshwater_flux` and the opaque
  `physics` passthrough, so an exchanger no longer has to dig the surface wind
  and water fluxes out of the physics dict itself.
- **`JCMComponent.initialize()` does not integrate.** The previous adapter ran
  a whole throwaway coupling step to learn the structure of the diagnostics it
  would later store, which cost a step per run and started the atmosphere one
  coupling interval ahead of the coupler's clock.
- **Output conventions**, so that `xr.merge` of two components' datasets from
  one run aligns instead of producing an outer join:
  - slab datasets use dims `("time", "lon", "lat")` with 1-D degree
    coordinates carrying JCM's own names and values (a curvilinear grid keeps
    dims `("time", "x", "y")` with 2-D auxiliary `lat`/`lon` and a CF
    `coordinates` attribute), replacing the old `latitude2D`/`longitude2D`
    auxiliary coordinates;
  - the `time` coordinate is an absolute `datetime64[ns]` axis, not
    "hours since <start>";
  - a record is labelled with the **end** of the interval it covers, which is
    JCM's convention;
  - state and derived variables keep their plain names and every variable that
    came from a component's *forcing* is written with a `forcing_` prefix
    (`jem.components.slab.base.FORCING_VARIABLE_PREFIX` and the
    `forcing_variable(name)` helper that applies it). Two components
    legitimately hold the same physical field — one produced it, the other
    received it — and without the prefix `xr.merge` of their datasets collides
    on the shared name. The renames are: `SlabAtmosphereModel` and
    `SlabLandModel` write `forcing_total_heat_flux` (was `total_heat_flux`),
    `SlabOceanModel` in Q-flux mode writes `forcing_q_flux` (was `q_flux`), and
    `SlabSeaiceModel` writes `forcing_ice_frazil_melt_energy` (was
    `ice_frazil_melt_energy`). Everything else keeps its name, including the
    ocean's own `total_heat_flux` (a derived quantity — the effective heat flux
    applied to the mixed layer) and `ice_frazil_melt_energy`.
- `Coupler.to_xarray(diagnostics, *, first_step=0)` replaces
  `predictions_to_xarray(predictions)`. Pass `first_step` — the `step` of the
  carry the chunk started from — when writing a chunked run, or every chunk is
  labelled with the first chunk's dates.
- `jem.constants` now holds only what `jcm.constants` does not define, in a
  `SurfaceConstants` singleton. The duplicated values were removed and their
  JCM counterparts now apply, which changes three numbers:
  latent heat of fusion `3.34e5 → 3.33e5 J/kg` (`c.alhf`, which JCM *derives*
  as `alhs - alhc`), dry-air specific heat `1004.0 → 1004.64 J/K/kg`
  (`c.cpd`, the ECHAM-6.3 value JCM's own `rd = akap*cpd` is built on) and the
  solar constant `1367 → 1361 W/m2` (`c.solc`, read by nothing in `jem`).
- `JCMComponent`'s clock-drift check scales with float32 resolution:
  `clock_tolerance_seconds(sim_time)` is one second or eight float32 ulps of
  the elapsed time, whichever is larger, so it neither fires on the rounding of
  a long run's clock nor stops noticing a real disagreement. That tolerance now
  lives in `jem.components.clock`, shared with the Veros wrapper, rather than
  in the JCM one; its behaviour is unchanged.
- `Coupler` logs at DEBUG instead of printing; nothing under `jem/` prints any
  more (the Veros checkpoint writer and the forcing/topography generator were
  the last two).

### Deprecated

- `jem.components.jcm_component.make_jem_compatible(model, coupling_timestep)`
  and `jem.components.veros_component.make_jem_compatible(model,
  coupling_timestep)` now return a `JCMComponent` / `VerosComponent` and warn.
  The `coupling_timestep` argument is ignored — the coupler supplies it, with
  the start date and calendar, through `bind()`. Unlike the old functions they
  do **not** attach methods to the wrapped model, so anything that called
  `model.initialize()` / `model.generate_step_function()` must call them on
  the returned component instead.

### Removed

- **`jem.base.interface`** (whole module): `resolve_interface`, the
  `__JEM_CUSTOMIZED_MAPPING__` method-remapping hook, `MethodNotFoundError`,
  `MemberNotFoundError`, `MemberTypeNotMatchError` and
  `NumberOfMethodParametersNotMatchError`. A component is now checked against
  the `Component` protocol; an object whose methods have other names is
  adapted by a wrapper class, which is what `JCMComponent` and
  `VerosComponent` are.
- **`jem.base.typing`** (whole module): `JEMComponent`, `MapperFunction`,
  `StepFunction`, `StepFunctionGenerator`, `TrajectoryFunction`,
  `PredictionsToXarrayFunction`, `GetInfoFunction`, `InitializeFunction`,
  `ComponentCarry`, `CoupledCarry`, `SimulationTime`, `Predictions`,
  `Workflow`, `ComponentName`, `Pytree`. The aliases that survive
  (`Carry`, `Diagnostics`, `CoupledCarry`, `Exchanger`) live in
  `jem.base.component`, and `CoupledCarry` is now a struct, not a dict alias.
- `Coupler.run(...)` — build the function and call it:
  `run = coupler.generate_trajectory_function(iterations)`, then
  `final_carry, diagnostics = run(coupler.initialize())`. It returned
  `(initial_carry, final_carry, predictions)`; the initial carry is now
  whatever you passed in.
- `Coupler.generate_step_function()` in its old form (it took a workflow and
  `jitted`/`show_progress` flags and returned a function of `(carry, step)`);
  the method now takes no arguments and returns `step(carry)`, the workflow
  and clock being the coupler's own.
- `Coupler.predictions_to_xarray(...)` → `Coupler.to_xarray(...)`.
- `Coupler.get_info()` → `repr(coupler)`.
- `Coupler.add_mapper` / `remove_mapper` / `.mappers` → `add_exchanger` /
  `remove_exchanger` / `.exchangers`.
- `jem.base.coupler.adhoc_scan` and the `jitted=False` debug path: use
  `generate_trajectory_function(..., jit=False)`, or `jax.disable_jit()`.
- The `verbose`, `show_progress`, `tqdm_kwargs`, `checkpoint` and
  `reuse_last_available_trajectory` parameters, and the `trajectory_holder`
  attribute that backed the last of them. A trajectory function is now built
  once by the caller and reused by calling it again; `remat=` replaces
  `checkpoint=`.
- Component-side `get_info()` and `predictions_to_xarray()` hooks (the
  coupler no longer looks for either name).
- `generate_slab_grid`, the `"JCM::T31"` specification-string DSL it parsed,
  `load_jcm_fractional_mask`, and `generate_slab_grid_from_scrip` →
  `SlabGrid.from_coords` and `SlabGrid.from_scrip`. The atmosphere's own grid
  object is the single source of truth for the grid a coupled run uses, and
  the land fraction comes from `jcm.terrain.TerrainData.fmask`, which is
  already on `SlabGrid`'s `(n_lon, n_lat)` layout.
- `jem.utils.tree_tools` (whole module): `print_tree`, `print_dict_tree`,
  `tree_to_dict`. It existed to render `Coupler.get_info()`; `repr(coupler)`
  replaced both.
- The `jem.constants` values `jcm.constants` already owns — `g0`,
  `stephan_boltzmann_const`, `freezing_point_K`, `ice_melting_point_K`,
  `ice_density`, `ice_latent_heat_fusion`,
  `atmosphere_specific_heat_capacity_at_constant_pressure` and `solar_const`
  — use `c.grav`, `c.sbc`, `c.tmelt`, `c.rhoi`, `c.alhf`, `c.cpd`, `c.solc`.
  `default_mld_min`/`default_mld_max` and
  `default_land_depth_min`/`default_land_depth_max` went with no replacement:
  a default for a component parameter belongs to that component's parameters
  dataclass.
- `jem.utils.bulk_op` (whole module): `stack_objects` and
  `unwrap_leading_dims`. They existed because the old coupler stacked each
  step's predictions by hand; `lax.scan` does it, so nothing called them.
- `jem.utils.time.TIME_ATTRS` → `TimeAxis.attrs`.
- `jem.utils.time` (whole module): its one function, `time_coordinate`, only
  returned `(time.datetimes(), dict(time.attrs))`, and `TimeAxis.attrs`
  already builds a fresh dict per access. The two `to_xarray` implementations
  call the `TimeAxis` directly.
- The dependency `typeguard`; nothing imports it now that `jem.base.typing`
  is gone.

### Fixed

- The seasonal cycle no longer restarts at the beginning of each chunk of a
  chunked run: the coupled step counter is part of the carry, so
  `year_fraction` continues across trajectory calls and checkpoint restarts.
  It is also computed in exact integer arithmetic modulo the steps in a year
  whenever the coupling step divides the year, so a float32 `sim_time` cannot
  quantise the annual cycle away in a century-long run.
- Two components can no longer disagree about the date: there is one clock,
  and `bind()` refuses a coupling timestep that is not a whole multiple of a
  component's internal timestep, or (for JCM) a start date or calendar that
  differs from the coupler's. `JCMComponent.step` additionally reports at
  ERROR if the dycore state's own `sim_time` has drifted from the coupler's,
  which can only happen if the carry came from a different run.
- `VerosComponent.step` makes that same check, which it previously did not: it
  advanced from Veros' own `variables.time` and ignored the coupler's clock
  entirely, so a setup that had already been integrated, or a Veros restart
  state paired with a `CoupledCarry.step` from elsewhere in the run, put the
  ocean at a different simulated date from every other component — silently,
  because the output axis is labelled from the coupler's clock. Veros has no
  calendar of its own, so `bind` records the `variables.time` the setup holds
  when the coupler adopts it as the reading that corresponds to the coupler's
  `start_date`, and `step` compares the difference with `time.sim_time`
  against `jem.components.clock.clock_tolerance_seconds`. As in the JCM
  wrapper it is reported at ERROR through `jax.debug.callback`, never raised:
  the check runs inside the coupled `lax.scan`.
- `Coupler.generate_step_function()` snapshots the components and exchangers when it is
  called, so registering a component afterwards cannot silently change an
  already-compiled step.
- The coupled step never mutates its input carry; it rebuilds the carries dict
  and returns a new `CoupledCarry`.
- `VerosComponent.to_xarray` writes the run's `time` coordinate. It was handed
  a `TimeAxis`, checked the record count against it and then dropped it, so an
  ocean dataset came out with a bare 0..n-1 integer `time` index: it could not
  be merged with any other component's output, and the dates of a chunked run
  were absent from the files entirely. It now labels its records with
  `TimeAxis.datetimes()` and `TimeAxis.attrs`, exactly as the slab models do.
- A run resumed from a checkpoint no longer restarts its seasonal cycle:
  `save_coupled_carry` writes the coupled step counter and `load_coupled_carry`
  restores it. A checkpoint written without one is refused with a `ValueError`
  naming the missing file, rather than resuming at step 0 or having its step
  guessed from a batch index — a guess that is only right while every batch has
  the same length.
- Every scalar parameter a component checks at construction must now be
  **finite** as well as in range. A bare `> 0` test admits `+inf`, and an
  infinite parameter is the quiet failure: `snow_depth_to_cover_scale = inf`
  reports no snow cover however deep the snow, `relaxation_time = inf` relaxes
  to nothing while the run still calls itself a relaxation run, and
  `tdland = inf` damps the land temperature by `inf / (1 + inf)` — NaN. The
  centre coordinates of a SCRIP grid file are now checked the same way and for
  the same reason: a NaN compares False against both ends of a range test, so
  it used to pass straight into the run's output coordinates.
- `SlabAtmosphereModel` now validates its four initial-condition parameters at
  construction: `initial_temperature_base` must be a finite positive
  temperature in kelvin, and `initial_temperature_amplitude`,
  `initial_zonal_wind` and `initial_meridional_wind` must be finite (they are
  signed, so nothing more is required of them). They are copied straight into
  the initial state, so a non-finite one used to be accepted and make the whole
  trajectory non-finite — quietly, in the case of an infinite wind, whose
  infinite bulk conductance leaves the column heat budget evaluating
  `inf + -inf`.
- `SlabOceanModel` now validates `initial_sst` at construction as a finite,
  strictly positive temperature in kelvin, beside the mixed-layer-depth checks.
  With no SST climatology it is the base of the idealized initial profile, so
  it fills every ocean cell of the initial state and every later SST inherits
  it; nothing downstream rejected a non-finite or non-physical value.
- `SlabOceanModel` writes `forcing_q_flux` when the **run** applied a Q-flux,
  not when the model object was constructed with one. `step` follows the
  `forcing_method` in `carry["params"]`, so a run started with
  `initialize(params)` (or `Coupler.initialize({"ocn": params})`) carrying a
  different method used to have an applied Q-flux dropped from its output, or
  a constant zero published as though a Q-flux were active. The step now
  publishes the snapshot it applied as its own diagnostics key and the output
  follows that key. Consequently `OceanDerived` no longer has a
  `q_flux_snapshot` field: the snapshot is per-step output that nothing reads
  back, and a `tree_math.struct` field would exist in every configuration,
  which is what forced the unconditional write in the first place.

### Known gaps

- The land model's ice-sheet branch is never reached in practice: nothing wires
  a real surface albedo into `SlabLandModel`, so `params.surface_albedo = 0.2`
  applies everywhere and every land cell is soil. Tracked as jax-esm#109 and
  cross-referenced from the model's docstring.
- The ECHAM surface exchange is not implemented: `exchange_fields.echam()`
  raises `NotImplementedError` naming jax-gcm#754. Coupled runs need SPEEDY
  physics until that lands.
- `coupling_timestep` cannot be shorter than one second: it is a
  `jax_datetime.Timedelta`, which holds whole seconds, even though only
  `Coupler.dt_seconds` is used downstream. Immaterial for a geoscience run; the
  reason the spring example in `examples/03_non_geoscience/` is rescaled from
  `dt = 0.01 s` to `dt = 1 s`. Tracked as jax-esm#110 and cross-referenced from
  `Coupler`'s docstring and the notebook.

## [0.2.0] - release blockers and repository health

Phase 0: what `jem.__version__` reports today. Not yet tagged.

### Added

- The monthly-climatology loader written for `SlabOceanModel` is public on the
  slab base module as `jem.components.slab.base.load_monthly_climatology`, so
  the other slab components can read their boundary conditions through the
  same name-based, grid-checked path.
- `CHANGELOG.md` (this file). Every pre-1.0 API removal or rename is recorded
  here rather than only in a commit message.

### Changed

- The version is single-sourced from `jem.__version__` (setuptools reads it
  via `[tool.setuptools.dynamic]`), so the package attribute and the installed
  distribution metadata can no longer disagree. Bumped to `0.2.0`.

- `pyproject.toml` now declares only the packages `jem` actually imports at
  runtime. Plotting (`matplotlib`, `cartopy`), documentation (`sphinx`,
  `shibuya`, `nbsphinx`, `myst-parser`) and the Veros ocean moved to the
  `plot`, `docs` and `veros` extras, so a model-running install no longer
  pulls a plotting or documentation stack.
- The `jcm` requirement is now `jcm>=2.1.0b0`, the first jax-gcm release with
  the 2-D surface-flux layout the JCM adapter is being rewritten against.
  **That release is not on PyPI yet**: the floor is currently satisfied only
  by an editable install of a jax-gcm `dev` checkout, which is what CI does
  (`pip install -e ./jax-gcm` before `pip install -e ".[dev]"`).
- CI (`.github/workflows/tests.yml`) now lints the *whole* repository with the
  pinned `ruff==0.15.17` (it previously ran an unpinned ruff over `jem/`
  only), runs `mypy jem/ --ignore-missing-imports` in the same job, checks out
  jax-gcm at `dev` and installs it editable *before* `pip install -e ".[dev]"`
  so `jcm>=2.1.0b0` resolves, and drops the jax-gcm `PYTHONPATH` export.
  The test matrix is `ubuntu-latest` x Python 3.11/3.12 — macOS and 3.13 were
  dropped (**maintainer decision, defaulted**: macOS added run time for a
  pure-Python package with no platform-specific code, and 3.13 is ahead of
  what the JAX/Veros stack is tested against). Notebook and `run.sh` example
  tests moved to their own job that runs on pull requests only, and the
  Codecov upload condition moved from Python 3.10 (which the matrix never
  contained) to 3.11.
- The `veros` extra is deliberately empty: the coupler needs a *jittable* fork
  of Veros that is not published on PyPI and must be installed from git (see
  the README).

### Fixed

- `jem.components.veros_component` now selects the JAX backend through a
  public `configure_veros_runtime()` (still run at import time, because the
  jittable Veros fork locks `runtime_settings` as soon as `veros.core` is
  imported) and raises a clear `RuntimeError` naming the import-order fix
  when Veros was already bound to another backend, instead of failing later
  inside `make_jem_compatible`. It no longer prints while doing so.
- `jem.utils.cycles.evaluate_periodic` mis-weighted points in the wrap-around
  interval (between the last and the first tick) when `x` lay above the last
  tick: the distance from the left tick was computed as `x + 1 - t0` there,
  over-counting by a full cycle. It now uses `(x - t0) mod 1`. The function
  also accepts traced indices, so `vmap_evaluate_periodic` works.
- `jem.components.slab.slab_ocean_model.__all__` listed `OceanForcing` and
  `OceanState` without importing them, so `from ... import *` raised
  `AttributeError`. Both are now imported.

### Removed

- Unused code, none of which had a caller anywhere in `jem`, `tests`,
  `examples` or `docs`:
  - `jem.utils.datetime_tools` (whole module).
  - `jem.utils.domain_grid_tools` (whole module).
  - `jem.utils.bulk_op.concat_objects` and `mean_leaf`.
  - `jem.utils.esmf_regrid.create_regridder_pair`,
    `create_regridder_from_xarray`, `example_usage` and the module's
    `__main__` demo block.
  - `jem.components.slab.grid.generate_slab_grid_from_ugrid` and its private
    helpers `_reshape_ugrid_face_field` / `_load_ugrid_fractional_mask`. The
    SCRIP reader (`generate_slab_grid_from_scrip`) is unaffected.
- Dependencies `dataclasses-json`, `typing-extensions`, `jax_tqdm` and `tqdm`:
  none of them are imported by `jem` any more. `coordax` is no longer a
  runtime dependency either: its only user, `jem.utils.cycles.evaluate_periodic`
  (kept at the maintainers' request although currently unused), imports it
  lazily, so it is needed only when that function is called (it is in the
  `dev` extra so the function stays tested).
- **Progress bars.** `Coupler.run`, `Coupler.generate_trajectory_function` and
  `Coupler.generate_step_function` no longer display a `tqdm` progress bar.
  Their `show_progress` and `tqdm_kwargs` parameters are still accepted so
  existing callers keep working, but are deprecated and ignored; they will be
  dropped in a later release once per-chunk logging replaces them.
- The dead, string-quoted `sys.modules` stub-injection block at the top of
  `tests/unit/test_coupler.py` (~70 lines). The tests have always run against
  the real dependencies.
- `[tool.black]` and `[tool.isort]` configuration: `ruff` (pinned to 0.15.17,
  matching jax-gcm) is the only linter.
