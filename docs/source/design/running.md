# Running a model

A `Coupler` produces functions, not runs. `jem.driver.run_chunked` is the one
loop that turns one into a run, and **every run default lives on its
signature** — the config group `coupled_run` (see {doc}`configuration`) names
the same keys and repeats none of the values.

```python
from jem import run_chunked

result = run_chunked(
    coupler,
    total_time="2190 days",       # a whole number of chunks
    chunk="30 days",
    output_dir="output",
    output_averages=True,         # one record per chunk: its 30-day-window mean
    checkpoint_path="checkpoint",
)
```

Per chunk it integrates, labels and writes the output, checkpoints, and
checks the state is still healthy:

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
(`int(final_carry.step)` — the run's position on the clock, including
whatever a checkpoint restored, not the number of steps this call
integrated), `completed`, one `report` per chunk, every `path` written, and
the `accumulator` if the run was given a reduction.

## Chunking

`total_time` and `chunk` are fixed durations — strings such as `"30 days"`
or numbers of days — parsed by `jcm.date.parse_duration_seconds`, which
rejects a calendar unit such as `"year"` or `"month"` as not fixed. Both must be whole
multiples of the coupling timestep — a coupled step is the smallest thing
the loop can integrate — and `total_time` must be a whole multiple of
`chunk`. All three are checked before anything is built or compiled, and
each message names both quantities. A final partial chunk is refused rather
than accommodated: it would need a second compiled trajectory for one call,
and a run length that does not divide into chunks is far more often a
mistake in the configuration than a request.

## The health gate

`default_health_check` runs `jcm.diagnostics.check_health` on
`datasets["atm"]`, so a coupled run stops on the same evidence an uncoupled
atmosphere does, and `bail_on_unhealthy` (the default) stops at the first
chunk it rejects rather than spending a queue slot integrating a broken
state. The output written so far is kept and `RunResult.completed` is
False. A coupled model with **no** atmosphere gets `{"skipped": "no
atmosphere"}` — an abstention, not a pass: a gate for the surface
components would have to know each one's physical ranges, which is the
components' business. `health_check=None` removes the gate entirely, and
`bail_on_unhealthy=False` logs and carries on, which is what a run studying
the instability itself wants.

The gate is given the chunk **as it was integrated** — every record — and
not the thinned or averaged datasets that were written. That distinction is
the difference between a working gate and one that cannot see:
`check_health` judges a chunk by its last record and by extremes, while
`output_averages=True` replaces the chunk with a mean that (xarray skips
NaNs) drops a NaN entirely and dilutes a finite extreme, and `subsample=n`
need not keep the last record at all. An atmosphere that blew up in the
last hours of a month would then be reported healthy and checkpointed. So
the loop labels the chunk once with `chunk_datasets`, hands *that* to the
gate, and applies `postprocess_datasets` only to the copy it writes. What
the gate can resolve is one **coupling step**: `JCMComponent` integrates
each coupling step with JCM's own `output_averages`, so the records being
judged are already step means — a NaN propagates through that mean, a
finite excursion shorter than a coupling step need not.

The gate also runs **before** the checkpoint, and a chunk it rejects is not
checkpointed (unless `bail_on_unhealthy=False`, where the run carries on and
so must stay resumable): there is only one checkpoint directory, overwritten
in place, so saving a rejected state would replace the last healthy restart
point with a broken one. Bailing instead leaves the restart point at the
last chunk that passed, so the run resumes by repeating the chunk that
failed — which is why that chunk's output files are overwritten on the
resume, with the warning `write_chunk` logs.

## Checkpoints and resume

**The clock is part of the checkpoint because it is part of the state.**
`CoupledCarry.time` — the carried `jax_datetime.Datetime` — and
`CoupledCarry.step` are saved beside every component's own carry; a
directory whose `carry.msgpack` does not hold them is refused rather than
resumed from a guess, since the run's position in the seasonal cycle would
otherwise be unrecoverable.

The format is jax-gcm's: `jem.checkpoint.save(carry, path)` flattens any
pytree to a list of typed arrays serialised with flax's **MessagePack**
codec (`flax.serialization.msgpack_serialize`, hence `carry.msgpack`); the
tree itself is never stored (which keeps the format small), but a manifest
of each leaf's path, shape and dtype is, plus the repr of the whole
`PyTreeDef`. `load(template, path)` rebuilds the tree from a *template*'s
treedef and checks every leaf against the manifest, so a checkpoint written
by a different grid or component composition fails naming the leaf, rather
than deserialising into something that only explodes later inside a
`lax.scan` — including the cases no single leaf can show by itself, such as
a component whose carry holds no arrays at all (renaming it would otherwise
load cleanly and resume a different composition). A **static**
(`pytree_node=False`) parameter that changed value is refused the same way:
JAX keeps it inside the `PyTreeDef`, so resuming with `forcing_method`
edited from `"none"` to `"qflux"` would otherwise silently run a different
model. A *differentiable* parameter is a leaf, so it is restored from the
checkpoint instead and a value edited between runs is overridden, not
refused.

**Loading is all-or-nothing**: the template built from `initialize()`
supplies only the pytree *structure*, every leaf of the restored carry comes
from the checkpoint, and a component the checkpoint does not hold is a
`ValueError` rather than being quietly left at its initial state — which
would continue one part of the model from the saved step while another
restarted at the start date, a run that is neither a resume nor a cold
start.

**The coupler is what a driver checkpoints through**, in one call each way:

```python
model.save_carry(final_carry, checkpoint_dir / f"step_{int(final_carry.step):08d}")
carry = model.load_carry(saved)
```

Which components need writing by hand rather than pickling is a property of
the *components* — `VerosComponent` has to go through Veros' HDF5 restart
writer because a `VerosState` is not a pytree — and the coupler is the one
object that knows them all, deriving the savers/loaders itself from
`isinstance(component, SupportsCheckpoint)`. A `Coupler` implements
`SupportsCheckpoint` itself, so this **recurses**: an outer coupler sees a
nested coupler as a component that writes itself, hands it `directory /
<its registered name>`, and the inner coupler writes its own components and
its own `carry.msgpack` there.

**The carry file is the marker**, written last by renaming a flushed,
fsynced temporary over its final name, and removed first when an existing
checkpoint is overwritten, so a failure part-way through cannot leave a
stale clock beside freshly written component data. A save interrupted
part-way through therefore leaves a directory with no marker, which
`load_coupled_carry` refuses and `latest_complete_checkpoint(root,
pattern="step_*")` skips over (with a warning) in favour of the newest
complete one.

`run_chunked` keeps a **single checkpoint directory**, rewritten after every
chunk the health gate accepts, rather than a directory of dated restart
points — that is what makes resuming a run the same command as starting it:
point at the path, and the run either starts from scratch or continues from
where it stopped. The cost is that only the newest state survives; a run
that wants a history of restart points keeps its own directory of them and
hands each one in as `initial_carry`. Checkpointing is **on by default**
(`checkpoint_path="checkpoint"`, resolved against `output_dir` when
relative — Hydra gives every run a fresh output directory, so resuming
means pointing a second run at the first's output directory).
`checkpoint_path=null` turns it off.

`checkpoint_interval` (`None` by default: after every chunk) saves less
often than every chunk, for a run whose chunks are short for one of the
*other* reasons a chunk exists — a health check every few days, an output
file per day. It must be a whole multiple of `chunk`, and is counted in
coupled steps from the **start of the run** rather than of the call, so a
run stopped and resumed checkpoints at the same points an uninterrupted one
does. The last chunk of a **completed** run is always checkpointed whatever
the interval says, and a run the health gate **stops** writes the last chunk
that *passed*, so what the interval risks is only a run that is *killed*
outright: that falls back to the last interval boundary and re-integrates
(and rewrites the output files of) the chunks after it on resume — safe
because a file is named after the coupled step its chunk starts at, so the
second pass writes the same names from the same starting state, provided
the resume keeps the same `chunk` (checked before compiling anything, with
a `ValueError` naming the files if a changed `chunk` would overwrite records
another chunk's file already holds).

`run_chunked` logs the provenance of the carry it is about to integrate, at
INFO, in one line before anything is compiled — "Starting from
`coupler.initialize()`", "Starting from the `initial_carry` argument", or
"Resumed from checkpoint ... at coupled step N" — because resuming a run is
the *same command* as starting one, so nothing else says which happened. A
directory an interrupted save left without its carry file is a WARNING (a
run died and its last chunk is gone); a path with nothing at it is INFO
(what every first run into a fresh output directory sees).

## Output files and reductions

**Output files.** One file per component per chunk,
`<output_dir>/<component>-<first step:08d>.nc`, named after the coupled
step the chunk starts at — unique however the run was chunked, since a
chunk *index* would collide across two runs chunked differently.
`write_chunk` warns when it overwrites an existing file (normally a rerun
into the same directory, or a resume that repeats a chunk).

`subsample=n` keeps every *n*-th coupling step of the **run**, counting from
its start rather than from the chunk in hand, so an uninterrupted run, the
same run in different-length chunks, and a run resumed from a checkpoint
all write exactly the same records. A component the workflow runs `n`
times per coupled step, or a nested coupler's inner steps, contributes `n`
records per step and they are kept or dropped together, so components
recording at different rates stay on one cadence. A chunk that contains no
step on the stride gets **no file**, rather than an empty one or a record
the run's cadence does not call for.

`output_averages=True` replaces each chunk's records with their time mean,
labelled at the chunk's midpoint and carrying the CF `cell_methods = "time:
mean"`, with any `time_bounds` variable (JCM's output carries one; the other
components do not) replaced by the chunk's own outer bounds rather than
averaged as data — because the coupler's own output interval, absent this
flag, is one record per coupling step, so this is literally jcm's own
meaning of "averaged output" applied at the chunk granularity. The bins are
the chunks: a 30-day chunk gives 30-day-*window* means, whose boundaries
drift against the calendar months, not calendar monthly means — those come
from `jem.accumulate.monthly_mean` (below). In a coupled run the atmosphere's
per-step records are *already* step means
(`JCMComponent` integrates with `output_averages=True`), so averaging a
chunk of them is the chunk mean exactly, with no double counting.

**In-scan reductions.** Writing every step out and reducing on the host
means holding a chunk's diagnostics — for an atmosphere, the largest array
in the run — until the chunk ends. `generate_trajectory_function(iterations,
accumulate=(init, update))` avoids that: `update(accumulator, diagnostics,
time)` runs inside the `lax.scan` body and the scan returns nothing per
step, so the call becomes `(carry, accumulator=None) -> (carry,
accumulator)` and a chunked run threads the accumulator from one call to
the next with no per-step diagnostics ever materialising:

```python
from jem.accumulate import monthly_mean

monthly = monthly_mean(coupler)                       # 12 bins: a climatology
trajectory = coupler.generate_trajectory_function(365, accumulate=monthly)
carry, accumulator = trajectory(coupler.initialize())
means = monthly.finalize(accumulator)        # (12, …) per variable
```

`run_chunked(..., accumulate=monthly, health_check=None)` does the same
across chunks. An accumulated run writes **no files** (`output_averages` /
`subsample` do nothing and warn if set — there is nothing for
`chunk_datasets` to label), and cannot be given a `health_check` at all: the
combination is a `ValueError`, since a long accumulated run of an
atmosphere is exactly the run that most needs the gate, so losing it has to
be something the caller asked for. The **accumulator is not
checkpointed** — the checkpoint is the model's restart state, and mixing an
analysis product into it would make the format depend on which reduction a
run chose — so a resumed run starts a fresh accumulator covering only what
it integrates from that point (and warns when it does); a mean across a
restart boundary is built by finalizing each call's accumulator and
combining them, or by running the span in one call.

`monthly_mean(coupler)` bins into the twelve calendar months (a run longer
than a year composites its Januaries into bin 0);
`monthly_mean(coupler, total_time=...)` (or `n_months=`) instead gives one
bin per calendar month the run passes through, sized by counting the
distinct Gregorian months the run's records touch — `total_time` is the
form to prefer, since a hand-counted `n_months` can under-size the
accumulator by one and silently wrap a run's last partial year into its
first. Each record is binned by its own interval **midpoint**
(`jcm.date.gregorian_ymd_from_days`), the same instant `TimeAxis.datetimes()`
labels it with, so `monthly.finalize(...)` and
`to_xarray(...).groupby("time.month").mean()` of the same run agree exactly.
`windowed_mean(coupler, window, n_windows=...)` is the same mechanism with a
calendar-agnostic binning rule instead: fixed-length windows measured in
whole coupling steps from the run's own start (a sub-seasonal forecast's
5-day or 7-day means), with `window` optionally a *sequence* of lengths the
accumulator cycles through (daily leads for a forecast's first week, then
pentads). Both wrap at the accumulator's size for a run that outlasts it.

The accumulator is an ordinary pytree in the scan carry, so a binned mean is
differentiable exactly like the trajectory it reduces — a calibration
against a monthly target varies a carried process parameter (see *Parameters*
in {doc}`carry_and_clock`) and differentiates through the reduction:

```python
monthly = monthly_mean(coupled)
trajectory = coupled.generate_trajectory_function(365, accumulate=monthly)
JULY = 6              # finalize()'s leading axis is January first

def loss(relaxation_time, target_july_sst):
    params = ocn.params.replace(relaxation_time=relaxation_time)
    _, accumulator = trajectory(coupled.initialize({"ocn": params}))
    july = monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature[JULY]
    return jnp.mean((july - target_july_sst) ** 2)

gradient = jax.grad(loss)(relaxation_time, target_july_sst)
```

`tests/unit/test_accumulate.py` runs exactly this snippet — the gradient of
an accumulated July mean equals the gradient of the same quantity computed
from the stacked diagnostics, and one descent step reduces the loss — so it
cannot rot silently. For a long calibration, `remat=True` on the trajectory
trades recomputation for the memory the backward pass would otherwise need.
