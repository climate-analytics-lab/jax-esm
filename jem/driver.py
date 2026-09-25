"""The chunked run loop: the one driver a coupled model is run through.

A :class:`~jem.base.coupler.Coupler` produces *functions*, not runs. Turning
one into a run means deciding how far to integrate, how often to stop to write
output and a checkpoint, and what to do when the state has gone bad -- and
every example, notebook and experiment driver in this repository used to
decide it again, slightly differently. :func:`run_chunked` is that loop,
written once:

    integrate a chunk -> label it -> write its (reduced) output ->
    check the state is still healthy -> checkpoint -> repeat

or, given an ``accumulate`` reduction, the same loop with the middle two steps
replaced by folding the chunk into an accumulator that crosses the chunk
boundaries -- see :func:`run_chunked`'s Notes for what an accumulated run
gives up in exchange (files, the health gate, and a checkpointed reduction).

**Every default a run has lives here**, on :func:`run_chunked`'s signature.
``jem/config/coupled_run/*.yaml`` names the same keys and :mod:`jem.runners`
passes them through, so a default can only be changed in one place.

Chunking rules
--------------
``total_time`` and ``chunk`` are durations -- a ``jem.base.component.parse_duration_days``
string (``"30 days"``, ``"1 year"``) or a plain number of days -- and both must
be whole multiples of the coupling timestep, because a coupled step is the
smallest thing this loop can integrate. ``total_time`` must in turn be a whole
multiple of ``chunk``: a final short chunk would need a second compiled
trajectory for one call, and a run whose length does not divide into chunks is
much more often a mistake in the configuration than a deliberate request. All
three are checked before anything is built or compiled, and the message names
both quantities -- as is ``subsample``, which is otherwise not read until the
first chunk has already been integrated. ``chunk`` is then free to be chosen
for memory and restart granularity alone: neither what the files hold nor when
the checkpoints fall depends on it, because ``subsample`` counts coupled steps
from the start of the **run** (:mod:`jem.output`) and so does
``checkpoint_interval``.

The trajectory is compiled **once**, for ``chunk`` worth of coupled steps, and
called once per chunk. A resumed run is the one case that can need a second
compile: a checkpoint written by a run with a different chunk length leaves the
step counter part-way through a chunk, so what remains does not divide into
whole chunks and :func:`jem.checkpoint.remaining_batches` makes the **last**
batch a short one -- one extra compile, on that batch only, in exchange for the
run stopping exactly at ``total_time``.

Resuming
--------
``checkpoint_path`` is a **single directory**, rewritten after every chunk,
not a directory of dated checkpoints. That is what makes resuming a run the
same command as starting it: point at the path and the run either starts from
scratch (nothing there) or continues from where it stopped. The cost is that
only the newest state survives; a run that wants a history of restart points
keeps its own directory of them and passes each one in turn.

**Checkpointing is on by default**, because a run long enough to be worth
chunking is a run worth being able to restart, and a default of ``None`` made
losing a week of compute the consequence of forgetting one argument. A
*relative* ``checkpoint_path`` -- including the default ``"checkpoint"`` -- is
resolved against ``output_dir`` rather than against the working directory, so
each run gets its own restart directory (Hydra makes a fresh output directory
per run) and two runs launched from one shell cannot overwrite each other's.
Resuming is then pointing a second run at the first's output directory, which
is the same action that would otherwise overwrite its files -- so it is never
accidental, and the provenance line below says which happened. An absolute
path is used as given, for a run that checkpoints to scratch while writing
output elsewhere; ``None`` disables checkpointing entirely.

``checkpoint_interval`` spaces those saves out, for a run whose chunks are
short for one of the *other* reasons a chunk exists -- a health check every few
days, an output file per day -- and which does not want its restart state
rewritten that often. It is counted in coupled steps from the **start of the
run** rather than from the start of the call, so a resumed run checkpoints at
the same points an uninterrupted one does, and it must be a whole number of
chunks, because a chunk boundary is the only place this loop stops. It need not
divide ``total_time``, since the last chunk is saved regardless, but the run
warns when it does not, because the last gap between saves is then shorter than
the interval. Two things survive it: the last chunk of a completed run is
always checkpointed, so a finished run leaves its final restart state; and a
run the health gate stops writes the last chunk that *passed* on the way out,
so bailing still leaves the restart point at the last healthy state. What it
gives up is a run that is **killed** -- a queue timeout, a node failure --
which falls back to the last interval boundary instead of the last chunk. The
chunks after it are then re-integrated on the resume and their output files
rewritten, which is safe because each file is named after the coupled step its
chunk starts at: the second pass writes the same names from the same starting
state.

That holds only while the resumed run keeps the earlier one's ``chunk``, which
nothing makes it do -- the chunk belongs to the run, not to the checkpoint. A
resume under a different chunk starts its files at different steps, so it would
write *beside* the killed run's leftovers rather than over them and leave the
directory holding two passes' records for overlapping simulated time. So a
resumed run checks, before anything is compiled, that every output file at or
after the step it restored is one this call really writes over
(:func:`_check_resumed_output_is_rewritable`): it starts on one of *this*
run's chunk boundaries **and** before the step this run stops at. Those it
rewrites -- or, for a chunk it keeps no record of (a ``subsample`` stride that
lands on none of that chunk's steps), removes, since this pass's output for
that name is nothing and leaving the earlier pass's file there would leave a
record this pass writes elsewhere standing twice. It says at INFO how many.
Anything else -- a file off the chunk grid, which would overlap this run's
output, or one at or past the end of this run, which nothing it writes reaches
-- it **refuses** with a ``ValueError`` naming the files, grouped by which of
the two they are, and the ways out:
resume with the chunk they were written under (and, for those past the end,
a ``total_time`` that reaches them), remove them, or write into another
``output_dir``. Deleting them instead would be a driver destroying a
killed run's output on its own initiative, which is not its decision to take.

Because there is only ever one checkpoint, the health gate runs *before* it
is written and a chunk the gate rejects is not checkpointed at all: saving it
would overwrite the last healthy restart point with a broken state, and a
resume would then start from the broken one with nothing left to go back to.
A rejected run therefore resumes by repeating the chunk that failed, from the
last chunk that passed.

The overwrite is safe because :mod:`jem.checkpoint` publishes the carry file
-- which holds the coupled step counter -- last, after removing the previous
one: a save interrupted half way through leaves a directory with no carry
file, which this loop refuses to resume from (it warns and starts from
``initial_carry`` instead) rather than resuming from a mixture of two steps.

Where the starting state came from
----------------------------------
Because resuming a run is the *same command* as starting one, nothing in the
command distinguishes the two, and a run that was meant to resume and quietly
cold-started repeats simulated time that has already been paid for. So
:func:`run_chunked` states the provenance of the carry it is about to
integrate, at INFO, in exactly one line, before anything is compiled --
``coupler.initialize()``, the ``initial_carry`` argument, or a named
checkpoint, always with the coupled step. A ``checkpoint_path`` that holds no
complete checkpoint says so on the line before, naming the failure and what
the run does instead -- for a run with no ``initial_carry``, in as many words,
that every component starts from its initial state rather than from a restart.
The wreckage of an interrupted save is a WARNING, because a run died and its
last chunk is gone; a path with nothing at it is INFO, because with
checkpointing on by default that is what every first run sees, and a warning
nobody can avoid is a warning nobody reads. Both name the path, so a mistyped
one is visible in the line the run always prints. :meth:`jem.base.coupler.Coupler.load_carry` completes the picture
by naming each component's own source -- the shared carry file or its own
``load_carry`` -- so no part of a resumed model's carry is unaccounted for.

``CoupledCarry.step`` restored from the checkpoint is the only source of truth
for how far the run has got. Nothing is derived from the chunk index or from a
file name -- and, for the same reason, each chunk's output files are named
after the coupled **step** they start at rather than after a chunk index: the
chunk length belongs to the run, not to the checkpoint, so two runs of the same
simulation with different chunks number their chunks differently while agreeing
exactly on the step. The chunk index survives as what it is, a counter for the
health check and the log line.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

from jem.base.component import (
    CoupledCarry,
    SupportsInternalStepping,
    SupportsXarray,
    parse_duration_days,
)
from jem.checkpoint import CARRY_FILENAME, remaining_batches
from jem.output import (
    check_subsample,
    chunk_datasets,
    output_file_step,
    postprocess_datasets,
    write_chunk,
)

if TYPE_CHECKING:  # pragma: no cover - only the type checker needs these
    from jem.base.coupler import Accumulator, Coupler

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0

#: Relative slack allowed when a duration is divided by the coupling timestep
#: before being called a whole number of steps. The durations come from
#: ``jem.base.component.parse_duration_days``, which returns float days, so an exactly
#: expressible request ("1 year" of daily steps) can still land a few ulps off
#: an integer; anything a user would call "not a whole number of steps" is
#: many orders of magnitude larger than this.
STEP_TOLERANCE = 1e-9

#: Where a run checkpoints when ``checkpoint_path`` is left at its default.
#: Relative, so it lands inside ``output_dir`` -- see
#: :func:`_checkpoint_directory`.
DEFAULT_CHECKPOINT_PATH = "checkpoint"

#: The component name :func:`default_health_check` looks for. It is
#: :class:`~jem.components.jcm.component.JCMComponent`'s own ``name``, and the
#: name the default exchange table wires the atmosphere under.
ATMOSPHERE_NAME = "atm"

#: What a health check returns and is given: ``(datasets, chunk_index,
#: elapsed_days) -> (ok, report)``, where ``datasets`` is the chunk's
#: **unreduced** output -- every record it integrated, whatever
#: ``output_averages`` and ``subsample`` do to the files. ``ok`` False stops
#: the run when ``bail_on_unhealthy``; ``report`` is kept in
#: :attr:`RunResult.reports` whatever it says.
HealthCheck = Callable[[dict[str, xr.Dataset], int, float], tuple[bool, dict]]


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What a chunked run produced, and whether it finished.

    Attributes
    ----------
    final_carry : jem.base.component.CoupledCarry
        The carry the last completed chunk returned. Hand it back as
        ``initial_carry`` to continue the run in the same process.
    steps_completed : int
        ``int(final_carry.step)``: coupled steps completed since the run's
        start date, **including** any restored from a checkpoint. It is the
        run's position on the clock, not the number of steps this call
        integrated.
    completed : bool
        False if the health gate stopped the run before ``total_time``. A run
        that had nothing left to do (a checkpoint already at ``total_time``)
        is completed with no chunks run.
    reports : list of dict
        One report per chunk, in order, exactly as the health check returned
        it. Empty when ``health_check`` is None.
    paths : list of pathlib.Path
        Every output file written, in the order they were written. Empty for
        an accumulated run, which writes none. It is one file per component
        per chunk except where a chunk had nothing to write: with
        ``subsample`` set, a chunk containing no coupled step on the stride
        is skipped rather than written as an empty file (see
        :func:`jem.output.write_chunk`), so ``paths`` is then shorter.
    accumulator : pytree or None
        What the run's ``accumulate`` reduction folded every coupled step
        into, threaded across the chunks; ``None`` when the run was not given
        one. Pass it to that reduction's ``finalize`` for the means. It is
        **not** checkpointed, so it holds only what *this* call integrated --
        see :func:`run_chunked`.

    """

    final_carry: CoupledCarry
    steps_completed: int
    completed: bool
    reports: list[dict]
    paths: list[Path]
    accumulator: Any = None


def default_health_check(
    datasets: Mapping[str, xr.Dataset], chunk_index: int, elapsed_days: float
) -> tuple[bool, dict]:
    """Run jax-gcm's blow-up diagnostics on the atmosphere's chunk of output.

    ``jcm.diagnostics.check_health`` inspects the last record of a dataset for
    the signatures of an atmosphere that has gone unstable (NaNs, temperatures
    and humidities outside anything physical). It is the atmosphere's own
    gate, applied to the atmosphere's own output, so a coupled run stops on
    the same evidence an uncoupled one does.

    Because it judges the chunk by its **last record and its extremes**, it
    must be given the chunk unreduced -- which is what
    :func:`run_chunked` passes it. A chunk mean skips NaNs and averages an
    extreme away, and a ``subsample`` stride need not keep the last record at
    all, so the same atmosphere blowing up in the last hours of a chunk would
    read as healthy in the reduced output that was written.

    The finest thing it can see is therefore one **coupling step**, not one
    model timestep: :class:`~jem.components.jcm.component.JCMComponent`
    integrates each coupling step with JCM's own ``output_averages``, so the
    records being stacked are already step means. A NaN still propagates
    through that mean, so a blow-up is caught; a finite excursion that is over
    within a coupling step is averaged with the rest of the step and can be
    missed. Coupling more often is what makes the gate look more closely.

    A coupled model with no atmosphere -- a slab-only test, a spring, an
    ocean-only configuration -- has nothing this can look at, so it is
    reported as skipped and the run continues. That is a deliberate "no
    opinion", not a pass: a gate for the surface components would have to
    know each one's physical ranges, which is the components' business and is
    tracked separately.

    Parameters
    ----------
    datasets : Mapping[str, xarray.Dataset]
        The chunk's full, unreduced output, keyed by component name.
    chunk_index : int
        Which chunk this is, from zero; passed through to the report.
    elapsed_days : float
        Simulated days at the end of the chunk; passed through to the report.

    Returns
    -------
    tuple
        ``(ok, report)``.

    """
    if ATMOSPHERE_NAME not in datasets:
        return True, {
            "chunk": chunk_index,
            "elapsed_days": elapsed_days,
            "skipped": "no atmosphere",
        }
    # Imported here rather than at module scope so that `import jem` does not
    # pull in jax-gcm (and with it dinosaur and the whole atmosphere) for the
    # sake of a function a slab-only run never calls.
    from jcm.diagnostics import check_health

    ok, report = check_health(datasets[ATMOSPHERE_NAME], chunk_index, elapsed_days)
    return bool(ok), dict(report)


def run_chunked(
    coupler: "Coupler",
    *,
    total_time: str | float,
    chunk: str | float = "30 days",
    initial_carry: CoupledCarry | None = None,
    output_dir: Path | str = "outputs",
    output_averages: bool = False,
    subsample: int = 1,
    health_check: HealthCheck | None = default_health_check,
    bail_on_unhealthy: bool = True,
    checkpoint_path: Path | str | None = DEFAULT_CHECKPOINT_PATH,
    checkpoint_interval: str | float | None = None,
    accumulate: "Accumulator | None" = None,
) -> RunResult:
    """Integrate ``coupler`` for ``total_time``, a chunk at a time.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model. Its workflow, exchangers and clock are the run's;
        there is no second place to configure them.
    total_time : str or float
        How far to integrate, as a ``jem.base.component.parse_duration_days`` string
        (``"90 days"``, ``"1 year"``) or a number of days. Must be a whole
        multiple of both the coupling timestep and ``chunk``.
    chunk : str or float
        Simulated time integrated between output files, checkpoints and
        health checks, in the same forms. Must be a whole multiple of the
        coupling timestep.
    initial_carry : jem.base.component.CoupledCarry, optional
        Where to start. Defaults to ``coupler.initialize()``. A complete
        checkpoint found at ``checkpoint_path`` takes precedence over it, and
        the run says so.
    output_dir : path-like
        Directory the chunk files are written into, created if absent.
    output_averages : bool
        Write each chunk's time mean instead of its individual records; see
        :mod:`jem.output` for what the averaging interval is. This reduces
        the *files* only; ``health_check`` still sees the whole chunk.
    subsample : int
        Keep every ``subsample``-th coupled step in the output -- again, in
        the files only. The stride counts coupled steps from the **start of
        the run** (the step a record covers, not its position in its chunk),
        so a chunked run and a resumed one keep exactly the records an
        uninterrupted run keeps, and the first step of the run is always
        kept. A component that records ``n`` times per coupled step keeps all
        ``n`` records of a kept step and none of a dropped one: the stride is
        in coupled steps, not in records. A chunk that contains no coupled
        step on the stride -- which a ``subsample`` longer than ``chunk`` can
        give, and so can the short final batch a resume under a different
        chunk length ends with -- writes **no file** rather than an empty
        one, and removes an earlier pass's file at that name if one is there.
        ``paths`` is then one file per component per chunk except for the
        chunks that kept nothing.
    health_check : callable, optional
        ``(datasets, chunk_index, elapsed_days) -> (ok, report)``, run after
        each chunk has been written and **before** it is checkpointed.
        ``datasets`` is the
        chunk **as it was integrated** -- every record, unreduced -- and not
        the thinned or averaged form written to disk, so that a gate judging
        a chunk by its last record or its extremes sees them however the run
        is configured to write output. ``None`` disables the gate entirely,
        and no reports are collected.
    bail_on_unhealthy : bool
        Stop at the first chunk the health check rejects, returning a result
        with ``completed=False``. That chunk's output is kept but it is not
        checkpointed, so the restart point stays at the last chunk that
        passed. False logs the failure, checkpoints and carries on -- which
        is what a run studying the instability itself wants.
    checkpoint_path : path-like or None
        Directory holding the run's restart state, ``"checkpoint"`` by
        default -- **checkpointing is on**. The coupled carry is written
        there after every chunk the health gate accepts, and a run started
        with a complete checkpoint already there resumes from it. ``None``
        disables checkpointing entirely.

        A **relative** path is resolved against ``output_dir``, not against
        the working directory: each run's output directory is its own (Hydra
        makes a fresh one per run), so the default gives every run its own
        restart directory, and pointing a second run at the same
        ``output_dir`` is what resumes it -- the one action that resumes a run
        is the one that would otherwise overwrite its output. An **absolute**
        path is used exactly as given, which is how a run checkpoints to
        scratch while writing output somewhere else, and how a queue script
        makes its resubmit command identical to its submit command.

        Two consequences of the default worth knowing. ``output_dir``
        defaults to a *fixed* ``"outputs"`` for a plain Python caller, so
        calling this function twice from the same working directory **resumes
        the first call** rather than repeating it -- and a second call that
        asks for no more time than the first has nothing to integrate, which
        it warns about. And a run launched through Hydra gets a new output
        directory each time, so resuming one from the command line means
        naming the earlier run's ``output_dir``, or giving an absolute
        ``checkpoint_path``.
    checkpoint_interval : str or float, optional
        How often that checkpoint is written, in the same duration forms as
        ``chunk``. ``None``, the default, writes one after **every** chunk the
        health gate accepts. A value must be a whole multiple of ``chunk`` --
        a chunk boundary is the only place this loop stops, so an interval
        between two of them could only round to one of them -- and is counted
        in coupled steps from the **start of the run**, not from the start of
        this call, so a resumed run checkpoints at the same points an
        uninterrupted one does. It is what a run whose chunks are short for
        another reason (a health check every few days, an output file per day)
        uses to stop rewriting its restart state that often. Giving it with
        ``checkpoint_path=None`` is a ``ValueError`` rather than a setting
        silently ignored.

        ``total_time`` need *not* be a whole number of intervals -- that
        costs nothing, since the last chunk of a completed run is checkpointed
        anyway -- but the run warns when it is not, naming both durations,
        because the final gap between saves is then shorter than the interval
        asked for.

        Two guarantees survive the interval. The last chunk of a completed run
        is always checkpointed, whatever the interval, so a finished run leaves
        its final restart state. And if the health gate stops the run
        (``bail_on_unhealthy``) while the last accepted chunk is still unsaved,
        that chunk is checkpointed before this returns -- so bailing still
        leaves the restart point at the last healthy state, exactly as it does
        without an interval.

        What the interval does cost is a run that is *killed* rather than
        stopped -- a queue timeout, a node failure -- which resumes from the
        last interval boundary instead of the last chunk. The chunks after it
        are re-integrated and their output files **rewritten**:
        :func:`jem.output.write_chunk` warns as it overwrites each one, and
        this function says at INFO, before it starts, how many existing files
        the resume will write again. The rewrite lands on the same names only
        while the resumed run keeps the same ``chunk`` *and* runs at least as
        far as the killed one got; a resume that changes the chunk writes
        files starting at different steps, which would leave the killed run's
        beside the new ones holding records for the same simulated time, and
        one that stops earlier leaves that run's later files beyond its own
        end. Either is **refused** (``ValueError``, before anything is
        compiled), naming the files that would be left behind, which of the
        two they are, and the ways out. Files *earlier* than the restart point
        are the run's history and are never in question.
    accumulate : pair of callables, optional
        An **in-scan reduction** of the per-step diagnostics --
        :func:`jem.accumulate.monthly_mean` or
        :func:`jem.accumulate.windowed_mean`, or any ``(init, update)`` pair
        :meth:`~jem.base.coupler.Coupler.generate_trajectory_function` takes.
        Each chunk's trajectory is built with it and the accumulator is
        threaded from chunk to chunk, so the memory a run needs stops growing
        with ``chunk``: the reduction *is* the output. See the Notes for what
        that costs.

    Returns
    -------
    RunResult
        With ``accumulator`` set when ``accumulate`` was given, and
        ``paths`` empty.

    Raises
    ------
    ValueError
        If ``chunk``, ``total_time`` or ``checkpoint_interval`` is not a
        whole number of coupling steps, ``total_time`` is not a whole number
        of chunks, ``checkpoint_interval`` is not a whole number of chunks or
        was given without a ``checkpoint_path``, or ``subsample`` is not a positive
        integer, or if ``accumulate`` is given with a ``health_check``. All of
        them are checked before anything is compiled or integrated. Also if
        the run resumes from a checkpoint and ``output_dir`` already holds
        output at or after the restored step that this run will not write
        again -- an earlier pass's files, off this run's chunk grid or past
        the step it stops at, which would otherwise be left overlapping the
        output this run is about to write or stranded beyond its end.

    Notes
    -----
    Each chunk's output files are named after the coupled step the chunk
    starts at (``<component>-<first step>.nc``), which is unique however the
    run is chunked -- so a run resumed with a different ``chunk`` writes new
    files rather than over the ones it already wrote. That is also why such a
    resume is refused when the directory already holds output at or after the
    restart point that this run will not write again -- off its chunk grid, so
    the two passes' files would overlap in simulated time, or past the step it
    stops at. See :mod:`jem.output`.

    **An accumulated run has no per-step diagnostics**, by construction: the
    scan returns the accumulator instead of stacking every step, which is the
    whole reason to use one. Three consequences, none of them hidden:

    - **No files are written.** ``chunk_datasets`` has nothing to label, so
      ``paths`` is empty and ``output_averages`` and ``subsample`` -- which
      reduce the files -- do nothing. The reduction is the output: take
      ``RunResult.accumulator`` to the reduction's own ``finalize``.
    - **The health gate cannot run**, so ``accumulate`` with a
      ``health_check`` is a ``ValueError`` rather than a gate quietly skipped.
      That is the safer of the two: the gate defaults to *on*, an atmosphere
      is exactly what a long accumulated run is for, and a warning in a log
      file is a poor way to find out weeks later that nothing was watching for
      a blow-up. Passing ``health_check=None`` makes giving up the gate a
      decision the caller took.
    - **The accumulator is not checkpointed**, and a resumed run therefore
      starts a fresh one and accumulates only what it integrates. The
      checkpoint is the *model's restart state*; the accumulator is an
      analysis product. Putting one in the other would make the checkpoint
      format depend on which reduction the run happened to choose -- a
      checkpoint only loadable by a run asking for the same means -- and would
      make a restart able to corrupt an analysis. A run that needs a mean
      across a restart boundary finalizes each call's accumulator and combines
      them, or runs the whole span in one call.

    """
    # Every argument that can be wrong on its own is checked here, before a
    # trajectory is compiled or a step integrated: `subsample` is not read
    # until the first chunk has already been written, and finding out then
    # that it was 0 would have cost a chunk of an atmosphere.
    check_subsample(subsample)
    coupling_days = coupler.dt_seconds / SECONDS_PER_DAY
    steps_per_chunk = _whole_steps(chunk, coupling_days, coupler, "chunk")
    total_steps = _whole_steps(total_time, coupling_days, coupler, "total_time")
    if total_steps % steps_per_chunk:
        raise ValueError(
            f"total_time ({total_time!r}, {total_steps} coupled steps) is not a "
            f"whole number of chunks of {chunk!r} ({steps_per_chunk} coupled "
            "steps): a final partial chunk is not supported, because it would "
            "need a second compiled trajectory for a single call. Choose a "
            "chunk that divides the run, or a run length that is a multiple of "
            "the chunk."
        )
    steps_per_checkpoint = _checkpoint_steps(
        checkpoint_interval, checkpoint_path, chunk, steps_per_chunk,
        coupling_days, coupler,
    )
    if accumulate is not None and health_check is not None:
        raise ValueError(
            "accumulate= reduces the per-step diagnostics inside the scan, so "
            "there are no per-step datasets for a health check to look at. "
            "Pass health_check=None to say that this run goes without the "
            "gate. It is refused rather than skipped because the gate is on "
            "by default and a long accumulated run of an atmosphere is "
            "exactly the run that needs it -- finding out from a log line, "
            "weeks later, that nothing was watching for a blow-up is not a "
            "trade anyone would make on purpose."
        )

    output_dir = Path(output_dir)
    checkpoint_dir = _checkpoint_directory(checkpoint_path, output_dir)

    carry, provenance, resumed = _starting_carry(
        coupler, initial_carry, checkpoint_dir
    )
    # Checked as soon as `first_step` (and the concrete starting carry) is
    # known (a checkpoint load, not a compile) and so still "up front": a run
    # that would carry a step counter -- the coupled one, a sub-stepped
    # element's, a nested coupler's own, a component's own internal counter,
    # or the proleptic Gregorian day count a step is dated with -- past what
    # int32 can hold is refused here rather than left to silently wrap
    # deep inside a traced step (see `_check_step_counters_fit_int32`'s own
    # docstring).
    _check_step_counters_fit_int32(
        coupler, int(carry.step), total_steps, carry.components
    )
    # One line, always, whatever the run does next: a modeller reading a log
    # has to be able to see at a glance whether the state being integrated is
    # a restart or a cold start, and which. It is the first thing the run
    # says, before anything is compiled.
    logger.info("%s", provenance)

    if resumed and accumulate is None and int(carry.step) < total_steps:
        # A resumed run is the one case where the output directory can already
        # hold files for simulated time this run is about to write again, and
        # under a rechunked resume rewriting them is not enough to keep the
        # directory consistent -- see
        # `_check_resumed_output_is_rewritable`, which refuses that resume
        # here, before a trajectory is built or a step integrated.
        #
        # Two conditions narrow it to the runs that can actually create an
        # overlap. An accumulated run writes no files at all. And a call with
        # nothing left to integrate (`carry.step` already at `total_steps`,
        # which is what makes `remaining_batches` empty below) writes none
        # either: the files past its restart point are a killed run's own
        # output, and this call is not the one superseding them, so it has no
        # business refusing to start over them.
        _check_resumed_output_is_rewritable(
            output_dir, _output_names(coupler), int(carry.step),
            steps_per_chunk, total_steps, chunk,
        )

    if accumulate is not None:
        logger.info(
            "Reducing each chunk inside the scan: no per-chunk files are "
            "written, and the reduction is returned on RunResult.accumulator."
        )
        if output_averages or subsample != 1:
            logger.warning(
                "output_averages=%r and subsample=%r reduce the output FILES, "
                "and an accumulated run writes none; they do nothing here.",
                output_averages, subsample,
            )
        if resumed:
            # This run's means really are partial, so this is the moment the
            # warning is about and the moment it can be acted on.
            logger.warning(
                "This run resumed from %s, and the accumulator is NOT part of "
                "a checkpoint -- that holds the model's restart state, not an "
                "analysis product. The reduction therefore starts empty and "
                "covers only the chunks this call integrates, not the ones "
                "the earlier run did.", checkpoint_dir,
            )
        elif checkpoint_dir is not None:
            # Nothing is wrong yet; say it once, at INFO, so that a later
            # resume is not a surprise. A warning here would fire on every
            # accumulated run and so be read by nobody.
            logger.info(
                "The accumulator is not part of the checkpoint written to %s: "
                "a run resumed from it will start a fresh reduction covering "
                "only what it integrates.", checkpoint_dir,
            )

    if steps_per_checkpoint is not None and total_steps % steps_per_checkpoint:
        # Not refused, unlike an interval that does not divide the CHUNK: this
        # one costs nothing, because a completed run always checkpoints its
        # last chunk. It is still worth saying, because the interval is what
        # someone sizing a requeue reasons with, and the last gap is shorter
        # than the one they asked for.
        logger.warning(
            "total_time (%r, %d coupled steps) is not a whole number of "
            "checkpoint_interval (%r, %d coupled steps), so the run's last "
            "checkpoint falls at the end of the run rather than on an interval "
            "boundary -- the final gap between saves is shorter than the "
            "interval. Nothing is lost by it: a completed run always "
            "checkpoints its last chunk.",
            total_time, total_steps, checkpoint_interval, steps_per_checkpoint,
        )

    # Built once and cached by length: every chunk but (at most) the first of
    # a resumed run has the same number of steps, so this compiles one
    # trajectory for the whole run.
    def build_trajectory(iterations: int) -> Callable[..., tuple[CoupledCarry, Any]]:
        return coupler.generate_trajectory_function(iterations, accumulate=accumulate)

    trajectories: dict[int, Callable[..., tuple[CoupledCarry, Any]]] = {
        steps_per_chunk: build_trajectory(steps_per_chunk)
    }

    reports: list[dict] = []
    paths: list[Path] = []
    # Seeded here rather than left as None for the first chunk to fill in.
    # `jax.jit` keys its cache on the argument treedefs, and None is a
    # different tree from the accumulator pytree -- passing None to the first
    # chunk and the real accumulator to the second would compile the whole
    # trajectory twice, which for an atmosphere is minutes. It also means an
    # accumulated run that had nothing to integrate still returns an
    # accumulator its reduction's `finalize` accepts (all bins empty, so all
    # NaN) rather than a None the caller has to special-case.
    accumulator: Any = None if accumulate is None else accumulate[0]()
    batches = remaining_batches(int(carry.step), total_steps, steps_per_chunk)
    if not batches:
        # WARNING, not INFO: the caller asked for a run and got none. With
        # checkpointing on by default, the usual way to reach this is calling
        # `run_chunked` twice with the same `output_dir` -- a re-run of a
        # script, or a notebook cell run again -- which resumes the first call
        # and finds it already finished. That is correct, and it is also not
        # what someone re-running a script to change something expects.
        logger.warning(
            "Nothing to integrate: the run starts at coupled step %d and asks "
            "for %d. %s", int(carry.step), total_steps, provenance,
        )
    elif steps_per_checkpoint is not None and int(carry.step) % steps_per_chunk:
        # The interval is counted from the start of the run and the loop can
        # only stop at a chunk boundary, so when the starting step is not a
        # whole number of THIS run's chunks, no chunk before the last can end
        # on a multiple of the interval. (The last one can: `remaining_batches`
        # makes the final batch the short one, so it ends exactly at
        # `total_steps`.) Every save in between is lost, which is worse than
        # what the interval asked for, so it is said rather than silently
        # accepted. A run starts at such a step in two ways -- a checkpoint
        # written by a run with a DIFFERENT `chunk`, or an `initial_carry`
        # handed in part-way through one -- and only the first has a chunk
        # length to go back to, so only it gets the remedy.
        remedy = (
            " Resuming with the chunk the checkpoint was written under "
            "restores the interval."
        ) if resumed else ""
        logger.warning(
            "This run starts at coupled step %d, which is not a whole number "
            "of the %d-step chunks it is using, so no chunk it integrates "
            "before the last can end on a multiple of the %d-step "
            "checkpoint_interval: it will checkpoint when it finishes (and, if "
            "the health gate stops it, at the last chunk that passed), but not "
            "in between.%s",
            int(carry.step), steps_per_chunk, steps_per_checkpoint, remedy,
        )

    # The last chunk the health gate accepted and the interval did NOT save, so
    # that a bail-out can still leave the restart point at the last healthy
    # state. Exactly one carry is held -- each accepted chunk replaces it -- so
    # the interval costs one carry of device memory, not a history of them.
    pending_carry: CoupledCarry | None = None
    for batch_index, steps in enumerate(batches):
        first_step = int(carry.step)
        # A counter for the health check and the log line only. It is
        # run-global -- how many whole chunks of THIS run's length fit before
        # `first_step` -- so a resumed run carries on numbering rather than
        # starting again at 0, and two chunks of one run never report the same
        # index. The output files are named after `first_step` instead,
        # because a chunk index means different simulated time under a
        # different chunk length.
        chunk_index = first_step // steps_per_chunk
        if steps not in trajectories:
            trajectories[steps] = build_trajectory(steps)

        datasets: dict[str, xr.Dataset] | None = None
        if accumulate is None:
            carry, diagnostics = trajectories[steps](carry)
            # The chunk is labelled once, unreduced, and then reduced only for
            # the copy that is written: `output_averages` and `subsample` are
            # both lossy in the direction the health gate cares about (a time
            # mean skips NaNs and dilutes a finite extreme; a stride can drop
            # the last record entirely), so a gate handed the reduced output
            # would pass a state that went bad at the end of the chunk. See
            # :mod:`jem.output`.
            datasets = chunk_datasets(coupler, diagnostics, first_step=first_step)
            # `first_step` and `steps` place the chunk on the run's clock, so
            # the `subsample` stride counts coupled steps of the RUN: the
            # retained cadence is then the same however the run was chunked
            # and wherever it was resumed, instead of restarting at each
            # chunk's first record.
            reduced = postprocess_datasets(
                datasets, output_averages=output_averages, subsample=subsample,
                first_step=first_step, steps=steps,
            )
            # `write_chunk` skips a component whose reduced chunk holds no
            # record, so what it returns -- not `len(reduced)` -- is what was
            # written.
            chunk_paths = write_chunk(reduced, output_dir, first_step)
            paths.extend(chunk_paths)
            written = f"{len(chunk_paths)} file(s) written"
        else:
            # The accumulator crosses the chunk boundary untouched, which is
            # what lets one compiled trajectory of whatever length suits the
            # machine produce a reduction over bins of any other length.
            carry, accumulator = trajectories[steps](carry, accumulator)
            written = "reduced into the accumulator, no files written"

        elapsed_days = float(coupler.coupling_time(carry.step).sim_time) / SECONDS_PER_DAY
        logger.info(
            "Chunk %d: %d coupled steps run, at step %d, %.4g simulated days "
            "(%.4g years); %s.",
            chunk_index, steps, int(carry.step), elapsed_days,
            elapsed_days / coupler.days_per_year, written,
        )

        # `datasets` is None exactly when `accumulate` is given, and that
        # pairing is refused before anything is compiled -- so the second
        # condition is never what decides this branch at run time. It is here
        # because the type checker cannot reach that argument check from here,
        # and an `assert` would be a claim in the shipped code rather than a
        # restatement of the guard.
        ok = True
        if health_check is not None and datasets is not None:
            ok, report = health_check(datasets, chunk_index, elapsed_days)
            reports.append(report)
            if ok:
                logger.debug(
                    "Chunk %d passed the health check: %s", chunk_index, report
                )
            else:
                logger.error(
                    "Chunk %d failed the health check at %.4g simulated days: %s",
                    chunk_index, elapsed_days, report,
                )
        stopping = not ok and bail_on_unhealthy

        # The gate runs before the checkpoint, and a chunk it rejects is not
        # checkpointed: `jem.checkpoint` keeps a SINGLE restart directory and
        # overwrites it, so saving a state the gate has just rejected would
        # replace the last good one with it and leave the run nothing to go
        # back to -- a resume would start from the broken state, fail again,
        # and have lost the chunk that was still healthy. Stopping instead
        # leaves the restart point at the last chunk that passed, so the run
        # resumes by repeating the chunk that failed. A run integrating an
        # unhealthy state deliberately (`bail_on_unhealthy=False`) does
        # checkpoint it: it is carrying on, and has to stay resumable.
        if checkpoint_dir is not None and not stopping:
            # `checkpoint_interval` is counted in coupled steps from the start
            # of the run -- `carry.step`, which a resumed run restored -- and
            # not from the start of this call, so an interrupted run
            # checkpoints at the same points an uninterrupted one does. The
            # last chunk of a completed run is saved whatever the interval
            # says: a finished run that left no final restart state would have
            # to be re-integrated to be continued.
            last_chunk = batch_index == len(batches) - 1
            if (
                steps_per_checkpoint is None
                or last_chunk
                or int(carry.step) % steps_per_checkpoint == 0
            ):
                coupler.save_carry(carry, checkpoint_dir)
                pending_carry = None
            else:
                pending_carry = carry
        if stopping:
            if checkpoint_dir is not None and pending_carry is not None:
                # The interval skipped the last accepted chunk and the run is
                # stopping here, so that chunk is the last healthy state there
                # will be: write it now rather than leave the restart point at
                # an older interval boundary and make the resume re-integrate
                # healthy chunks it has already paid for.
                coupler.save_carry(pending_carry, checkpoint_dir)
                logger.info(
                    "Checkpointed the last chunk the health gate accepted, at "
                    "coupled step %d: checkpoint_interval had skipped it, and "
                    "it is the state this run stops from.",
                    int(pending_carry.step),
                )
                pending_carry = None
            logger.error(
                "Stopping after %d coupled steps; the output written so far is "
                "kept, and the checkpoint still holds the last chunk that "
                "passed. Pass bail_on_unhealthy=False to integrate an "
                "unhealthy state anyway.", int(carry.step),
            )
            return RunResult(
                carry, int(carry.step), False, reports, paths, accumulator
            )

        # Both copies of the chunk are dropped before the next one is built.
        # Each is a chunk of host-side arrays -- a month of an atmosphere is
        # gigabytes -- and holding one while `chunk_datasets` labels the next
        # would double the run's peak memory for nothing. (An accumulated run
        # never built either.)
        if datasets is not None:
            del datasets, reduced

    return RunResult(carry, int(carry.step), True, reports, paths, accumulator)


def _whole_steps(
    duration: str | float, coupling_days: float, coupler: "Coupler", what: str
) -> int:
    """Return ``duration`` as a whole number of coupled steps, or raise.

    The duration is parsed on the *coupler's own* calendar
    (``jem.base.component.parse_duration_days`` /
    ``jem.base.component.days_per_year``), which for ``"gregorian"`` -- the
    coupler's own default, and the only calendar a real atmosphere accepts --
    is JEM's own fixed-average year, ``365.2425`` days: "1 year" is that many
    days exactly, not the number of days whatever real calendar year the
    run's own dates happen to fall in actually has (365 or 366). On
    ``"365_day"`` it is ``365`` days flat, with no such distinction to draw.
    """
    days = float(parse_duration_days(duration, coupler.calendar))
    steps = days / coupling_days
    rounded = round(steps)
    if abs(steps - rounded) > STEP_TOLERANCE * max(1.0, abs(steps)) or rounded < 1:
        raise ValueError(
            f"{what}={duration!r} is {days:g} days, which is {steps:g} coupling "
            f"steps of {coupling_days:g} days ({coupler.coupling_timestep!r}). "
            "A run is integrated in whole coupled steps, so it must be a whole "
            "positive number of them."
        )
    return int(rounded)


#: The largest magnitude a JAX int32 value can hold. Every counter
#: `_check_step_counters_fit_int32` bounds a run against is one: the coupled
#: step itself, a sub-stepped element's or a nested coupler's own counter,
#: and (via `jem.base.calendar.max_safe_record`) the proleptic Gregorian day
#: count a clock's step is dated with, on every calendar (see
#: `_day_count_limit`).
_STEP_INT32_MAX = 2**31 - 1


def _max_element_rate(coupler: Any) -> int:
    """Return the fastest rate, in records per ``coupler``'s own coupled step,
    any element anywhere in it runs at.

    "Anywhere in it" includes a nested coupler's own step counter (which
    advances ``ratio`` times for every one of the *outer* coupler's coupled
    steps -- :meth:`~jem.base.coupler.Coupler.step`'s own docstring: "the
    inner clock comes from the inner carry's own step counter... continuous
    across outer steps") and, recursively, that nested coupler's own
    elements, whose own combined rate (multiplicity times whatever nesting is
    further inside) is scaled by ``ratio`` again to express it in terms of
    the *outermost* coupler's steps. A plain (non-nested) element of
    multiplicity ``m`` is rate ``m`` on its own
    (:meth:`~jem.base.coupler.Coupler.coupling_time_at_substep`'s ``substep =
    step * m + call``, which is what overflows before the coupled step
    counter itself does whenever ``m > 1``).

    This is exactly what bounds every raw step/substep counter **the coupler
    hierarchy itself owns** against int32, independent of calendar: a
    component run ``n`` times ``r`` levels of nesting deep sees a counter
    that grows ``n`` times faster than the outermost coupled step, so ``n``
    times fewer outer steps are safe.

    **Also covers a component's own internal counters, when it reports
    them.** A plain (non-nested) element additionally contributes its own
    :meth:`~jem.base.component.SupportsInternalStepping.internal_steps_per_call`
    (optional; a component that does not implement it is rate 1, as before
    this capability existed), multiplied by its workflow multiplicity the
    same way a nested coupler's own rate is multiplied by ``ratio`` -- so a
    component that keeps a raw counter of its own faster than the coupled
    step calling it (JCM's ``RunState.step``, or the ``time.step *
    self._inner_steps()`` product ``JCMComponent
    ._report_authoritative_clock_drift`` computes from it) is covered by this
    rate, by :func:`_max_safe_coupled_steps` and by
    :func:`_check_step_counters_fit_int32`, exactly like a workflow
    multiplicity or a nested ``Coupler`` is. ``jem.driver`` still carries no
    jcm-specific knowledge here -- ``SupportsInternalStepping`` is a generic,
    optional capability any component may implement, the same way
    :class:`~jem.base.component.SupportsBind` is. A component that keeps
    such a counter but does not implement the capability is still not
    covered -- reporting it accurately is that component's own
    responsibility, the same way agreeing to :class:`SupportsBind`'s clock
    contract is.

    This rate alone assumes every internal counter starts at zero and
    advances in lockstep with the coupled step -- true of the OUTERMOST
    coupler's own step (``run_chunked`` reads its starting value directly as
    ``first_step``, so ``total_steps`` alone, via
    :func:`_max_safe_coupled_steps`, is enough to bound it), but not of a
    component's own internal counter (a ``VerosComponent`` may wrap a model
    integrated before it was bound; any component's carry may come from a
    resumed run), a nested coupler's own step field (nothing analogous to
    ``first_step`` reads it up front, so a hand-built or otherwise
    out-of-lockstep initial carry can hold it anywhere), or a per-call
    sub-step derived from one (an element run ``m > 1`` times inside a
    coupler whose own step is one of those out-of-lockstep ones -- see
    :func:`_component_internal_counters`'s own docstring).
    :func:`_check_step_counters_fit_int32` checks each of those separately,
    via :func:`_component_internal_counters`, so this rate is only ever one
    part of what actually protects a run.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The (possibly nested) coupled model. Typed ``Any`` here rather than
        ``Coupler`` only because the recursive call reaches a nested
        coupler through ``coupler.components``, whose values are typed as
        the ``Component`` protocol -- narrowed back with ``getattr``
        (``outer_ratio``) and ``hasattr`` (``multiplicities``) rather than
        an ``isinstance`` check, so a nested coupled model does not have to
        import :class:`~jem.base.coupler.Coupler` to be recognised as one.

    Returns
    -------
    int
        At least 1 (a coupler with no multiplicity and no nested coupler
        inside it never counts faster than its own coupled step).

    """
    rate = 1
    for name, multiplicity in coupler.multiplicities().items():
        # `multiplicities()` counts every name in the workflow, exchangers
        # included; only a registered component can be a nested coupler, and
        # `coupler.components` holds only those, so an exchanger's name (or
        # any component that is not itself a coupler) is simply not found and
        # contributes only its own multiplicity below.
        component = coupler.components.get(name)
        inner_ratio = getattr(component, "outer_ratio", None)
        if inner_ratio and hasattr(component, "multiplicities"):
            rate = max(rate, multiplicity * inner_ratio * _max_element_rate(component))
        else:
            # A nested coupler is tested for with `getattr`/`hasattr` above
            # (see the Parameters note on why, not `isinstance`), but this is
            # an ordinary optional *capability* a leaf component opts into,
            # with no import-cycle reason to avoid `isinstance` -- and this
            # module's own convention (`jem.base.component`'s docstring) is
            # to test capabilities that way.
            internal_rate = (
                component.internal_steps_per_call()
                if isinstance(component, SupportsInternalStepping)
                else 1
            )
            rate = max(rate, multiplicity * internal_rate)
    return rate


def _day_count_limit(coupler: Any) -> int:
    """Return the largest step ``coupler``'s own clock can be dated at without an int32 day count wrapping.

    A coupler's step is turned into a proleptic Gregorian date -- an int32
    count of days since the epoch -- whatever ``coupler.calendar`` is, so this
    bound applies on every calendar:

    - :meth:`~jem.base.component.TimeAxis.datetimes` labels every record in
      proleptic Gregorian dates on every calendar (the ``TimeAxis`` class
      docstring). The outermost coupler's records, and those of every
      element and nested coupler under it, are all placed on the outermost
      coupler's own step axis (``Coupler.to_xarray``).
    - ``CouplingTime.year_fraction`` dates every step in-scan via
      :func:`~jem.base.calendar.gregorian_instant` on ``"gregorian"``. On
      ``"365_day"`` it reduces the step modulo one year's worth of steps
      instead and forms no day count.
    - The :class:`~jem.base.component.CouplingTime` a coupler hands its
      components carries ``start_day``/``start_second``, so a component may
      date the step it is handed on the Gregorian calendar itself
      (``JCMComponent``'s clock-agreement check does, through
      :func:`~jem.base.calendar.gregorian_instant`). This is why a nested
      coupler's own step, which never labels output, is bounded on every
      calendar too.

    Checked at ``offset_seconds = coupler.dt_seconds`` (the END of a step's
    interval) rather than ``0``: :func:`~jem.base.calendar.max_safe_record`'s
    bound only ever shrinks as ``offset_seconds`` grows, so this is the
    smallest bound that still covers every instant within a step that a
    caller dates -- its start (``year_fraction``, and a sub-stepped
    element's clock, which falls inside the step it is derived from), its
    midpoint (``jem.accumulate``'s gregorian monthly-mean rules, and
    ``TimeAxis``'s labels) and its end (``TimeAxis``'s interval bound).

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The (possibly nested) coupled model whose own clock (its own
        ``start_date`` and ``dt_seconds``) is being bounded.

    Returns
    -------
    int
        The largest step of ``coupler``'s own clock whose whole interval can
        be dated exactly.

    """
    from jem.base.calendar import max_safe_record

    dt_seconds = int(round(coupler.dt_seconds))
    start = coupler.start_date
    return max_safe_record(
        dt_seconds,
        offset_seconds=dt_seconds,
        start_seconds=int(start.delta.seconds),
        start_days=int(start.delta.days),
    )


def _max_safe_coupled_steps(coupler: "Coupler") -> int:
    """Return the largest coupled-step index ``coupler``'s own clock can hold exactly.

    The minimum of two, genuinely different, int32 limits:

    - **A raw counter limit**, from :func:`_max_element_rate`: the fastest
      any step/substep counter in the coupled hierarchy counts, relative to
      ``coupler``'s own coupled step, bounds how many of *those* the counter
      itself (an ``int32``, however it is eventually used) can hold before
      it, not the outer coupled step, is what overflows first. A counter
      that runs at ``rate`` per outer coupled step, and is itself persisted
      (either the outer ``carry.step`` at ``rate == 1``, or a nested
      coupler's own step counter at whatever ``rate`` its nesting works out
      to -- see :func:`_max_element_rate`'s own docstring), is incremented
      and stored again *after* the last coupled step this run reaches
      (:meth:`~jem.base.coupler.Coupler.step`'s own body: ``step=carry.step
      + 1``). So reaching coupled step ``L`` needs not just the raw value
      *computed while processing* ``L`` to fit (``(L + 1) * rate - 1``, the
      value at the last substep of the last step), but the counter's own
      *next, persisted* value to fit too (``(L + 1) * rate``) -- one more
      than the first, and the one that actually binds: the largest safe
      ``L`` is ``floor(_STEP_INT32_MAX / rate) - 1``. The weaker,
      transient-value bound ``floor((_STEP_INT32_MAX - rate + 1) / rate)``
      would be exactly 1 too generous -- e.g. ``rate == 1`` gives exactly
      ``2**31 - 1``: correct as the last *computed* step, but the
      ``carry.step`` this run would then persist, ``2**31``, silently wraps.
    - **The exact int32 day-count limit** of ``coupler``'s own clock
      (:func:`_day_count_limit`, on every calendar -- see its docstring for
      where a step is dated). This is a property of *elapsed simulated
      time*, not of how many counters divide it up, so it is checked once,
      in terms of ``coupler``'s own coupled step and coupling timestep, and
      not separately for every sub-stepped element's own (finer, but
      proportionally more frequent) clock: a component sub-stepped ``n``
      times covers the same elapsed time in ``n`` times more, ``n`` times
      shorter records, so its own day count limit, expressed in *its own*
      records, is exactly ``n`` times the coupled-step limit -- the same
      number of *coupled* steps either way. A nested coupler's own step is
      not covered by this argument, because it is not guaranteed to stay in
      lockstep with ``coupler``'s (see :func:`_component_internal_counters`),
      so :func:`_check_step_counters_fit_int32` checks it against its own
      :func:`_day_count_limit` separately.

      This day-count limit does **not** need its own ``+1``-style
      reservation the way the raw counter limit above does: it bounds the
      OUTER coupled step (``record == coupler``'s own ``carry.step``)
      directly, at ``rate == 1``, and the raw counter limit above is *always*
      at most ``_STEP_INT32_MAX // 1 - 1 == 2**31 - 2`` for any ``rate >=
      1`` -- so the ``min`` of the two can never exceed that ceiling either,
      whichever one binds.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model a run's (absolute) ``total_steps`` is checked
        against.

    Returns
    -------
    int
        The largest coupled-step index safe to reach.

    """
    rate = _max_element_rate(coupler)
    counter_limit = _STEP_INT32_MAX // rate - 1
    return min(counter_limit, _day_count_limit(coupler))


def _component_internal_counters(
    coupler: Any, carries: dict[str, Any], outer_rate: int = 1, path: str = "",
) -> list[tuple[str, int, int, str, int | None]]:
    """Return ``(path, rate, counter, kind, day_limit)`` for every counter whose starting value is not assumed.

    ``_max_element_rate`` finds the single fastest RATE anywhere in
    ``coupler``, which is enough to bound a counter that always starts at
    zero and advances in lockstep with the coupled step that owns it -- the
    *outermost* coupler's own step, since ``run_chunked`` reads its starting
    value directly as ``first_step`` (:func:`_check_step_counters_fit_int32`
    checks it against that, not against an assumed zero). Every OTHER
    counter this hierarchy carries is not guaranteed to start in lockstep,
    so its own current value has to be read from the concrete carry and
    checked against its own rate individually. Three kinds:

    - A component's own internal counter
      (:class:`~jem.base.component.SupportsInternalStepping`): a
      ``VerosComponent`` may wrap a model that was already integrated before
      it was bound, or ``JCMComponent``'s carry may hold a ``RunState.step``
      from wherever this run's own starting carry came from.
    - A nested coupler's own ``CoupledCarry.step``: the same kind of
      persisted, incrementing counter as the outermost coupler's, but
      nothing analogous to ``run_chunked``'s own ``first_step`` reads it up
      front -- a hand-built or otherwise out-of-lockstep initial carry can
      hold it anywhere relative to ``first_step * rate``. Unlike the other
      two kinds, this one also carries a ``day_limit``, the
      :func:`_day_count_limit` of its own clock (its own ``start_date`` and
      ``dt_seconds``): :func:`_max_safe_coupled_steps`'s "elapsed time is
      invariant to how finely a clock is divided" argument bounds a nested
      coupler's day count only while its step stays in lockstep with the
      outer one, which, like its raw counter, it is not guaranteed to do.
    - A per-call SUB-STEP derived from a nested coupler's own step, for an
      element listed more than once in THAT coupler's own workflow
      (:meth:`~jem.base.coupler.Coupler.coupling_time_at_substep`: ``substep
      = step * multiplicity + call``, computed fresh every call rather than
      persisted). Composing this element's own rate the way
      `_max_element_rate` does -- multiplying the nested coupler's rate by
      `multiplicity` -- correctly bounds how fast the substep *grows*, but
      says nothing about the value it *starts* from: unlike the outermost
      coupler's own step (always exactly `first_step`, never independently
      out of lockstep), a nested coupler's own step can already sit anywhere
      the entry above allows, and multiplying an already-large starting
      value by `multiplicity` can overflow before a single further outer
      step even runs. This is why the outermost coupler needs no equivalent
      entry: its own step is never anything but the value already checked
      via `first_step`/`_max_safe_coupled_steps`, so multiplying it by any
      local multiplicity is already covered by the RATE `_max_element_rate`
      composes into that same check. This same reasoning is why a per-call
      sub-step (the third kind above) and a leaf's own internal counter (the
      first kind) carry no ``day_limit`` of their own (``None``): a sub-step
      is a fixed offset from the nested coupler's own step it is derived
      from (``substep * sub_dt`` differs from ``nested_step * dt`` by less
      than one whole record), whose day-count is already checked by that
      coupler's own entry above, in the same call to this function -- and a
      leaf's own internal counter (JCM's ``RunState.step``, Veros'
      ``itt``) never itself feeds `jem.base.calendar.gregorian_instant`; only
      the coupled-step clock a component is *handed* does, and that clock's
      own day-count is what the coupler-level entries (this function's second
      kind, and the top-level check in `_max_safe_coupled_steps`) already
      cover.

    This walks the same workflow multiplicities and nested-coupler ratios
    ``_max_element_rate`` composes into its single maximum, but keeps one
    ``(rate, counter)`` entry *per* counter found, rather than only the
    largest rate, since each one's own starting value has to be checked
    against its own rate.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The (possibly nested) coupled model.
    carries : dict[str, Any]
        ``coupler``'s own carries mapping at this level of the recursion --
        the concrete, starting carry a counter is read from, not a traced
        one.
    outer_rate : int
        How many times ``coupler``'s own coupled step advances per the
        outermost coupler's coupled step: 1 at the top of the recursion,
        multiplied by each nesting's own ratio going down.
    path : str
        Slash-separated component names from the outermost coupler down to
        ``coupler``, for the error message -- empty at the top of the
        recursion.

    Returns
    -------
    list of (str, int, int, str, int or None)
        One ``(path, rate, counter, kind, day_limit)`` tuple per counter
        found. ``rate`` is that counter's own advance per outermost coupled
        step, ``counter`` is its value in ``carries``, ``kind`` is a short,
        accurate description of what is being checked, for
        :func:`_check_step_counters_fit_int32`'s own error message -- the
        three kinds above are genuinely different counters (a component's own
        internal one, a nested coupler's persisted step, or a transient
        per-call sub-step), and conflating their wording would misdescribe
        whichever one actually overflowed -- and ``day_limit`` is
        :func:`_day_count_limit` of that nested coupler's own clock for the
        second kind, or ``None`` for the other two (see above for why they
        need no day-count check of their own).

    """
    from jem.base.coupler import _inner_carries

    found: list[tuple[str, int, int, str, int | None]] = []
    for name, multiplicity in coupler.multiplicities().items():
        component = coupler.components.get(name)
        if component is None:
            continue
        component_path = f"{path}/{name}" if path else name
        component_rate = outer_rate * multiplicity
        inner_ratio = getattr(component, "outer_ratio", None)
        if inner_ratio and hasattr(component, "multiplicities"):
            nested_rate = component_rate * inner_ratio
            nested_step = int(carries[name].step)
            # The nested coupler's OWN `CoupledCarry.step` is a counter in
            # exactly the same sense as a leaf's `SupportsInternalStepping`
            # one: persisted, incremented once per its own coupled step
            # (`step=carry.step + 1`), and not guaranteed to already equal
            # `first_step * nested_rate` -- a hand-built or otherwise
            # out-of-lockstep initial carry can hold it anywhere. Recursing
            # below finds every counter *inside* this nested coupler; this
            # records the nested coupler's own step alongside them, at the
            # same rate its own recursion is scaled by.
            found.append((
                component_path, nested_rate, nested_step,
                "own step counter (this is itself a nested Coupler)",
                _day_count_limit(component),
            ))
            # Every element run more than once in THIS coupler's own
            # workflow is handed `coupling_time_at_substep`'s derived
            # `nested_step * sub_multiplicity + call` -- see this function's
            # own docstring for why that is a THIRD, separate counter to
            # check here, rather than already covered by `nested_step`
            # above or by `_max_element_rate`'s own rate composition. An
            # exchanger (`component.components.get(sub_name) is None`) is
            # handed the same derived clock as a component would be
            # (`Coupler.generate_step_function`'s own workflow loop does not
            # distinguish them), so it is checked here too, not skipped the
            # way the OUTER loop above skips it for lack of a carry to read
            # a *persisted* counter from -- this entry needs none, since the
            # substep is transient. A further-nested coupler, by contrast,
            # is genuinely exempt: its own `step` ignores the multiplied
            # `time` it is handed and advances from its OWN persisted
            # counter instead (`Coupler.step`'s own docstring), which is
            # exactly the entry the recursion below already adds for it.
            for sub_name, sub_multiplicity in component.multiplicities().items():
                if sub_multiplicity <= 1:
                    continue
                sub_component = component.components.get(sub_name)
                sub_outer_ratio = getattr(sub_component, "outer_ratio", None)
                if sub_outer_ratio and hasattr(sub_component, "multiplicities"):
                    continue
                found.append((
                    f"{component_path}/{sub_name}[substep]",
                    nested_rate * sub_multiplicity,
                    nested_step * sub_multiplicity,
                    "own per-call sub-step counter (Coupler.coupling_time_at_substep's "
                    "step * multiplicity + call, from this coupler's own step)",
                    None,  # covered by `component`'s own "own step counter" entry above
                ))
            found.extend(
                _component_internal_counters(
                    component, _inner_carries(carries, name),
                    nested_rate, component_path,
                )
            )
        elif isinstance(component, SupportsInternalStepping):
            rate = component_rate * component.internal_steps_per_call()
            counter = component.internal_counter(carries[name])
            found.append((
                component_path, rate, counter,
                "own internal counter (SupportsInternalStepping)",
                None,  # a leaf's own counter never itself feeds a Gregorian date
            ))
    return found


def _check_step_counters_fit_int32(
    coupler: "Coupler", first_step: int, total_steps: int, carries: dict[str, Any],
) -> None:
    """Refuse a run whose ``total_steps`` would overflow a clock's own counter.

    Checked once, up front (as soon as ``first_step`` is known from the
    starting carry, before any trajectory is compiled), against
    :func:`_max_safe_coupled_steps` -- rather than discovered from a wrong
    date deep inside a traced step: without this check, a run past this
    bound fails silently (a daily ``year_fraction`` of a 73453 s coupling
    from 2000-01-01 goes wrong at step 58471, about 25 years in, with no
    error at all).

    ``total_steps`` is ``run_chunked``'s and :func:`~jem.checkpoint
    .remaining_batches`'s own ``total_steps``: the ABSOLUTE coupled-step
    count the *whole run* (not just this call) is asked to reach, counting
    from the run's own step 0 -- never a count of steps still to integrate
    from ``first_step``. The last coupled step this run reaches is therefore
    ``total_steps - 1`` regardless of where a resume starts (a resume only
    changes how much of ``0 .. total_steps - 1`` THIS CALL still has to
    integrate, not the run's own target). Computing
    ``first_step + total_steps - 1`` instead would double-count
    ``first_step`` -- refusing a real, legitimate 6000-year run of a doubly
    nested 24x6x5 coupler resumed at coupled step 1,000,000, even though its
    true last step is still well inside the limit.

    Two checks, against two different kinds of counter -- and two different
    kinds of *limit*: a raw int32 range, and the Gregorian day-count bound of
    a coupler's clock (:func:`_day_count_limit`, on every calendar):

    - The OUTERMOST coupler's own step (:func:`_max_element_rate` composes
      the fastest rate anywhere under it, including a workflow multiplicity
      or a nested coupler's own ratio), read from the carry as ``first_step``
      and checked against ``total_steps`` via
      :func:`_max_safe_coupled_steps`, which already folds both limits (the
      raw counter range and ``coupler``'s own day-count bound) into one.
    - Every OTHER counter in the hierarchy (:func:`_component_internal_counters`,
      whose own docstring names the three kinds): a component's own internal
      counter (:class:`~jem.base.component.SupportsInternalStepping`), a
      nested coupler's own step field, and a per-call sub-step derived from
      one -- each ultimately read from ``carries``, the concrete carry this
      run is about to start from, because such a counter's current value is
      not guaranteed to be ``first_step * rate``: a ``VerosComponent`` may
      wrap a model integrated before it was bound, a component's carry may
      come from a resumed run, and a nested coupler's own step (and anything
      derived from it) has nothing analogous to ``first_step`` reading it up
      front. A nested coupler's own step field is also checked against its
      own day-count bound (:func:`_component_internal_counters`'s
      ``day_limit``), for the same reason. That bound is on the last step
      the nested coupler *dates* in this run, ``counter_at_end - 1`` (its
      step is incremented after it is dated), which is the same convention
      the outermost check applies to ``total_steps - 1``; the raw-counter
      bound is instead on ``counter_at_end`` itself, the value persisted in
      the carry.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model being run.
    first_step : int
        The coupled step this call starts at (``int(carry.step)``).
    total_steps : int
        The absolute coupled-step count the whole run is asked to reach.
    carries : dict[str, Any]
        ``coupler``'s own starting carries (``CoupledCarry.components``) --
        the concrete carry a component's internal counter is read from.

    Raises
    ------
    ValueError
        If ``total_steps - 1`` -- the last coupled step this run reaches --
        exceeds :func:`_max_safe_coupled_steps`, if any component's own
        internal counter would exceed ``2**31 - 1`` by the time this run
        reaches ``total_steps``, or if the last step a nested coupler dates
        in this run is past the Gregorian day-count bound of its own clock.

    """
    limit = _max_safe_coupled_steps(coupler)
    last_step = total_steps - 1
    if last_step > limit:
        raise ValueError(
            f"This run asks for {total_steps} coupled step(s) (0 through "
            f"{last_step}) from the start of the run; this call starts at "
            f"step {first_step} and would integrate {total_steps - first_step} "
            f"more of them. {last_step} is past {limit}, the largest this "
            "coupler's own clock can hold exactly: either a step/sub-step "
            "counter somewhere in the coupled hierarchy, or the int32 day "
            "count of the proleptic Gregorian date a step is dated with (on "
            "every calendar: output labels are always Gregorian dates), "
            "would silently wrap rather than stay correct. Refused up front "
            "rather than left to go wrong partway through -- see "
            "jem.driver._max_safe_coupled_steps for exactly what is being "
            "checked."
        )

    steps_this_call = total_steps - first_step
    for path, rate, counter, kind, day_limit in _component_internal_counters(coupler, carries):
        counter_at_end = counter + steps_this_call * rate
        if counter_at_end > _STEP_INT32_MAX:
            raise ValueError(
                f"This run would advance {path!r}'s {kind} from {counter} to "
                f"{counter_at_end} -- {rate} per coupled step, over the "
                f"{steps_this_call} coupled step(s) this run still has to "
                f"reach {total_steps} -- past {_STEP_INT32_MAX}, the largest "
                "an int32 counter can hold: it would silently wrap rather "
                "than stay correct. Refused up front rather than left to go "
                "wrong partway through -- see "
                "jem.driver._component_internal_counters for exactly what "
                "is being checked."
            )
        # The step is dated before it is incremented, so the last step this
        # run dates is one short of the value it leaves in the carry.
        last_dated = counter_at_end - 1
        if day_limit is not None and last_dated > day_limit:
            raise ValueError(
                f"This run would advance {path!r}'s {kind} from {counter} to "
                f"{counter_at_end} -- {rate} per coupled step, over the "
                f"{steps_this_call} coupled step(s) this run still has to "
                f"reach {total_steps} -- dating step {last_dated}, past "
                f"{day_limit}, the largest step {path!r}'s own clock (its "
                "own start date and coupling timestep, via "
                "jem.base.calendar.max_safe_record) can be dated at without "
                "an int32 day count wrapping: the proleptic Gregorian date "
                "of its step (year_fraction on \"gregorian\", and any "
                "component dating the clock it is handed, on every calendar) "
                "would silently go wrong rather than stay correct. Refused up "
                "front rather than left to go wrong partway through -- see "
                "jem.driver._day_count_limit for exactly what is being "
                "checked."
            )


def _checkpoint_steps(
    checkpoint_interval: str | float | None,
    checkpoint_path: Path | str | None,
    chunk: str | float,
    steps_per_chunk: int,
    coupling_days: float,
    coupler: "Coupler",
) -> int | None:
    """Return ``checkpoint_interval`` in coupled steps, or None for every chunk.

    Checked here, with the other durations, so that a run whose checkpointing
    is misconfigured says so before it compiles a trajectory rather than after
    it has integrated a chunk of an atmosphere.

    The one modulo test covers both halves of "a whole number of chunks, and at
    least one": an interval is a whole positive number of coupled steps
    (:func:`_whole_steps` refuses anything else), and a positive number smaller
    than ``steps_per_chunk`` can never be a multiple of it.
    """
    if checkpoint_interval is None:
        return None
    if checkpoint_path is None:
        # Refused rather than ignored: the two settings say opposite things
        # about a run, and a run that asked to checkpoint less often and got no
        # checkpoints at all would find out when it tried to resume.
        raise ValueError(
            f"checkpoint_interval={checkpoint_interval!r} was given with "
            "checkpoint_path=None, which switches checkpointing off entirely, "
            "so there would be no saves for the interval to space out. Give a "
            "checkpoint_path, or drop the interval."
        )
    steps = _whole_steps(
        checkpoint_interval, coupling_days, coupler, "checkpoint_interval"
    )
    if steps % steps_per_chunk:
        raise ValueError(
            f"checkpoint_interval ({checkpoint_interval!r}, {steps} coupled "
            f"steps) is not a whole number of chunks of {chunk!r} "
            f"({steps_per_chunk} coupled steps): a checkpoint is only ever "
            "written where the run stops, which is a chunk boundary, so an "
            "interval between two of them -- or shorter than one chunk -- "
            "could only be rounded to one of them. Choose a multiple of the "
            "chunk, or None to checkpoint after every chunk."
        )
    return steps


def _checkpoint_directory(
    checkpoint_path: Path | str | None, output_dir: Path
) -> Path | None:
    """Return the directory the run checkpoints into, or None for no checkpoint.

    A **relative** ``checkpoint_path`` is resolved against ``output_dir``
    rather than against the working directory. That is what makes
    checkpointing safe to have on by default: every run's output directory is
    its own (Hydra makes a fresh one per run), so the default ``"checkpoint"``
    cannot have two runs writing over each other's restart state, and the one
    way to resume a run -- pointing a second run at the same ``output_dir`` --
    is the same action that would otherwise overwrite its output, which the
    provenance line then reports. Resolving against the working directory
    instead would give every run launched from the same shell the same
    checkpoint directory.

    An absolute path is used as given, for a run that checkpoints to scratch
    while writing its output elsewhere.
    """
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    return path if path.is_absolute() else output_dir / path


def _output_names(coupler: "Coupler") -> list[str]:
    """Return every name this coupler's output files can be written under.

    :meth:`jem.base.coupler.Coupler.to_xarray` keys each dataset by the
    component that produced it, except for a component that is itself a
    coupler: that one returns a dataset per *its* components and they are
    flattened into the result under those inner names. So the names a file in
    ``output_dir`` can carry are this coupler's components with every nested
    coupler replaced, recursively, by the components inside it.

    A component that does not implement
    :class:`~jem.base.component.SupportsXarray` is left out, because
    ``to_xarray`` skips it and no file is ever written under its name. The
    same runtime-checkable predicate is used here as there, so the two cannot
    disagree: a name this run can never write is a name it must not reason
    about, or an unrelated file that happens to be called after such a
    component would refuse a perfectly good resume.

    This is used to decide which files in a directory are **this run's**, and
    it is therefore deliberately conservative in the one case it cannot
    enumerate: a component whose ``to_xarray`` returns a mapping under names
    of its own invention -- which nothing in this repository does but the
    contract allows, since ``_named_datasets`` takes any mapping -- is counted
    under its registered name only. The cost is that
    :func:`_check_resumed_output_is_rewritable` could miss an overlap in such
    a component's files; the alternative, matching on the file-name *shape*
    alone, would have a run refuse to start over files it never wrote and
    cannot reason about, which is the worse of the two.
    """
    # Imported here rather than at module scope because `jem.base.coupler`
    # imports this module's siblings; a function-scope import cannot become a
    # cycle whatever the package grows into.
    from jem.base.coupler import Coupler

    names: list[str] = []
    for name, component in coupler.components.items():
        if isinstance(component, Coupler):
            names.extend(_output_names(component))
        elif isinstance(component, SupportsXarray):
            names.append(name)
    return names


def _check_resumed_output_is_rewritable(
    output_dir: Path,
    names: Sequence[str],
    restored_step: int,
    steps_per_chunk: int,
    total_steps: int,
    chunk: str | float,
) -> list[Path]:
    """Refuse a resume that would leave output this run does not replace.

    A resumed run starts at the step its checkpoint holds, but the directory
    can already hold output for steps **after** it: the earlier run wrote each
    chunk before checkpointing it, so anything it integrated past its last save
    -- a ``checkpoint_interval`` that spaced the saves out, a chunk the health
    gate rejected, a kill between the write and the save -- is on disk with no
    checkpoint behind it. That is not by itself a problem, and with an interval
    it is the ordinary state of a killed run: a file is named after the coupled
    step its chunk starts at, so a resume that keeps the same ``chunk`` writes
    the same names from the same starting state and rewrites each of them
    identically.

    What breaks is a resume that *rechunks*. The chunk belongs to the run and
    not to the checkpoint, so a different one starts its files at different
    steps: a checkpoint at step 4 with one-day files at steps 4 and 5, resumed
    with two-day chunks, overwrites the step-4 file with the records for steps
    5-6 and leaves the step-5 file holding step 6 a second time. Nothing
    downstream can tell that duplicate from a real one, and no later chunk of
    this run will ever rewrite it.

    A file at or after the restart point is therefore acceptable only if this
    call really writes over it, which takes **both** halves of what the run is
    about to do:

    - it starts on this run's **chunk grid**, ``restored_step + k *
      steps_per_chunk`` -- where this run's chunks begin, the short final batch
      included (:func:`jem.checkpoint.remaining_batches` puts it last, so it
      too starts on the grid); and
    - it starts **before** ``total_steps``, because that is where this run
      stops. A run resumed for less simulated time than an earlier, longer pass
      already wrote leaves that pass's files beyond its own end: on the grid or
      not, nothing this call writes replaces them, and the directory would
      afterwards be one run's output followed by another's.

    Everything else raises **before** a trajectory is compiled or a step
    integrated, with the files named and grouped by which of the two they fail
    -- an overlap in the middle of the run, or output past the end of it --
    since the remedy a modeller reaches for differs. The alternatives are both
    worse: silently writing the overlap is the bug this exists for, and
    deleting a file this run is not going to write is not a decision a driver
    should be taking.

    The files this call *does* write are another matter, and one of them is
    written by being removed: a chunk this run keeps no record of (a
    ``subsample`` stride landing on none of its coupled steps) has nothing to
    put at its name, so :func:`jem.output.write_chunk` unlinks whatever is
    there. That is the degenerate case of the rewrite this check has just
    declared -- a name this pass is responsible for, replaced with this pass's
    output for it -- and not a run deleting output it is not responsible for.

    Files *before* the restart point are the run's history -- this call does
    not integrate that simulated time and nothing about it changes -- and are
    ignored. So is any file this run did not write:
    :func:`jem.output.output_file_step` returns a step only for a name one of
    ``names`` would have been written under, so a reanalysis, another model's
    output or another coupler's files in the same directory neither block a
    run nor are touched by one.

    Parameters
    ----------
    output_dir : pathlib.Path
        The directory the run writes its chunks into.
    names : Sequence[str]
        The dataset names this run writes under, from :func:`_output_names`.
    restored_step : int
        The coupled step the run resumed from.
    steps_per_chunk : int
        This run's chunk, in coupled steps.
    total_steps : int
        The coupled step this run stops at; the first step it does not write.
    chunk : str or float
        The same chunk as the caller wrote it, for the message.

    Returns
    -------
    list[pathlib.Path]
        The existing files this run rewrites -- or removes, where it keeps no
        record for that chunk -- in name order; empty if there are none.

    Raises
    ------
    ValueError
        If any file at or after ``restored_step`` is one this run will not
        write again -- off its chunk grid, or at or beyond ``total_steps``.

    """
    if not output_dir.is_dir():
        return []
    existing: list[tuple[Path, int]] = sorted(
        (path, step)
        for path in output_dir.iterdir()
        if path.is_file()
        and (step := output_file_step(path, names)) is not None
        and step >= restored_step
    )
    # One pass, three lists: a directory of a long run holds thousands of
    # files, and every one of them is classified by two comparisons.
    rewritten: list[Path] = []
    overlapping: list[Path] = []
    past_the_end: list[Path] = []
    for path, step in existing:
        if step >= total_steps:
            past_the_end.append(path)
        elif (step - restored_step) % steps_per_chunk:
            overlapping.append(path)
        else:
            rewritten.append(path)

    if overlapping or past_the_end:
        reasons = []
        if overlapping:
            reasons.append(
                f"{', '.join(path.name for path in overlapping)} -- written by "
                "an earlier pass under a different chunk, so they would be "
                "left beside this run's output holding records for the same "
                "simulated time, a duplicate nothing reading the directory "
                "back could tell from a real one"
            )
        if past_the_end:
            reasons.append(
                f"{', '.join(path.name for path in past_the_end)} -- at or "
                f"past coupled step {total_steps}, where this run ends, so "
                "nothing it writes reaches them and they would be left beyond "
                "its output as an earlier, longer pass's"
            )
        raise ValueError(
            f"{output_dir} already holds output at or after coupled step "
            f"{restored_step}, which this run resumed from, that this run will "
            "never write again. A file is named after the coupled step its "
            f"chunk starts at, and this run writes chunks starting at "
            f"{restored_step} + k x {steps_per_chunk} coupled steps "
            f"(chunk={chunk!r}) up to step {total_steps}. "
            + ". ".join(reasons)
            + ". Resume with the chunk those files were written under (and, "
            "for those past the end, a total_time that reaches them), or "
            "remove them, or write this run into another output_dir. (The "
            f"output before step {restored_step} is the run's history and is "
            "not in question.)"
        )
    if rewritten:
        # INFO, not a warning: this is the ordinary state of a run killed
        # between an interval's saves, and `write_chunk` warns again as it
        # overwrites each file. Said once, before the run starts, because it
        # is also the evidence that the check above looked and was satisfied.
        logger.info(
            "%d existing output file(s) at or after coupled step %d start on "
            "this run's chunk boundaries and before it ends, so this run "
            "writes them again from the same state -- or removes them, where "
            "it keeps no record of that chunk: %s.",
            len(rewritten), restored_step,
            ", ".join(path.name for path in rewritten),
        )
    return rewritten


def _starting_carry(
    coupler: "Coupler",
    initial_carry: CoupledCarry | None,
    checkpoint_dir: Path | None,
) -> tuple[CoupledCarry, str, bool]:
    """Return the starting carry, where it came from, and whether it was resumed.

    There are only three places a run's starting state can come from -- a
    checkpoint, the ``initial_carry`` argument, or ``coupler.initialize()`` --
    and which one it was decides what the run *means*: a cold start where a
    restart was intended repeats simulated time that has already been paid
    for, and does it silently, because the command that resumes a run is the
    command that starts one. So the choice is made here, in one place, and
    handed back with the sentence :func:`run_chunked` logs, rather than each
    branch logging its own half of the story.

    A checkpoint directory with no carry file is not a checkpoint but the
    wreckage of an interrupted save (:mod:`jem.checkpoint` publishes that file
    last), so it is stepped over rather than loaded -- loading it would mean
    resuming from component states written at a step this run cannot know.
    Either way -- an interrupted save, or a ``checkpoint_path`` with nothing
    at it -- the message names the failure *and* what the run does instead,
    which is why it is composed here and not in :func:`_load_checkpoint`: a
    caller who also passed an ``initial_carry`` -- a spun-up state, say -- is
    not starting the model from scratch, and saying so would be false. The
    *level* comes from :func:`_load_checkpoint`, which is what distinguishes
    the two: the wreckage of an interrupted save is a WARNING, and a path with
    nothing at it is INFO, because with checkpointing on by default that is
    what every first run sees.
    """
    path = checkpoint_dir
    restored, failure, level = (
        (None, None, logging.INFO) if path is None else _load_checkpoint(coupler, path)
    )
    if restored is not None:
        # A caller who passed both gets told which one won, because the
        # argument they wrote is not the state that is being integrated.
        ignored = "" if initial_carry is None else (
            " The initial_carry argument was not used."
        )
        return restored, (
            f"Resumed from checkpoint {path} at coupled step "
            f"{int(restored.step)}.{ignored}"
        ), True

    if initial_carry is not None:
        carry = initial_carry
        provenance = (
            "Starting from the initial_carry argument at coupled step "
            f"{int(carry.step)}."
        )
        # `initial_carry` may well be a spun-up state, so the consequence of
        # not resuming is only that this call starts where that carry is --
        # not that the model is back at its initial state.
        consequence = (
            "the run starts from the initial_carry argument at coupled step "
            f"{int(carry.step)} rather than from a restart"
        )
    else:
        carry = coupler.initialize()
        reason = (
            "no checkpoint was given"
            if path is None
            else f"{path} holds no complete checkpoint"
        )
        provenance = (
            f"Starting from coupler.initialize() at coupled step "
            f"{int(carry.step)} ({reason})."
        )
        consequence = (
            "every component starts from its initial state rather than from a "
            "restart, and the run begins again at the start date"
        )
    if failure is not None:
        logger.log(level, "%s Nothing is restored from it: %s.", failure, consequence)
    return carry, provenance, False


def _load_checkpoint(
    coupler: "Coupler", checkpoint_path: Path
) -> tuple[CoupledCarry | None, str | None, int]:
    """Return the carry ``checkpoint_path`` holds, or None, why not, and how loudly.

    The reason comes back as a sentence rather than being logged here,
    because only :func:`_starting_carry` knows what the run will do
    *instead* -- and a message that named the failure without its consequence,
    or asserted a consequence that the caller's ``initial_carry`` makes false,
    would be worse than none. :meth:`jem.base.coupler.Coupler.load_carry` logs
    which component came from where, so nothing is said here about a load that
    worked.

    The two ways of not resuming are not equally alarming, which is why the
    level comes back too. A directory with no carry file is the wreckage of an
    interrupted save (:mod:`jem.checkpoint` publishes that file last) -- a run
    died, and its last chunk is gone -- so it is a **warning**. A path with
    nothing at it at all is what every first run sees, and since
    checkpointing is on by default into a fresh output directory, that is the
    common case: it is reported at INFO, alongside the provenance line, which
    names the path so a mistyped one is still visible.
    """
    if (checkpoint_path / CARRY_FILENAME).exists():
        return coupler.load_carry(checkpoint_path), None, logging.INFO
    if checkpoint_path.is_dir():
        return None, (
            f"{checkpoint_path} holds no {CARRY_FILENAME}, so it is not a "
            "complete checkpoint: the save that wrote it was interrupted."
        ), logging.WARNING
    return None, (
        f"There is no checkpoint at {checkpoint_path}; a run asked to resume "
        "from it cannot. One will be written there after every chunk."
    ), logging.INFO
