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
``total_time`` and ``chunk`` are durations -- a ``jcm.date.parse_duration_days``
string (``"30 days"``, ``"1 year"``) or a plain number of days -- and both must
be whole multiples of the coupling timestep, because a coupled step is the
smallest thing this loop can integrate. ``total_time`` must in turn be a whole
multiple of ``chunk``: a final short chunk would need a second compiled
trajectory for one call, and a run whose length does not divide into chunks is
much more often a mistake in the configuration than a deliberate request. All
three are checked before anything is built or compiled, and the message names
both quantities -- as is ``subsample``, which is otherwise not read until the
first chunk has already been integrated.

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
complete checkpoint adds a WARNING naming the failure and what the run does
instead -- for a run with no ``initial_carry``, in as many words, that every
component starts from its initial state rather than from a restart. It covers
both an interrupted save and a path with nothing at it, because the loop
cannot tell a first run from a mistyped path and the consequence is the same
either way. :meth:`jem.base.coupler.Coupler.load_state` completes the picture
by naming each component's own source -- the shared carry file or its own
``load_state`` -- so no part of a resumed model's state is unaccounted for.

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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

from jem.base.component import CoupledCarry
from jem.checkpoint import CARRY_FILENAME, remaining_batches
from jem.output import (
    check_subsample,
    chunk_datasets,
    postprocess_datasets,
    write_chunk,
)

if TYPE_CHECKING:  # pragma: no cover - only the type checker needs these
    from jem.base.coupler import Accumulator, Coupler

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0

#: Relative slack allowed when a duration is divided by the coupling timestep
#: before being called a whole number of steps. The durations come from
#: ``jcm.date.parse_duration_days``, which returns float days, so an exactly
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
        an accumulated run, which writes none.
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
    accumulate: "Accumulator | None" = None,
) -> RunResult:
    """Integrate ``coupler`` for ``total_time``, a chunk at a time.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model. Its workflow, exchangers and clock are the run's;
        there is no second place to configure them.
    total_time : str or float
        How far to integrate, as a ``jcm.date.parse_duration_days`` string
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
        the files only.
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
        scratch while writing output somewhere else.
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
        If ``chunk`` or ``total_time`` is not a whole number of coupling
        steps, ``total_time`` is not a whole number of chunks, or
        ``subsample`` is not a positive integer, or if ``accumulate`` is
        given with a ``health_check``. All of them are checked before
        anything is compiled or integrated.

    Notes
    -----
    Each chunk's output files are named after the coupled step the chunk
    starts at (``<component>-<first step>.nc``), which is unique however the
    run is chunked -- so a run resumed with a different ``chunk`` writes new
    files rather than over the ones it already wrote. See :mod:`jem.output`.

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

    carry, provenance = _starting_carry(coupler, initial_carry, checkpoint_dir)
    # One line, always, whatever the run does next: a modeller reading a log
    # has to be able to see at a glance whether the state being integrated is
    # a restart or a cold start, and which. It is the first thing the run
    # says, before anything is compiled.
    logger.info("%s", provenance)

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
        if checkpoint_dir is not None:
            logger.warning(
                "The accumulator is not part of the checkpoint -- that holds "
                "the model's restart state, not an analysis product -- so a "
                "run resumed from %s starts a fresh accumulator and its means "
                "cover only the chunks that call integrates.", checkpoint_dir,
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
        logger.info(
            "Nothing to integrate: the run starts at coupled step %d and asks "
            "for %d.", int(carry.step), total_steps,
        )
    for steps in batches:
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
            reduced = postprocess_datasets(
                datasets, output_averages=output_averages, subsample=subsample
            )
            paths.extend(write_chunk(reduced, output_dir, first_step))
            written = f"{len(reduced)} file(s) written"
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
            coupler.save_state(carry, checkpoint_dir)
        if stopping:
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

    The duration is parsed on the *coupler's* calendar, so "1 year" is as long
    as the atmosphere's year rather than as long as a Gregorian one.
    """
    # Imported here rather than at module scope: see `default_health_check`.
    from jcm.date import parse_duration_days

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


def _starting_carry(
    coupler: "Coupler",
    initial_carry: CoupledCarry | None,
    checkpoint_dir: Path | None,
) -> tuple[CoupledCarry, str]:
    """Return the carry the run starts from, and a sentence saying where from.

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
    That, and a ``checkpoint_path`` with nothing at it at all, are **warned**
    about rather than merely noted: in both the run was asked to resume and
    could not, whether because a save died or because the path is not the one
    the earlier run wrote (a typo resolves to a directory that does not
    exist). The warning names the failure *and* what the run does instead,
    which is why it is composed here and not in :func:`_load_checkpoint`: a
    caller who also passed an ``initial_carry`` -- a spun-up state, say -- is
    not starting the model from scratch, and a warning saying so would be
    false.
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
        )

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
    return carry, provenance


def _load_checkpoint(
    coupler: "Coupler", checkpoint_path: Path
) -> tuple[CoupledCarry | None, str | None, int]:
    """Return the carry ``checkpoint_path`` holds, or None, why not, and how loudly.

    The reason comes back as a sentence rather than being logged here,
    because only :func:`_starting_carry` knows what the run will do
    *instead* -- and a message that named the failure without its consequence,
    or asserted a consequence that the caller's ``initial_carry`` makes false,
    would be worse than none. :meth:`jem.base.coupler.Coupler.load_state` logs
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
        return coupler.load_state(checkpoint_path), None, logging.INFO
    if checkpoint_path.is_dir():
        return None, (
            f"{checkpoint_path} holds no {CARRY_FILENAME}, so it is not a "
            "complete checkpoint: the save that wrote it was interrupted."
        ), logging.WARNING
    return None, (
        f"There is no checkpoint at {checkpoint_path}; a run asked to resume "
        "from it cannot. One will be written there after every chunk."
    ), logging.INFO
