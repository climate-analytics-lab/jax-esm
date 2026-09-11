"""The chunked run loop: the one driver a coupled model is run through.

A :class:`~jem.base.coupler.Coupler` produces *functions*, not runs. Turning
one into a run means deciding how far to integrate, how often to stop to write
output and a checkpoint, and what to do when the state has gone bad -- and
every example, notebook and experiment driver in this repository used to
decide it again, slightly differently. :func:`run_chunked` is that loop,
written once:

    integrate a chunk -> label and write its output -> checkpoint ->
    check the state is still healthy -> repeat

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
both quantities.

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

The overwrite is safe because :mod:`jem.checkpoint` publishes the carry file
-- which holds the coupled step counter -- last, after removing the previous
one: a save interrupted half way through leaves a directory with no carry
file, which this loop refuses to resume from (it logs and starts from
``initial_carry`` instead) rather than resuming from a mixture of two steps.

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
from jem.output import datasets_for_chunk, write_chunk

if TYPE_CHECKING:  # pragma: no cover - only the type checker needs the class
    from jem.base.coupler import Coupler

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0

#: Relative slack allowed when a duration is divided by the coupling timestep
#: before being called a whole number of steps. The durations come from
#: ``jcm.date.parse_duration_days``, which returns float days, so an exactly
#: expressible request ("1 year" of daily steps) can still land a few ulps off
#: an integer; anything a user would call "not a whole number of steps" is
#: many orders of magnitude larger than this.
STEP_TOLERANCE = 1e-9

#: The component name :func:`default_health_check` looks for. It is
#: :class:`~jem.components.jcm.component.JCMComponent`'s own ``name``, and the
#: name the default exchange table wires the atmosphere under.
ATMOSPHERE_NAME = "atm"

#: What a health check returns and is given: ``(datasets, chunk_index,
#: elapsed_days) -> (ok, report)``. ``ok`` False stops the run when
#: ``bail_on_unhealthy``; ``report`` is kept in :attr:`RunResult.reports`
#: whatever it says.
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
        Every output file written, in the order they were written.

    """

    final_carry: CoupledCarry
    steps_completed: int
    completed: bool
    reports: list[dict]
    paths: list[Path]


def default_health_check(
    datasets: Mapping[str, xr.Dataset], chunk_index: int, elapsed_days: float
) -> tuple[bool, dict]:
    """Run jax-gcm's blow-up diagnostics on the atmosphere's chunk of output.

    ``jcm.diagnostics.check_health`` inspects the last record of a dataset for
    the signatures of an atmosphere that has gone unstable (NaNs, temperatures
    and humidities outside anything physical). It is the atmosphere's own
    gate, applied to the atmosphere's own output, so a coupled run stops on
    the same evidence an uncoupled one does.

    A coupled model with no atmosphere -- a slab-only test, a spring, an
    ocean-only configuration -- has nothing this can look at, so it is
    reported as skipped and the run continues. That is a deliberate "no
    opinion", not a pass: a gate for the surface components would have to
    know each one's physical ranges, which is the components' business and is
    tracked separately.

    Parameters
    ----------
    datasets : Mapping[str, xarray.Dataset]
        The chunk's postprocessed output, keyed by component name.
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
    checkpoint_path: Path | str | None = None,
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
        Where to start. Defaults to ``coupler.initialize()``. A checkpoint
        found at ``checkpoint_path`` replaces it.
    output_dir : path-like
        Directory the chunk files are written into, created if absent.
    output_averages : bool
        Write each chunk's time mean instead of its individual records; see
        :mod:`jem.output` for what the averaging interval is.
    subsample : int
        Keep every ``subsample``-th coupled step in the output.
    health_check : callable, optional
        ``(datasets, chunk_index, elapsed_days) -> (ok, report)``, run on
        each chunk's output after it is written. ``None`` disables the gate
        entirely, and no reports are collected.
    bail_on_unhealthy : bool
        Stop at the first chunk the health check rejects, returning a result
        with ``completed=False``. False logs the failure and carries on --
        which is what a run studying the instability itself wants.
    checkpoint_path : path-like, optional
        Directory holding the run's restart state. The coupled carry is
        written there after every chunk, and a run started with a complete
        checkpoint already there resumes from it. ``None`` disables
        checkpointing.

    Returns
    -------
    RunResult

    Raises
    ------
    ValueError
        If ``chunk`` or ``total_time`` is not a whole number of coupling
        steps, or ``total_time`` is not a whole number of chunks.

    Notes
    -----
    Each chunk's output files are named after the coupled step the chunk
    starts at (``<component>-<first step>.nc``), which is unique however the
    run is chunked -- so a run resumed with a different ``chunk`` writes new
    files rather than over the ones it already wrote. See :mod:`jem.output`.

    """
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

    carry = initial_carry if initial_carry is not None else coupler.initialize()
    if checkpoint_path is not None:
        carry = _resume(coupler, carry, Path(checkpoint_path))

    output_dir = Path(output_dir)
    # Built once and cached by length: every chunk but (at most) the first of
    # a resumed run has the same number of steps, so this compiles one
    # trajectory for the whole run.
    trajectories: dict[int, Callable[[CoupledCarry], tuple[CoupledCarry, Any]]] = {
        steps_per_chunk: coupler.generate_trajectory_function(steps_per_chunk)
    }

    reports: list[dict] = []
    paths: list[Path] = []
    batches = remaining_batches(int(carry.step), total_steps, steps_per_chunk)
    if not batches:
        logger.info(
            "Nothing to integrate: the run starts at coupled step %d and asks "
            "for %d.", int(carry.step), total_steps,
        )
    for steps in batches:
        first_step = int(carry.step)
        # A counter for the health check and the log line only: it says which
        # chunk of *this* call is running. The output files are named after
        # `first_step` instead, because a chunk index means different
        # simulated time under a different chunk length.
        chunk_index = first_step // steps_per_chunk
        if steps not in trajectories:
            trajectories[steps] = coupler.generate_trajectory_function(steps)
        carry, diagnostics = trajectories[steps](carry)

        datasets = datasets_for_chunk(
            coupler,
            diagnostics,
            first_step=first_step,
            output_averages=output_averages,
            subsample=subsample,
        )
        paths.extend(write_chunk(datasets, output_dir, first_step))
        if checkpoint_path is not None:
            coupler.save_state(carry, Path(checkpoint_path))

        elapsed_days = float(coupler.coupling_time(carry.step).sim_time) / SECONDS_PER_DAY
        logger.info(
            "Chunk %d: %d coupled steps run, at step %d, %.4g simulated days "
            "(%.4g years); %d file(s) written.",
            chunk_index, steps, int(carry.step), elapsed_days,
            elapsed_days / coupler.days_per_year, len(datasets),
        )
        if health_check is None:
            continue

        ok, report = health_check(datasets, chunk_index, elapsed_days)
        reports.append(report)
        if ok:
            logger.debug("Chunk %d passed the health check: %s", chunk_index, report)
            continue
        logger.error(
            "Chunk %d failed the health check at %.4g simulated days: %s",
            chunk_index, elapsed_days, report,
        )
        if bail_on_unhealthy:
            logger.error(
                "Stopping after %d coupled steps; the output written so far is "
                "kept. Pass bail_on_unhealthy=False to integrate an unhealthy "
                "state anyway.", int(carry.step),
            )
            return RunResult(carry, int(carry.step), False, reports, paths)

    return RunResult(carry, int(carry.step), True, reports, paths)


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


def _resume(
    coupler: "Coupler", carry: CoupledCarry, checkpoint_path: Path
) -> CoupledCarry:
    """Return the carry to start from: the checkpoint's, if there is one.

    A directory with no carry file is not a checkpoint but the wreckage of an
    interrupted save (:mod:`jem.checkpoint` publishes that file last), so it is
    logged and stepped over rather than loaded -- loading it would mean
    resuming from component states written at a step this run cannot know.
    """
    if not (checkpoint_path / CARRY_FILENAME).exists():
        if checkpoint_path.is_dir():
            logger.warning(
                "%s holds no %s, so it is not a complete checkpoint: the save "
                "that wrote it was interrupted. Starting from the initial "
                "carry instead.", checkpoint_path, CARRY_FILENAME,
            )
        else:
            logger.info(
                "No checkpoint at %s; starting a new run and writing one there "
                "after every chunk.", checkpoint_path,
            )
        return carry
    restored = coupler.load_state(checkpoint_path)
    logger.info(
        "Resumed from %s at coupled step %d.", checkpoint_path, int(restored.step)
    )
    return restored
