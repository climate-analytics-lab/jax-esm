"""Checkpointing: writing a coupled carry to disk and reading it back.

A checkpoint directory holds one file per component -- ``{name}_carry.pkl``
for a plain pytree, or a subdirectory written by the component itself when its
carry is not picklable (Veros) -- plus one file, :data:`COUPLED_STEP_FILENAME`,
holding the coupled step counter.

The step counter is part of the checkpoint because it is part of the state.
``CoupledCarry.step`` is the model's only clock: every component's
:class:`~jem.base.component.CouplingTime` -- and therefore its position in the
seasonal cycle -- is derived from it. A checkpoint that saved only the
component carries would resume at step 0, restarting the seasonal cycle in
January however far into the run it was written.
"""

import logging
import os
import pickle
from pathlib import Path
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from jem.base.component import Carry, CoupledCarry

logger = logging.getLogger(__name__)

#: Name of the file in a checkpoint directory that holds the coupled step
#: counter. It cannot collide with a component's ``{name}_carry.pkl``.
COUPLED_STEP_FILENAME = "coupled_step.pkl"


def save_component_carries(
    carries: dict[str, Carry],
    checkpoint_dir: str | Path,
    component_savers: dict[str, Callable[[Carry, Path], None]] | None = None,
) -> None:
    """Save a mapping of name -> carry into ``checkpoint_dir``.

    This is the component half of a checkpoint; :func:`save_coupled_carry`
    wraps it and adds the coupled step counter. It is also what
    :func:`save_veros_carry` uses to write the picklable part of the Veros
    carry, which is a sub-carry and has no clock of its own.

    Pure JAX pytree carries are pickled as ``{name}_carry.pkl``. Carries
    listed in ``component_savers`` are delegated to the provided callable with
    signature ``(carry, checkpoint_dir / name)``.

    Parameters
    ----------
    carries : dict[str, Carry]
        Mapping of component name to carry.
    checkpoint_dir : path-like
        Directory to save into (created if absent).
    component_savers : dict[str, callable], optional
        Mapping of component name to a saver, for carries that are not plain
        pytrees. ``VerosComponent.save_state`` is one.

    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    component_savers = component_savers or {}

    for name, carry in carries.items():
        if name in component_savers:
            component_checkpoint_dir = checkpoint_dir / name
            component_checkpoint_dir.mkdir(exist_ok=True, parents=True)
            component_savers[name](carry, component_checkpoint_dir)
        else:
            carry_numpy = jax.tree_util.tree_map(np.array, carry)
            with open(checkpoint_dir / f"{name}_carry.pkl", "wb") as f:
                pickle.dump(carry_numpy, f)


def load_component_carries(
    checkpoint_dir: str | Path,
    component_names: Iterable[str],
    component_loaders: dict[str, Callable[[Path], Carry]] | None = None,
) -> dict[str, Carry]:
    """Load a mapping of name -> carry from ``checkpoint_dir``.

    The inverse of :func:`save_component_carries`. Use
    :func:`load_coupled_carry` for a whole coupled checkpoint; this is for the
    component half alone -- a sub-carry (Veros' ``derived``/``forcing``), or a
    pre-clock checkpoint whose step counter has to be supplied by the caller.

    Parameters
    ----------
    checkpoint_dir : path-like
        Directory to load from.
    component_names : iterable of str
        The names to load.
    component_loaders : dict[str, callable], optional
        Mapping of component name to a loader with signature
        ``(checkpoint_dir / name) -> carry``.

    Returns
    -------
    dict[str, Carry]

    """
    checkpoint_dir = Path(checkpoint_dir)
    component_loaders = component_loaders or {}

    carries = {}
    for name in component_names:
        if name in component_loaders:
            carries[name] = component_loaders[name](checkpoint_dir / name)
        else:
            with open(checkpoint_dir / f"{name}_carry.pkl", "rb") as f:
                carries[name] = jax.tree_util.tree_map(jnp.array, pickle.load(f))

    return carries


def save_coupled_carry(
    coupled_carry: CoupledCarry,
    checkpoint_dir: str | Path,
    component_savers: dict[str, Callable[[Carry, Path], None]] | None = None,
) -> None:
    """Save a whole coupled carry -- every component's carry and the clock.

    Parameters
    ----------
    coupled_carry : jem.base.component.CoupledCarry
        The carry a trajectory function returned.
    checkpoint_dir : path-like
        Directory to save into (created if absent).
    component_savers : dict[str, callable], optional
        As :func:`save_component_carries`.

    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_file = checkpoint_dir / COUPLED_STEP_FILENAME
    # The step file is the completion marker: `load_coupled_carry` refuses a
    # directory without it. It is removed first and written last, so that a
    # failure part-way through overwriting an existing checkpoint leaves a
    # directory that is refused, rather than one whose stale marker would
    # silently combine component carries from two different steps.
    step_file.unlink(missing_ok=True)
    save_component_carries(
        coupled_carry.components, checkpoint_dir, component_savers)
    # The marker appears at its final name only once it is complete and on
    # disk: its mere existence is what a resume trusts, so a marker that is
    # half-written -- a kill, a full disk, or a pickle that raised -- has to be
    # impossible rather than merely unlikely. Writing to a temporary file in
    # the same directory (so the rename is within one filesystem and therefore
    # atomic) and fsyncing before the rename keeps that promise across a crash
    # as well as across an exception.
    temporary_file = step_file.with_suffix(".pkl.tmp")
    try:
        with open(temporary_file, "wb") as f:
            pickle.dump(np.asarray(coupled_carry.step), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_file, step_file)
    finally:
        temporary_file.unlink(missing_ok=True)


def load_coupled_carry(
    checkpoint_dir: str | Path,
    component_names: Iterable[str],
    component_loaders: dict[str, Callable[[Path], Carry]] | None = None,
) -> CoupledCarry:
    """Load a whole coupled carry -- every component's carry and the clock.

    Parameters
    ----------
    checkpoint_dir : path-like
        Directory to load from.
    component_names : iterable of str
        The component names to load; normally ``coupler.components``.
    component_loaders : dict[str, callable], optional
        As :func:`load_component_carries`.

    Returns
    -------
    jem.base.component.CoupledCarry
        Ready to be handed straight back to a trajectory function, which will
        continue the run from the step the checkpoint was written at.

    Raises
    ------
    ValueError
        If the checkpoint holds no step counter. Checkpoints written before
        the counter was part of the format cannot say which step they are at,
        and guessing (step 0, or a batch index times a batch length) would
        silently restart the seasonal cycle or shift it. The message names the
        missing file and the way to resume anyway.

    """
    checkpoint_dir = Path(checkpoint_dir)
    step_file = checkpoint_dir / COUPLED_STEP_FILENAME
    if not step_file.exists():
        raise ValueError(
            f"{step_file} does not exist, so this checkpoint does not record "
            "the coupled step counter and the run's position in the seasonal "
            "cycle cannot be recovered from it. (Checkpoints written before "
            "the step counter joined the format look like this.) To resume "
            "from it anyway, load the component carries with "
            "`jem.utils.checkpoints.load_component_carries` and build the "
            "`CoupledCarry` yourself with the step you know the run reached."
        )

    components = load_component_carries(
        checkpoint_dir, component_names, component_loaders)
    with open(step_file, "rb") as f:
        step = jnp.asarray(pickle.load(f), dtype=jnp.int32)
    return CoupledCarry(components=components, step=step)


def latest_complete_checkpoint(
    checkpoint_root: str | Path, pattern: str = "step_*"
) -> Path | None:
    """Return the newest checkpoint directory that is actually complete.

    A driver that resumes by taking the last of ``sorted(root.glob(pattern))``
    can pick a directory that no run can load. :func:`save_coupled_carry`
    writes :data:`COUPLED_STEP_FILENAME` *last*, precisely so that a save
    interrupted part-way through leaves a directory without it rather than one
    that silently mixes component carries from two different steps. The cost of
    that ordering is that an interrupted save leaves behind a marker-less
    directory which sorts newest -- so the newest name is not necessarily the
    newest *checkpoint*, and resuming has to skip it.

    Directories are ordered by sorted name. The drivers name a checkpoint
    after the coupled step it was written at, zero-padded to a fixed width
    (``step_00000000``, ``step_00000005``, ...), so sorting the names sorts
    the checkpoints by simulated time. The step a resumed run continues from
    is read from inside the checkpoint, not from its name -- the name is only
    what makes "newest" well defined.

    Each skipped directory is logged at WARNING: a marker-less directory means
    an earlier run died mid-save, which is worth knowing about even though the
    resume itself recovers.

    Parameters
    ----------
    checkpoint_root : path-like
        Directory holding the individual checkpoint directories. A root that
        does not exist is not an error -- it just holds no checkpoints.
    pattern : str
        Glob matched against the names in ``checkpoint_root``. Names that do
        not match are ignored entirely (they are not checkpoints, so they are
        not incomplete ones either).

    Returns
    -------
    pathlib.Path or None
        The newest matching directory holding a completion marker, or None if
        there is no such directory.

    """
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.is_dir():
        return None

    candidates = sorted(
        path for path in checkpoint_root.glob(pattern) if path.is_dir()
    )
    complete = [
        path for path in candidates if (path / COUPLED_STEP_FILENAME).exists()
    ]
    if not complete:
        incomplete_directories = candidates
    else:
        incomplete_directories = candidates[candidates.index(complete[-1]) + 1:]

    for incomplete in incomplete_directories:
        logger.warning(
            "Skipping incomplete checkpoint %s: it has no %s, so the save that "
            "wrote it was interrupted.",
            incomplete,
            COUPLED_STEP_FILENAME,
        )
    return complete[-1] if complete else None


def remaining_batches(
    steps_done: int, total_steps: int, steps_per_batch: int
) -> list[int]:
    """Return the lengths of the batches a run still has to integrate.

    A chunked driver runs its coupled steps in batches of ``steps_per_batch``
    so that it can write output and a checkpoint between them. How many are
    left is a function of the *step counter the checkpoint restored*, never of
    the checkpoint's name or of a batch index: the batch length is a run-time
    choice that may differ between the run that wrote a checkpoint and the run
    that resumes it, so a batch index means nothing across the two, whereas the
    coupled step counts the same coupling steps in both.

    The last batch is short when the total is not a whole number of batches.
    It is returned with its true length rather than rounded up, so a run stops
    exactly at ``total_steps``; a driver pays for it with one extra trajectory
    compile, only on that final batch.

    Parameters
    ----------
    steps_done : int
        Coupled steps already integrated -- ``int(carry.step)`` after loading
        a checkpoint, or 0 for a fresh run.
    total_steps : int
        Coupled steps the whole run is asked for.
    steps_per_batch : int
        Coupled steps in a full batch.

    Returns
    -------
    list of int
        One entry per batch left to run, in order, each the number of coupled
        steps to integrate in it. Empty when the run is already done (or past
        its target, which a shortened ``--total-simulation-days`` produces).

    Raises
    ------
    ValueError
        If ``steps_per_batch`` is not positive, or either step count is
        negative.

    """
    if steps_per_batch <= 0:
        raise ValueError(
            f"steps_per_batch must be positive; got {steps_per_batch!r}."
        )
    if steps_done < 0 or total_steps < 0:
        raise ValueError(
            f"step counts must be non-negative; got steps_done={steps_done!r}, "
            f"total_steps={total_steps!r}."
        )

    remaining = total_steps - steps_done
    if remaining <= 0:
        return []

    full_batches, leftover = divmod(remaining, steps_per_batch)
    batches = [steps_per_batch] * full_batches
    if leftover:
        batches.append(leftover)
    return batches


def _set_veros_runtime_setting(name, value):
    from veros import runtime_settings as rs
    object.__setattr__(rs, "__locked__", False)
    setattr(rs, name, value)
    object.__setattr__(rs, "__locked__", True)


def save_veros_carry(ocn_carry, checkpoint_dir):
    """Save Veros OCN carry: state via HDF5 restart, derived/forcing via pickle.

    Args:
        ocn_carry: Dict with keys "state", "derived", "forcing".
        checkpoint_dir: Directory to save into (created if absent).

    """
    checkpoint_dir = Path(checkpoint_dir)

    from veros.restart import write_restart
    ocn_state = ocn_carry["state"]
    with ocn_state.settings.unlock():
        ocn_state.settings.restart_output_filename = str(checkpoint_dir / "veros.restart.h5")
        logger.info(
            "Saving ocean restart file to %s",
            ocn_state.settings.restart_output_filename,
        )
    write_restart(ocn_state, force=True)

    # The picklable half of the Veros carry. `save_component_carries`, not
    # `save_coupled_carry`: these are two pieces of one component's carry, not
    # a coupled model, and they have no step counter of their own -- the
    # coupled clock is written once, by the checkpoint that contains this one.
    save_component_carries(
        {"derived": ocn_carry["derived"], "forcing": ocn_carry["forcing"]},
        checkpoint_dir,
    )


def load_veros_carry(checkpoint_dir, ocn_model):
    """Load Veros OCN carry: state via HDF5 restart, derived/forcing via pickle.

    Args:
        checkpoint_dir: Directory to load from.
        ocn_model: Veros model instance whose state is mutated in-place.

    Returns:
        Dict with keys "state", "derived", "forcing".

    """
    checkpoint_dir = Path(checkpoint_dir)

    from veros.restart import read_restart
    ocn_state = ocn_model.state
    _set_veros_runtime_setting("force_overwrite", False)
    with ocn_state.settings.unlock():
        ocn_state.settings.restart_input_filename = str(checkpoint_dir / "veros.restart.h5")
    read_restart(ocn_state)
    _set_veros_runtime_setting("force_overwrite", True)

    aux = load_component_carries(checkpoint_dir, ["derived", "forcing"])
    return dict(state=ocn_state, **aux)
