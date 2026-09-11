"""Checkpointing: writing a carry to disk and reading it back.

Two layers live here, and a driver normally touches only the second:

- :func:`save` / :func:`load` persist **any** pytree carry to a single
  ``msgpack`` file. The leaves are flattened to a plain list of arrays and
  serialised with ``flax.serialization``; the *structure* is not stored at
  all but taken from a ``template`` at load time, which is what makes the
  format both small and self-checking -- a checkpoint written by a different
  model configuration fails on the leaf count, the leaf's name, its shape or
  its dtype rather than deserialising into something that only explodes later
  inside a ``lax.scan``.
- :func:`save_coupled` / :func:`load_coupled` lay a whole
  :class:`~jem.base.component.CoupledCarry` out in a directory, delegating
  the components that write themselves. They are what
  :meth:`jem.base.coupler.Coupler.save_state` and
  :meth:`~jem.base.coupler.Coupler.load_state` are built on; a driver calls
  those, because only the coupler knows which of its components are
  :class:`~jem.base.component.SupportsCheckpoint`.

The directory layout is::

    <directory>/
        <name>/               one per SupportsCheckpoint component (Veros, a
                              nested Coupler), written by the component itself
        carry.msgpack         every other component's carry, plus the clock

**The coupled step counter is part of the checkpoint because it is part of
the state.** ``CoupledCarry.step`` is the model's only clock: every
component's :class:`~jem.base.component.CouplingTime` -- and therefore its
position in the seasonal cycle -- is derived from it. A checkpoint that saved
only the component carries would resume at step 0, restarting the seasonal
cycle in January however far into the run it was written. A directory whose
``carry.msgpack`` does not hold a step is refused with ``ValueError`` rather
than resumed from a guess.

**The carry file is the completion marker.** It is written last, and
published by renaming a fully-flushed, fsynced temporary file over its final
name, so it exists only once the whole checkpoint is on disk: an interrupted
save leaves a directory that :func:`load_coupled` refuses and
:func:`latest_complete_checkpoint` skips. A separate ``COMPLETE`` file would
add a second thing to keep in step with no gain, because the clock -- which
a resume cannot do without -- lives in the carry file anyway, so its presence
is exactly the condition "this checkpoint can be resumed from". When an
existing checkpoint is overwritten the carry file is removed *first*, so a
failure part-way through cannot leave a stale clock next to freshly written
component data.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from jem.base.component import Carry, CoupledCarry

logger = logging.getLogger(__name__)

#: Name of the file holding the leaves that are not delegated to a component,
#: and with them the coupled step counter. Written last, so its presence is
#: what marks a checkpoint directory complete.
CARRY_FILENAME = "carry.msgpack"

#: Key the flattened leaves are stored under inside that file. Named rather
#: than bare so that a future format can add fields beside it.
_LEAVES_KEY = "leaves"

#: Key the leaves' paths in the pytree are stored under. They are what makes
#: a template mismatch a named error instead of a silent mis-assignment: two
#: carries can have the same number of leaves, with the same shapes and
#: dtypes, and still be different carries -- two components swapped, a field
#: renamed, a dict reordered -- and only the paths tell them apart.
_PATHS_KEY = "leaf_paths"


def _canonical_leaf(leaf: Any) -> np.ndarray:
    """Return ``leaf`` as the numpy array JAX would carry it as.

    Both the saved value and the template go through this, and they have to,
    because the same carry has two spellings: ``Coupler.initialize()`` may
    leave a parameter as a Python ``float`` (a component's default), while the
    carry that comes back out of ``lax.scan`` has it as a float32 array. Going
    through ``jnp.asarray`` applies JAX's own dtype canonicalisation to both,
    so the two spellings compare equal instead of reporting a spurious
    float64-vs-float32 mismatch; ``np.asarray`` then brings it back to the host
    for serialisation.
    """
    return np.asarray(jnp.asarray(leaf))


def save(carry: Carry, path: str | Path) -> Path:
    """Write any pytree ``carry`` to ``path`` as one msgpack file.

    Only the leaves are written, as arrays; the pytree structure is
    reconstructed at load time from a template (see :func:`load`). Scalar
    leaves keep their dtype -- the JCM physics carry holds int and bool flags
    whose meaning a silent cast to float would destroy -- because every leaf
    is stored as a typed array rather than as a bare msgpack number.

    The file is published atomically: the bytes go to ``<path>.tmp``, which is
    flushed and fsynced, and only then renamed over ``path``. A save
    interrupted by a kill or a full disk therefore leaves the previous
    checkpoint intact rather than a truncated file that would fail to load.

    Parameters
    ----------
    carry : Carry
        Any pytree. Leaves must be array-like.
    path : path-like
        File to write; its parent directory is created if absent.

    Returns
    -------
    pathlib.Path
        ``path``, for chaining.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keyed_leaves = jax.tree_util.tree_flatten_with_path(carry)[0]
    payload = {
        _LEAVES_KEY: [_canonical_leaf(leaf) for _, leaf in keyed_leaves],
        _PATHS_KEY: [
            jax.tree_util.keystr(key_path) for key_path, _ in keyed_leaves
        ],
    }
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with open(temporary_path, "wb") as f:
            f.write(flax.serialization.msgpack_serialize(payload))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        # Nothing is left behind on the failure path either: a stray `.tmp`
        # next to a checkpoint directory reads as a checkpoint in progress.
        temporary_path.unlink(missing_ok=True)
    return path


def load(template: Carry, path: str | Path) -> Carry:
    """Read back a pytree written by :func:`save`, shaped like ``template``.

    ``template`` supplies everything the file does not: the pytree structure
    and, per leaf, the shape and dtype the model expects. All three are
    checked -- the file records each leaf's *path* in the tree it came from as
    well as its array, so two carries with the same leaf count, shapes and
    dtypes but different names are not silently interchanged -- and a
    checkpoint from a different grid, a different vertical resolution or a
    different component composition is refused *here*, with the offending leaf
    named, rather than deserialising cleanly and failing much later inside a
    traced step.

    The values in ``template`` are never used; only its structure is. A
    caller with no carry to hand can therefore build one from
    ``jax.eval_shape``.

    Parameters
    ----------
    template : Carry
        A pytree with the structure, shapes and dtypes the result must have --
        typically ``component.initialize()``.
    path : path-like
        A file written by :func:`save`.

    Returns
    -------
    Carry
        ``template``'s structure, filled with the saved values as JAX arrays.

    Raises
    ------
    ValueError
        If the file is unreadable, holds a different number of leaves, holds
        them under different names, or holds a leaf whose shape or dtype
        differs from the template's. The message names the file and, for a
        leaf, its path in the pytree.

    """
    path = Path(path)
    try:
        payload = flax.serialization.msgpack_restore(path.read_bytes())
    except Exception as exc:  # noqa: BLE001 - re-raised with the file named
        raise ValueError(
            f"{path} is not a readable JEM checkpoint file: {exc}"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or _LEAVES_KEY not in payload
        or _PATHS_KEY not in payload
    ):
        raise ValueError(
            f"{path} does not hold a JEM checkpoint payload (it has no "
            f"{_LEAVES_KEY!r} and {_PATHS_KEY!r} entries)."
        )
    saved_leaves = list(payload[_LEAVES_KEY])
    saved_paths = list(payload[_PATHS_KEY])

    keyed_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    if len(saved_leaves) != len(keyed_leaves):
        raise ValueError(
            f"{path} holds {len(saved_leaves)} leaves but the model expects "
            f"{len(keyed_leaves)}. It was written by a different component "
            "composition, or by a version whose carry structs had different "
            "fields."
        )

    restored: list[jnp.ndarray] = []
    for (key_path, template_leaf), saved, saved_path in zip(
        keyed_leaves, saved_leaves, saved_paths, strict=True
    ):
        expected_path = jax.tree_util.keystr(key_path)
        if saved_path != expected_path:
            raise ValueError(
                f"{path}: the saved carry does not have the model's structure "
                f"-- its leaf {len(restored)} is {saved_path}, where the model "
                f"has {expected_path}. Something was renamed, reordered, added "
                "or removed since this checkpoint was written."
            )
        expected = _canonical_leaf(template_leaf)
        saved = np.asarray(saved)
        if saved.shape != expected.shape or saved.dtype != expected.dtype:
            raise ValueError(
                f"{path}: leaf {jax.tree_util.keystr(key_path)} was saved as "
                f"{saved.dtype}{saved.shape} but the model expects "
                f"{expected.dtype}{expected.shape}. This checkpoint was "
                "written by a different configuration (grid, resolution or "
                "component parameters)."
            )
        restored.append(jnp.asarray(saved))
    return jax.tree_util.tree_unflatten(treedef, restored)


def save_coupled(
    coupled_carry: CoupledCarry,
    directory: str | Path,
    component_savers: Mapping[str, Callable[[Carry, Path], None]] | None = None,
) -> None:
    """Write a whole coupled carry -- every component's carry and the clock.

    Components named in ``component_savers`` write themselves into
    ``directory / <name>``; every other component's carry, together with
    ``coupled_carry.step``, goes into the single
    :data:`CARRY_FILENAME` file, which is written **last** and is therefore
    the checkpoint's completion marker (see the module docstring).

    Parameters
    ----------
    coupled_carry : jem.base.component.CoupledCarry
        The carry a trajectory function returned.
    directory : path-like
        Directory to save into (created if absent).
    component_savers : mapping, optional
        ``{component name: (carry, directory) -> None}`` for the components
        whose carry is not a plain pytree -- in practice the ``save_state`` of
        every :class:`~jem.base.component.SupportsCheckpoint` component, which
        :meth:`jem.base.coupler.Coupler.save_state` assembles.

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    component_savers = component_savers or {}

    carry_file = directory / CARRY_FILENAME
    # Removed before anything else is written: overwriting a checkpoint in
    # place would otherwise leave the old clock beside new component data if
    # the save died in between, and that mixture is indistinguishable from a
    # good checkpoint.
    carry_file.unlink(missing_ok=True)

    plain: dict[str, Carry] = {}
    for name, carry in coupled_carry.components.items():
        saver = component_savers.get(name)
        if saver is None:
            plain[name] = carry
        else:
            component_directory = directory / name
            component_directory.mkdir(parents=True, exist_ok=True)
            saver(carry, component_directory)

    save({"step": coupled_carry.step, "components": plain}, carry_file)


def load_coupled(
    directory: str | Path,
    component_templates: Mapping[str, Carry],
    component_loaders: Mapping[str, Callable[[Path], Carry]] | None = None,
) -> CoupledCarry:
    """Read back a coupled carry written by :func:`save_coupled`.

    Parameters
    ----------
    directory : path-like
        A directory written by :func:`save_coupled`.
    component_templates : mapping
        ``{component name: template carry}`` for the components stored in the
        shared carry file -- everything ``component_loaders`` does not cover.
        Only the structure, shapes and dtypes are used;
        ``component.initialize()`` is the natural source.
    component_loaders : mapping, optional
        ``{component name: directory -> carry}`` for the components that read
        themselves back, i.e. the ``load_state`` of every
        :class:`~jem.base.component.SupportsCheckpoint` component. A name here
        must not also appear in ``component_templates``.

    Returns
    -------
    jem.base.component.CoupledCarry
        Ready to be handed straight back to a trajectory function, which
        continues the run from the step the checkpoint was written at.

    Raises
    ------
    ValueError
        If the directory holds no :data:`CARRY_FILENAME` -- it is not a
        complete checkpoint, so the coupled step counter, and with it the
        run's position in the seasonal cycle, cannot be recovered -- or if
        the stored carry does not match the templates.

    """
    directory = Path(directory)
    component_loaders = component_loaders or {}
    carry_file = directory / CARRY_FILENAME
    if not carry_file.exists():
        raise ValueError(
            f"{carry_file} does not exist, so {directory} is not a complete "
            "checkpoint: the save that wrote it was interrupted before the "
            "carry file (which is written last, and holds the coupled step "
            "counter) was published. Resume from an earlier checkpoint -- "
            "`jem.checkpoint.latest_complete_checkpoint` picks the newest one "
            "that is loadable."
        )
    overlap = sorted(set(component_loaders) & set(component_templates))
    if overlap:
        raise ValueError(
            f"{overlap!r} are named both as components that load themselves and "
            "as components stored in the shared carry file; each component is "
            "one or the other."
        )

    template = {"step": jnp.int32(0), "components": dict(component_templates)}
    stored = load(template, carry_file)

    components = dict(stored["components"])
    for name, loader in component_loaders.items():
        components[name] = loader(directory / name)
    return CoupledCarry(components=components, step=stored["step"])


def latest_complete_checkpoint(
    checkpoint_root: str | Path, pattern: str = "step_*"
) -> Path | None:
    """Return the newest checkpoint directory that is actually complete.

    A driver that resumes by taking the last of ``sorted(root.glob(pattern))``
    can pick a directory that no run can load. :func:`save_coupled` writes
    :data:`CARRY_FILENAME` *last*, precisely so that a save interrupted
    part-way through leaves a directory without it rather than one that
    silently mixes component carries from two different steps. The cost of
    that ordering is that an interrupted save leaves behind a carry-less
    directory which sorts newest -- so the newest name is not necessarily the
    newest *checkpoint*, and resuming has to skip it.

    Directories are ordered by sorted name. The drivers name a checkpoint
    after the coupled step it was written at, zero-padded to a fixed width
    (``step_00000000``, ``step_00000005``, ...), so sorting the names sorts
    the checkpoints by simulated time. The step a resumed run continues from
    is read from inside the checkpoint, not from its name -- the name is only
    what makes "newest" well defined.

    Each skipped directory is logged at WARNING: a carry-less directory means
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
    complete = [path for path in candidates if (path / CARRY_FILENAME).exists()]
    if not complete:
        incomplete_directories: Iterable[Path] = candidates
    else:
        incomplete_directories = candidates[candidates.index(complete[-1]) + 1:]

    for incomplete in incomplete_directories:
        logger.warning(
            "Skipping incomplete checkpoint %s: it has no %s, so the save that "
            "wrote it was interrupted.",
            incomplete,
            CARRY_FILENAME,
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
        its target, which a shortened total simulation time produces).

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
