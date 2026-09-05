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
    with open(step_file, "wb") as f:
        pickle.dump(np.asarray(coupled_carry.step), f)


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
