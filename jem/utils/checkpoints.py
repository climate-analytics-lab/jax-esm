from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def save_coupled_carry(coupled_carry, checkpoint_dir, component_savers=None):
    """Save coupled model carry to checkpoint_dir.

    Pure JAX pytree components are pickled as {name}_carry.pkl.
    Components listed in component_savers are delegated to the provided
    callable with signature (carry, checkpoint_dir / name).

    Args:
        coupled_carry: Dict mapping component name -> carry.
        checkpoint_dir: Directory to save into (created if absent).
        component_savers: Optional dict mapping component name -> callable.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    component_savers = component_savers or {}

    for name, carry in coupled_carry.items():
        if name in component_savers:
            component_savers[name](carry, checkpoint_dir / name)
        else:
            carry_numpy = jax.tree_util.tree_map(np.array, carry)
            with open(checkpoint_dir / f"{name}_carry.pkl", "wb") as f:
                pickle.dump(carry_numpy, f)


def load_coupled_carry(checkpoint_dir, component_names, component_loaders=None):
    """Load coupled model carry from checkpoint_dir.

    Pure JAX pytree components are loaded from {name}_carry.pkl.
    Components listed in component_loaders are delegated to the provided
    callable with signature (checkpoint_dir / name) -> carry.

    Args:
        checkpoint_dir: Directory to load from.
        component_names: List of component names to load.
        component_loaders: Optional dict mapping component name -> callable.

    Returns:
        Dict mapping component name -> carry.
    """
    checkpoint_dir = Path(checkpoint_dir)
    component_loaders = component_loaders or {}

    coupled_carry = {}
    for name in component_names:
        if name in component_loaders:
            coupled_carry[name] = component_loaders[name](checkpoint_dir / name)
        else:
            with open(checkpoint_dir / f"{name}_carry.pkl", "rb") as f:
                coupled_carry[name] = jax.tree_util.tree_map(jnp.array, pickle.load(f))

    return coupled_carry


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
    write_restart(ocn_state, force=True)

    save_coupled_carry(
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

    aux = load_coupled_carry(checkpoint_dir, ["derived", "forcing"])
    return dict(state=ocn_state, **aux)
