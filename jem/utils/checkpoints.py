from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def save_coupled_carry(coupled_carry, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for component_name, carry in coupled_carry.items():
        carry_numpy = jax.tree_util.tree_map(np.array, carry)
        with open(checkpoint_dir / f"{component_name:s}_carry.pkl", "wb") as f:
            pickle.dump(carry_numpy, f)

def load_coupled_carry(checkpoint_dir, component_names):
    checkpoint_dir = Path(checkpoint_dir)

    coupled_carry = dict()
    for component_name in component_names:
        with open(checkpoint_dir / f"{component_name:s}_carry.pkl", "rb") as f:
            coupled_carry[component_name] = jax.tree_util.tree_map(jnp.array, pickle.load(f))
    
    return coupled_carry
