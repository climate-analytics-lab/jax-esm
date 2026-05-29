from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def save_coupled_carry(coupled_carry, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    atm_carry_numpy = jax.tree_util.tree_map(np.array, coupled_carry["atm"])
    with open(checkpoint_dir / "atm_carry.pkl", "wb") as f:
        pickle.dump(atm_carry_numpy, f)

    ocn_carry_numpy = jax.tree_util.tree_map(np.array, coupled_carry["ocn"])
    with open(checkpoint_dir / "ocn_carry.pkl", "wb") as f:
        pickle.dump(ocn_carry_numpy, f)

    co2_carry_numpy = jax.tree_util.tree_map(np.array, coupled_carry["co2_atm_boxmodel"])
    with open(checkpoint_dir / "co2_atm_boxmodel_carry.pkl", "wb") as f:
        pickle.dump(co2_carry_numpy, f)


def load_coupled_carry(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / "atm_carry.pkl", "rb") as f:
        atm_carry = pickle.load(f)
    atm_carry = jax.tree_util.tree_map(jnp.array, atm_carry)

    with open(checkpoint_dir / "ocn_carry.pkl", "rb") as f:
        ocn_carry = pickle.load(f)
    ocn_carry = jax.tree_util.tree_map(jnp.array, ocn_carry)

    with open(checkpoint_dir / "co2_atm_boxmodel_carry.pkl", "rb") as f:
        co2_carry = pickle.load(f)
    co2_carry = jax.tree_util.tree_map(jnp.array, co2_carry)

    return dict(
        atm=atm_carry,
        ocn=ocn_carry,
        co2_atm_boxmodel=co2_carry,
    )
