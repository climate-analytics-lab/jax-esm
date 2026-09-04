"""Tests for ``jem.utils.checkpoints``.

These cover the plain-pytree path only: the Veros helpers in the same module
need the Veros fork and a real ``VerosState``, and are exercised by the
experimental example drivers rather than here.

The property under test is that a checkpoint round-trip is lossless *including
the clock*. ``CoupledCarry.step`` is the coupled model's only clock, so a
checkpoint that dropped it would resume every run in January.
"""

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from jem.base.component import CoupledCarry
from jem.utils.checkpoints import (
    COUPLED_STEP_FILENAME,
    load_component_carries,
    load_coupled_carry,
    save_component_carries,
    save_coupled_carry,
)


def toy_coupled_carry(step=0):
    """Return a coupled carry with two components, in the usual carry layout."""
    return CoupledCarry(
        components={
            "ocn": {
                "state": {"sea_surface_temperature": jnp.array([288.0, 290.5])},
                "forcing": {"total_heat_flux": jnp.array([1.5, -2.5])},
            },
            "lnd": {
                "state": {"land_surface_temperature": jnp.array([[275.0, 276.0]])},
            },
        },
        step=jnp.int32(step),
    )


def test_round_trip_preserves_the_component_carries(tmp_path):
    """Every leaf of every component's carry comes back unchanged."""
    carry = toy_coupled_carry(step=7)

    save_coupled_carry(carry, tmp_path / "checkpoint")
    loaded = load_coupled_carry(tmp_path / "checkpoint", ["ocn", "lnd"])

    assert set(loaded.components) == {"ocn", "lnd"}
    np.testing.assert_array_equal(
        loaded.components["ocn"]["state"]["sea_surface_temperature"],
        carry.components["ocn"]["state"]["sea_surface_temperature"],
    )
    np.testing.assert_array_equal(
        loaded.components["ocn"]["forcing"]["total_heat_flux"],
        carry.components["ocn"]["forcing"]["total_heat_flux"],
    )
    np.testing.assert_array_equal(
        loaded.components["lnd"]["state"]["land_surface_temperature"],
        carry.components["lnd"]["state"]["land_surface_temperature"],
    )


def test_round_trip_preserves_the_coupled_step(tmp_path):
    """The clock survives, so a resumed run continues its seasonal cycle."""
    carry = toy_coupled_carry(step=365)

    save_coupled_carry(carry, tmp_path / "checkpoint")
    loaded = load_coupled_carry(tmp_path / "checkpoint", ["ocn", "lnd"])

    assert isinstance(loaded, CoupledCarry)
    assert int(loaded.step) == 365
    # The step is part of the scanned carry, so its dtype has to match the one
    # `Coupler.initialize` produces or `lax.scan` rejects the resumed run.
    assert loaded.step.dtype == jnp.int32


def test_a_checkpoint_without_a_step_is_refused_by_name(tmp_path):
    """A pre-clock checkpoint raises rather than silently resuming at step 0."""
    checkpoint_dir = tmp_path / "checkpoint"
    carry = toy_coupled_carry(step=42)
    # What the old format wrote: the component carries and nothing else.
    save_component_carries(carry.components, checkpoint_dir)

    with pytest.raises(ValueError, match=COUPLED_STEP_FILENAME):
        load_coupled_carry(checkpoint_dir, ["ocn", "lnd"])

    # The escape hatch the message points at still works.
    components = load_component_carries(checkpoint_dir, ["ocn", "lnd"])
    assert set(components) == {"ocn", "lnd"}


def test_a_component_saver_is_delegated_to(tmp_path):
    """A carry that is not a plain pytree is written by its own component."""
    saved = {}

    def save_ocn(carry, directory):
        saved["directory"] = directory
        (directory / "marker").write_text("written by the component")

    def load_ocn(directory):
        return {"loaded_from": str(directory)}

    checkpoint_dir = tmp_path / "checkpoint"
    save_coupled_carry(
        toy_coupled_carry(step=3), checkpoint_dir,
        component_savers={"ocn": save_ocn},
    )

    assert saved["directory"] == checkpoint_dir / "ocn"
    assert (checkpoint_dir / "ocn" / "marker").exists()
    # The delegating saver replaces the pickle, it does not accompany it.
    assert not (checkpoint_dir / "ocn_carry.pkl").exists()

    loaded = load_coupled_carry(
        checkpoint_dir, ["ocn", "lnd"], component_loaders={"ocn": load_ocn},
    )
    assert loaded.components["ocn"] == {"loaded_from": str(checkpoint_dir / "ocn")}
    assert int(loaded.step) == 3


def test_the_step_file_holds_a_plain_numpy_scalar(tmp_path):
    """The checkpoint is readable without jax, as the pickled carries are."""
    save_coupled_carry(toy_coupled_carry(step=11), tmp_path / "checkpoint")

    with open(tmp_path / "checkpoint" / COUPLED_STEP_FILENAME, "rb") as f:
        step = pickle.load(f)

    assert isinstance(step, np.ndarray)
    assert int(step) == 11
