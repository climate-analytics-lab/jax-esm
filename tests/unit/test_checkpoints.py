"""Tests for ``jem.utils.checkpoints``.

These cover the plain-pytree path only: the Veros helpers in the same module
need the Veros fork and a real ``VerosState``, and are exercised by the
experimental example drivers rather than here.

The property under test is that a checkpoint round-trip is lossless *including
the clock*. ``CoupledCarry.step`` is the coupled model's only clock, so a
checkpoint that dropped it would resume every run in January.
"""

import logging
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from jem.base.component import CoupledCarry
from jem.utils.checkpoints import (
    COUPLED_STEP_FILENAME,
    latest_complete_checkpoint,
    load_component_carries,
    load_coupled_carry,
    remaining_batches,
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


def test_a_failed_overwrite_leaves_no_completion_marker(tmp_path):
    """Overwriting a checkpoint removes the step file first, so a partial write is refused."""
    save_coupled_carry(toy_coupled_carry(step=3), tmp_path)
    assert (tmp_path / COUPLED_STEP_FILENAME).exists()

    def failing_saver(carry, directory):
        del carry, directory
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_coupled_carry(
            toy_coupled_carry(step=4), tmp_path, component_savers={"lnd": failing_saver}
        )
    assert not (tmp_path / COUPLED_STEP_FILENAME).exists()
    with pytest.raises(ValueError, match=COUPLED_STEP_FILENAME):
        load_coupled_carry(tmp_path, ["ocn", "lnd"])


def test_a_failed_marker_write_leaves_no_marker(tmp_path, monkeypatch):
    """The marker is published atomically, so it never exists half-written.

    `latest_complete_checkpoint` and `load_coupled_carry` both trust the marker's
    existence alone, so a marker truncated by a kill or a full disk would be
    accepted as a complete checkpoint and then fail to unpickle on resume.
    """
    real_dump = pickle.dump

    def fail_on_the_step(obj, file):
        # The step is the only bare array written; the component carries are
        # dicts, and they have to succeed for the save to reach the marker.
        if isinstance(obj, np.ndarray):
            # Write something first: a marker truncated part-way through is
            # exactly the state this must not leave behind.
            file.write(b"half a pickle")
            raise OSError("disk full")
        return real_dump(obj, file)

    monkeypatch.setattr(pickle, "dump", fail_on_the_step)
    with pytest.raises(OSError, match="disk full"):
        save_coupled_carry(toy_coupled_carry(step=6), tmp_path / "checkpoint")

    # The component carries were written, so the failure really was the
    # marker's own write and not an earlier one.
    assert (tmp_path / "checkpoint" / "ocn_carry.pkl").exists()
    assert not (tmp_path / "checkpoint" / COUPLED_STEP_FILENAME).exists()
    # Nor is the scratch file it wrote through left lying around.
    assert list((tmp_path / "checkpoint").glob("*.tmp")) == []


def test_the_newest_incomplete_checkpoint_is_skipped(tmp_path, caplog):
    """A marker-less directory sorts newest but cannot be loaded, so it is skipped.

    This is the state a run killed mid-save leaves behind: the directory and
    some component carries exist, but the completion marker -- written last --
    does not.
    """
    save_coupled_carry(toy_coupled_carry(step=5), tmp_path / "step_00000005")
    save_coupled_carry(toy_coupled_carry(step=10), tmp_path / "step_00000010")
    (tmp_path / "step_00000015").mkdir()

    with caplog.at_level(logging.WARNING, logger="jem.utils.checkpoints"):
        latest = latest_complete_checkpoint(tmp_path)

    assert latest == tmp_path / "step_00000010"
    assert "step_00000015" in caplog.text
    assert "step_00000010" not in caplog.text

    # The returned directory is loadable, and the step a resume continues from
    # comes from inside it rather than from its name.
    assert int(load_coupled_carry(latest, ["ocn", "lnd"]).step) == 10


def test_no_complete_checkpoint_returns_none(tmp_path, caplog):
    """With nothing loadable there is nothing to resume from -- including no root."""
    assert latest_complete_checkpoint(tmp_path / "absent") is None

    (tmp_path / "step_00000000").mkdir()
    with caplog.at_level(logging.WARNING, logger="jem.utils.checkpoints"):
        assert latest_complete_checkpoint(tmp_path) is None
    assert "step_00000000" in caplog.text


def test_names_that_do_not_match_the_pattern_are_ignored(tmp_path, caplog):
    """Only ``pattern`` names a checkpoint, so nothing else is skipped or warned about."""
    save_coupled_carry(toy_coupled_carry(step=2), tmp_path / "step_00000002")
    (tmp_path / "output").mkdir()
    (tmp_path / "zzz_scratch").mkdir()
    (tmp_path / "step_00000004.tmp").write_text("not a directory")

    with caplog.at_level(logging.WARNING, logger="jem.utils.checkpoints"):
        latest = latest_complete_checkpoint(tmp_path)

    assert latest == tmp_path / "step_00000002"
    assert caplog.text == ""


def test_remaining_batches_runs_out_the_full_batches_then_the_remainder():
    """A run of 20 steps in batches of 6 is three full batches and a short one."""
    assert remaining_batches(0, 20, 6) == [6, 6, 6, 2]
    # A total that is a whole number of batches has no short one.
    assert remaining_batches(0, 18, 6) == [6, 6, 6]


def test_remaining_batches_resumes_from_the_step_not_from_a_batch_index():
    """The batch length may differ from the run that wrote the checkpoint.

    This is the case a batch index cannot express: a run that got to step 10 in
    batches of 5 is resumed with batches of 2, and still has exactly the 10
    steps between it and a 20-step total left to do.
    """
    assert remaining_batches(10, 20, 2) == [2, 2, 2, 2, 2]
    # ... and the same restart asked for 10-step batches runs the one batch it
    # needs rather than exiting as "already done".
    assert remaining_batches(10, 20, 10) == [10]
    # A restart part-way through a batch is not special: what is left is what
    # is left, whether or not it divides evenly.
    assert remaining_batches(7, 20, 5) == [5, 5, 3]


def test_remaining_batches_is_empty_when_the_run_is_done():
    """Nothing left to run -- including a total that a longer run passed."""
    assert remaining_batches(20, 20, 5) == []
    assert remaining_batches(25, 20, 5) == []
    assert remaining_batches(0, 0, 5) == []


@pytest.mark.parametrize(
    "steps_done, total_steps, steps_per_batch",
    [(0, 10, 0), (0, 10, -1), (-1, 10, 5), (0, -10, 5)],
    ids=["zero_batch", "negative_batch", "negative_done", "negative_total"],
)
def test_remaining_batches_rejects_nonsense_counts(
    steps_done, total_steps, steps_per_batch
):
    """A zero-length batch would loop forever; a negative count is a caller bug."""
    with pytest.raises(ValueError):
        remaining_batches(steps_done, total_steps, steps_per_batch)
