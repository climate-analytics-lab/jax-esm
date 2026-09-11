"""Tests for :mod:`jem.checkpoint`.

Three properties are under test.

*A round trip is lossless, including the clock and including the dtypes.*
``CoupledCarry.step`` is the coupled model's only clock, so a checkpoint that
dropped it would resume every run in January; and the JCM physics carry holds
int and bool leaves whose meaning a silent cast to float would destroy, so
every leaf has to come back with the dtype it went in with.

*A mismatch is refused where it happens.* The format stores the leaves and a
manifest of the tree they came from, and pours them back into a template, so a
checkpoint from another grid or another component composition would otherwise
deserialise cleanly and explode much later inside a traced step. Leaf count,
leaf path, shape and dtype are all checked, and the message names the leaf;
the tree structure is checked too, because a component whose carry holds no
arrays -- an empty carry, or one belonging to a component that checkpoints
itself -- has no leaf to name and would otherwise be renamed unnoticed.

*An interrupted save is never loadable.* The carry file is written last, so a
run killed mid-save leaves a directory that :func:`latest_complete_checkpoint`
skips and :func:`load_coupled` refuses.

The Veros helpers are exercised by the experimental example drivers rather
than here: they need the Veros fork and a real ``VerosState``.
"""

import logging

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jem.base.component import CoupledCarry
from jem.checkpoint import (
    CARRY_FILENAME,
    latest_complete_checkpoint,
    load,
    load_coupled,
    remaining_batches,
    save,
    save_coupled,
)


def toy_coupled_carry(step=0):
    """Return a coupled carry with two components, in the usual carry layout.

    The ocean carries a bool and an int leaf beside its floats, because
    keeping a non-float leaf's dtype is part of the contract and a carry of
    float arrays alone would not test it.
    """
    return CoupledCarry(
        components={
            "ocn": {
                "state": {
                    "sea_surface_temperature": jnp.array([288.0, 290.5]),
                    "is_frozen": jnp.array([False, True]),
                },
                "forcing": {"total_heat_flux": jnp.array([1.5, -2.5])},
                "derived": {"steps_below_freezing": jnp.array([0, 3], dtype=jnp.int32)},
            },
            "lnd": {
                "state": {"land_surface_temperature": jnp.array([[275.0, 276.0]])},
            },
        },
        step=jnp.int32(step),
    )


def toy_component_templates():
    """Return the per-component templates ``load_coupled`` needs for the toy."""
    return dict(toy_coupled_carry().components)


@struct.dataclass
class _StaticallyConfigured:
    """A parameter struct in the shape the components use: one of each kind.

    ``depth`` is a differentiable leaf, ``method`` is static aux data JAX
    keeps in the ``PyTreeDef``. Defined here rather than imported from a
    component so that the tests below pin the *format*'s treatment of the two
    kinds, and do not break when a component's parameters change.
    """

    depth: jnp.ndarray = 60.0
    method: str = struct.field(pytree_node=False, default="none")


def assert_trees_equal(left, right):
    """Compare two pytrees leaf by leaf, exactly, dtypes included."""
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for a, b in zip(left_leaves, right_leaves, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# The generic pytree file
# ---------------------------------------------------------------------------


def test_any_pytree_round_trips_with_its_dtypes(tmp_path):
    """Floats, ints and bools all come back as they went in."""
    carry = {
        "temperature": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "counters": jnp.array([1, 2, 3], dtype=jnp.int32),
        "flags": jnp.array([True, False]),
        "scalar": jnp.int32(7),
    }

    save(carry, tmp_path / "carry.msgpack")
    assert_trees_equal(load(carry, tmp_path / "carry.msgpack"), carry)


def test_a_python_float_leaf_matches_the_array_a_scan_returns(tmp_path):
    """A component default and the carry a scan returned are the same checkpoint.

    ``initialize()`` may leave a parameter as a Python ``float`` while the
    carry coming out of ``lax.scan`` has it as a float32 array. Both spellings
    have to be one leaf, or every template built from ``initialize()`` would
    report a spurious dtype mismatch.
    """
    from_scan = {"relaxation_time": jnp.asarray(60 * 86400.0)}
    from_initialize = {"relaxation_time": 60 * 86400.0}

    save(from_scan, tmp_path / "carry.msgpack")
    loaded = load(from_initialize, tmp_path / "carry.msgpack")

    assert loaded["relaxation_time"].dtype == from_scan["relaxation_time"].dtype
    np.testing.assert_array_equal(loaded["relaxation_time"], from_scan["relaxation_time"])


def test_a_shape_mismatch_names_the_leaf(tmp_path):
    """A checkpoint from another grid is refused, by leaf path."""
    save({"state": {"sst": jnp.zeros((4, 3))}}, tmp_path / "carry.msgpack")

    with pytest.raises(ValueError, match=r"\['state'\]\['sst'\]"):
        load({"state": {"sst": jnp.zeros((8, 6))}}, tmp_path / "carry.msgpack")


def test_a_dtype_mismatch_names_the_leaf(tmp_path):
    """An int leaf read back as a float is a mismatch, not a cast."""
    save({"physics": {"convection_count": jnp.zeros(3, dtype=jnp.int32)}},
         tmp_path / "carry.msgpack")

    with pytest.raises(ValueError, match=r"\['physics'\]\['convection_count'\]"):
        load({"physics": {"convection_count": jnp.zeros(3, dtype=jnp.float32)}},
             tmp_path / "carry.msgpack")


def test_a_different_leaf_count_is_refused(tmp_path):
    """A carry from another component composition has the wrong number of leaves."""
    save({"a": jnp.zeros(2), "b": jnp.zeros(2)}, tmp_path / "carry.msgpack")

    with pytest.raises(ValueError, match="2 leaves but the model expects 3"):
        load({"a": jnp.zeros(2), "b": jnp.zeros(2), "c": jnp.zeros(2)},
             tmp_path / "carry.msgpack")


def test_a_renamed_leaf_is_refused_even_when_the_shapes_line_up(tmp_path):
    """Same leaf count, same shapes, different names is a different carry.

    Nothing but the stored leaf paths can tell these two apart, and swapping
    one component's carry for another's would otherwise resume a run with the
    fields silently exchanged.
    """
    save({"ocn": jnp.zeros(3), "lnd": jnp.ones(3)}, tmp_path / "carry.msgpack")

    with pytest.raises(ValueError, match=r"\['ice'\]"):
        load({"ice": jnp.zeros(3), "ocn": jnp.zeros(3)}, tmp_path / "carry.msgpack")


def test_a_subtree_with_no_leaves_cannot_be_renamed_unnoticed(tmp_path):
    """An empty carry leaves no leaf to compare, so only the structure catches it.

    A component with an empty carry contributes no leaf, no leaf path and no
    shape, so every leaf-level check passes and the saved leaves would be
    unflattened into whatever tree the template describes -- resuming a
    different model at the saved step. The stored ``PyTreeDef`` is what makes
    that a refusal.
    """
    save({"components": {"old": {}}, "step": jnp.int32(7)},
         tmp_path / "carry.msgpack")

    with pytest.raises(ValueError, match="pytree structure") as excinfo:
        load({"components": {"new": {}}, "step": jnp.int32(0)},
             tmp_path / "carry.msgpack")

    # Both trees are named, so the reader can see which key moved.
    assert "'old'" in str(excinfo.value)
    assert "'new'" in str(excinfo.value)


def test_a_container_that_changed_type_is_refused(tmp_path):
    """A list is not a tuple, even holding the same leaf at the same path."""
    save({"a": [jnp.zeros(2)]}, tmp_path / "carry.msgpack")

    with pytest.raises(ValueError, match="pytree structure"):
        load({"a": (jnp.zeros(2),)}, tmp_path / "carry.msgpack")


def test_a_big_structure_mismatch_is_reported_as_a_window(tmp_path):
    """Two thousand-character reprs are excerpted around the first difference.

    A whole-model carry's structure repr is far too long to read; the message
    has to point at the difference rather than print both trees in full.
    """
    saved_tree = {f"component_{index:03d}": {} for index in range(60)}
    template_tree = dict(saved_tree)
    template_tree["component_042"] = {"renamed": {}}
    save(saved_tree, tmp_path / "carry.msgpack")

    with pytest.raises(ValueError, match="first differ at character") as excinfo:
        load(template_tree, tmp_path / "carry.msgpack")

    message = str(excinfo.value)
    assert "'renamed'" in message
    assert "..." in message
    # The window, not the whole tree: the first and last components are far
    # from the difference and so are not in the message.
    assert "component_000" not in message


def test_a_changed_static_parameter_is_refused_and_named(tmp_path):
    """A ``pytree_node=False`` parameter lives in the structure, so it is checked.

    JAX keeps static parameters in the ``PyTreeDef`` rather than in the
    leaves, so recording the structure makes editing one between writing a
    checkpoint and resuming from it a refusal. That is the intended reading --
    a static parameter selects a code path, so the resumed run would be a
    different model -- and the message has to say so, because "a component was
    renamed" alone would send the reader looking for something that did not
    happen.
    """
    save({"params": _StaticallyConfigured(), "x": jnp.zeros(2)},
         tmp_path / "carry.msgpack")

    with pytest.raises(ValueError, match="static") as excinfo:
        load({"params": _StaticallyConfigured(method="qflux"), "x": jnp.zeros(2)},
             tmp_path / "carry.msgpack")

    # The difference itself is in the message, so the reader sees which
    # setting moved rather than being told only that something did.
    assert "'none'" in str(excinfo.value)
    assert "'qflux'" in str(excinfo.value)


def test_a_differentiable_parameter_is_restored_not_checked(tmp_path):
    """The other half of that contract: a leaf parameter comes back as saved.

    A tunable is a leaf, so a checkpoint holds its value and a resume
    continues from it -- editing one between runs is not a mismatch, it is
    simply overridden by the saved run.
    """
    save({"params": _StaticallyConfigured(depth=50.0)}, tmp_path / "carry.msgpack")

    loaded = load({"params": _StaticallyConfigured(depth=10.0)},
                  tmp_path / "carry.msgpack")

    assert float(loaded["params"].depth) == 50.0


def test_a_checkpoint_without_the_structure_manifest_names_what_is_missing(tmp_path):
    """A payload written before the structure was recorded is not loadable.

    Refusing it is the point: such a file cannot be checked for the very
    mismatch the manifest exists to catch, so accepting it would reintroduce
    the silent resume it was added to prevent. The message names the entry
    that is absent, which is what separates "an older checkpoint, re-run from
    the start" from "not a checkpoint at all, check the path".
    """
    (tmp_path / "old_format.msgpack").write_bytes(
        flax.serialization.msgpack_serialize(
            {"leaves": [np.zeros(2)], "leaf_paths": ["['a']"]}
        )
    )
    with pytest.raises(ValueError, match="old_format.msgpack") as excinfo:
        load({"a": jnp.zeros(2)}, tmp_path / "old_format.msgpack")

    assert "'tree_structure'" in str(excinfo.value)
    assert "'leaves'" not in str(excinfo.value)


def test_a_file_that_is_not_a_checkpoint_is_refused_by_name(tmp_path):
    """An unreadable or foreign file names itself in the error."""
    (tmp_path / "rubbish.msgpack").write_bytes(b"not msgpack at all")
    with pytest.raises(ValueError, match="rubbish.msgpack"):
        load({"a": jnp.zeros(2)}, tmp_path / "rubbish.msgpack")

    (tmp_path / "other.msgpack").write_bytes(
        flax.serialization.msgpack_serialize({"something_else": np.zeros(2)})
    )
    with pytest.raises(ValueError, match="other.msgpack"):
        load({"a": jnp.zeros(2)}, tmp_path / "other.msgpack")


def test_the_temporary_file_is_not_left_behind(tmp_path):
    """The atomic rename leaves nothing a `.tmp` glob would find."""
    save({"a": jnp.zeros(2)}, tmp_path / "carry.msgpack")
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# The coupled directory layout
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_restores_step(tmp_path):
    """Every leaf and the clock come back exactly, dtypes included.

    The step is checked by value *and* dtype: it is part of the scanned carry,
    so an int32 restored as an int64 (or a float) would make ``lax.scan``
    reject the resumed run.
    """
    carry = toy_coupled_carry(step=365)

    save_coupled(carry, tmp_path / "checkpoint")
    loaded = load_coupled(tmp_path / "checkpoint", toy_component_templates())

    assert isinstance(loaded, CoupledCarry)
    assert int(loaded.step) == 365
    assert loaded.step.dtype == jnp.int32
    assert_trees_equal(loaded, carry)


def test_a_checkpoint_without_a_carry_file_is_refused_by_name(tmp_path):
    """A directory with no carry file cannot say which step it is at."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    with pytest.raises(ValueError, match=CARRY_FILENAME):
        load_coupled(checkpoint_dir, toy_component_templates())


def test_a_component_saver_is_delegated_to(tmp_path):
    """A carry that is not a plain pytree is written, and read, by its component."""
    saved = {}

    def save_ocn(carry, directory):
        saved["directory"] = directory
        (directory / "marker").write_text("written by the component")

    def load_ocn(directory):
        return {"loaded_from": str(directory)}

    checkpoint_dir = tmp_path / "checkpoint"
    save_coupled(
        toy_coupled_carry(step=3), checkpoint_dir, component_savers={"ocn": save_ocn}
    )

    assert saved["directory"] == checkpoint_dir / "ocn"
    assert (checkpoint_dir / "ocn" / "marker").exists()

    loaded = load_coupled(
        checkpoint_dir,
        {"lnd": toy_component_templates()["lnd"]},
        component_loaders={"ocn": load_ocn},
    )
    assert loaded.components["ocn"] == {"loaded_from": str(checkpoint_dir / "ocn")}
    assert int(loaded.step) == 3


def test_a_renamed_component_with_an_empty_carry_is_refused(tmp_path):
    """A component that carries nothing is still part of the composition.

    This is the coupled-level form of the structure check: nothing in the
    leaves distinguishes a run with a component called ``old`` from one with a
    component called ``new`` when neither carries an array, so before the
    structure was recorded this resumed the wrong model at the saved step.
    """
    save_coupled(
        CoupledCarry(components={"old": {}}, step=jnp.int32(7)),
        tmp_path / "checkpoint",
    )

    with pytest.raises(ValueError, match="pytree structure"):
        load_coupled(tmp_path / "checkpoint", {"new": {}})


def test_a_renamed_component_with_leaves_is_refused_by_leaf_path(tmp_path):
    """A component that carries arrays is caught by the leaf paths, as before.

    The structure check is a catch-all behind the leaf-level checks, not a
    replacement for them: where a leaf *can* name the mismatch it still does,
    because that message is the more useful of the two.
    """
    save_coupled(toy_coupled_carry(step=1), tmp_path / "checkpoint")

    templates = toy_component_templates()
    templates["ice"] = templates.pop("ocn")
    with pytest.raises(ValueError, match=r"\['components'\]\['ice'\]"):
        load_coupled(tmp_path / "checkpoint", templates)


def test_a_renamed_delegated_component_is_refused_by_name(tmp_path):
    """Renaming a component that checkpoints itself is a composition change.

    Its carry never reaches the shared file, so only its name is there to
    check. Without that name the rename would surface -- at best -- as its
    loader failing on a directory that does not exist, and a loader that does
    not look at its directory would not notice at all.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    save_coupled(
        toy_coupled_carry(step=3),
        checkpoint_dir,
        component_savers={"ocn": lambda carry, directory: None},
    )

    with pytest.raises(ValueError, match="pytree structure") as excinfo:
        load_coupled(
            checkpoint_dir,
            {"lnd": toy_component_templates()["lnd"]},
            component_loaders={"ocean": lambda directory: {}},
        )
    assert "'ocn'" in str(excinfo.value)


def test_a_component_that_gained_a_save_state_is_refused(tmp_path):
    """"Delegated" is part of the composition, not just the component's name.

    A component that carries nothing at all and one that keeps its state in
    its own subdirectory both contribute no leaf, so the marker
    ``save_coupled`` stores for a delegated component has to be distinguishable
    from an empty carry. Otherwise a stateless component later given a
    ``save_state`` would load from a checkpoint that never wrote its
    directory, and fail inside its own loader instead of here.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    save_coupled(
        CoupledCarry(components={"ocn": None}, step=jnp.int32(2)), checkpoint_dir
    )

    with pytest.raises(ValueError, match="pytree structure"):
        load_coupled(
            checkpoint_dir, {}, component_loaders={"ocn": lambda directory: {}}
        )


def test_an_unchanged_delegated_composition_still_round_trips(tmp_path):
    """The name recorded for a delegated component costs the round trip nothing.

    The component's data stays in its own directory -- the shared file gains
    only the name -- so the loader still supplies the carry in full.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    save_coupled(
        toy_coupled_carry(step=9),
        checkpoint_dir,
        component_savers={"ocn": lambda carry, directory: None},
    )

    loaded = load_coupled(
        checkpoint_dir,
        {"lnd": toy_component_templates()["lnd"]},
        component_loaders={"ocn": lambda directory: {"restored": True}},
    )

    assert loaded.components["ocn"] == {"restored": True}
    assert_trees_equal(loaded.components["lnd"], toy_coupled_carry().components["lnd"])
    assert int(loaded.step) == 9


def test_a_component_cannot_be_both_delegated_and_templated(tmp_path):
    """Naming a component twice is a caller bug, and says so."""
    checkpoint_dir = tmp_path / "checkpoint"
    save_coupled(toy_coupled_carry(step=1), checkpoint_dir)

    with pytest.raises(ValueError, match="ocn"):
        load_coupled(
            checkpoint_dir,
            toy_component_templates(),
            component_loaders={"ocn": lambda directory: {}},
        )


def test_a_failed_overwrite_leaves_no_carry_file(tmp_path):
    """Overwriting removes the carry file first, so a partial write is refused.

    Without that ordering the old clock would survive beside freshly written
    component data, and the mixture is indistinguishable from a good
    checkpoint.
    """
    save_coupled(toy_coupled_carry(step=3), tmp_path)
    assert (tmp_path / CARRY_FILENAME).exists()

    def failing_saver(carry, directory):
        del carry, directory
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_coupled(
            toy_coupled_carry(step=4), tmp_path, component_savers={"lnd": failing_saver}
        )
    assert not (tmp_path / CARRY_FILENAME).exists()
    with pytest.raises(ValueError, match=CARRY_FILENAME):
        load_coupled(tmp_path, toy_component_templates())


def test_an_interrupted_save_is_skipped_by_latest_complete_checkpoint(tmp_path, caplog):
    """A carry-less directory sorts newest but cannot be loaded, so it is skipped.

    This is exactly the state a run killed mid-save leaves behind: the
    directory and the delegated components' data exist, but the carry file --
    written last -- does not.
    """
    save_coupled(toy_coupled_carry(step=5), tmp_path / "step_00000005")
    save_coupled(toy_coupled_carry(step=10), tmp_path / "step_00000010")
    # Killed after the component directory, before the carry file.
    interrupted = tmp_path / "step_00000015"
    save_coupled(
        toy_coupled_carry(step=15),
        interrupted,
        component_savers={"ocn": lambda carry, directory: None},
    )
    (interrupted / CARRY_FILENAME).unlink()

    with caplog.at_level(logging.WARNING, logger="jem.checkpoint"):
        latest = latest_complete_checkpoint(tmp_path)

    assert latest == tmp_path / "step_00000010"
    assert "step_00000015" in caplog.text
    assert "step_00000010" not in caplog.text

    # The returned directory is loadable, and the step a resume continues from
    # comes from inside it rather than from its name.
    assert int(load_coupled(latest, toy_component_templates()).step) == 10


def test_no_complete_checkpoint_returns_none(tmp_path, caplog):
    """With nothing loadable there is nothing to resume from -- including no root."""
    assert latest_complete_checkpoint(tmp_path / "absent") is None

    (tmp_path / "step_00000000").mkdir()
    with caplog.at_level(logging.WARNING, logger="jem.checkpoint"):
        assert latest_complete_checkpoint(tmp_path) is None
    assert "step_00000000" in caplog.text


def test_names_that_do_not_match_the_pattern_are_ignored(tmp_path, caplog):
    """Only ``pattern`` names a checkpoint, so nothing else is skipped or warned about."""
    save_coupled(toy_coupled_carry(step=2), tmp_path / "step_00000002")
    (tmp_path / "output").mkdir()
    (tmp_path / "zzz_scratch").mkdir()
    (tmp_path / "step_00000004.tmp").write_text("not a directory")

    with caplog.at_level(logging.WARNING, logger="jem.checkpoint"):
        latest = latest_complete_checkpoint(tmp_path)

    assert latest == tmp_path / "step_00000002"
    assert caplog.text == ""


# ---------------------------------------------------------------------------
# Through a real coupler
# ---------------------------------------------------------------------------


class DriftingCounter:
    """A component whose ``initialize`` leaves a parameter as a Python float.

    That is how the packaged components are written -- a tunable's default is
    a Python number -- while the carry ``lax.scan`` hands back has it as a
    float32 array. Both are the same leaf, and a checkpoint written from one
    has to load into a template built from the other.
    """

    def __init__(self, name="ocn"):
        """Name the component."""
        self.name = name

    def initialize(self):
        return {"params": {"rate": 0.5}, "state": {"value": jnp.float32(0.0)}}

    def step(self, carry, time):
        del time
        value = carry["state"]["value"] + carry["params"]["rate"]
        new_carry = dict(carry, state={"value": value})
        return new_carry, {"value": value}


def test_a_coupled_run_resumes_from_its_own_checkpoint(tmp_path):
    """Two steps, a checkpoint, two more: the same run as four in one go.

    ``Coupler.load_state`` builds its template from the components'
    ``initialize()``, so this is also the test that a Python-float default and
    the float32 array a scan returns are one leaf.
    """
    from jem.base.coupler import Coupler

    def build():
        import jax_datetime as jdt

        return Coupler(
            {"ocn": DriftingCounter()},
            coupling_timestep=jdt.to_timedelta(1, "day"),
            start_date=jdt.to_datetime("2001-01-01"),
        )

    model = build()
    initial = model.initialize()
    continuous, _ = model.generate_trajectory_function(4)(initial)

    two = model.generate_trajectory_function(2)
    carry, _ = two(initial)
    model.save_state(carry, tmp_path / "checkpoint")

    # A fresh model, as a resumed process would build: nothing of the run is
    # carried over except the checkpoint itself.
    resumed_model = build()
    loaded = resumed_model.load_state(tmp_path / "checkpoint")
    assert int(loaded.step) == 2
    resumed, _ = resumed_model.generate_trajectory_function(2)(loaded)

    assert_trees_equal(resumed, continuous)


# ---------------------------------------------------------------------------
# How much of a chunked run is left
# ---------------------------------------------------------------------------


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
