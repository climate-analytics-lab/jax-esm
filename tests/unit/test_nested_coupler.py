"""A ``Coupler`` used as a component of a slower ``Coupler``.

The pattern under test is the GFDL one: an atmosphere and a land model
exchanging fluxes on a fast loop inside a slower ocean coupling. JAX-ESM
expresses it in two ways -- one coupler whose workflow repeats the fast block,
and a fast coupler registered as a component of a slow one -- and the point of
this module is that both are ordinary uses of the same contract and that they
produce the same run, down to the last bit.

The toys are deliberately arithmetic: a component counts up by one plus
whatever an exchanger handed it, and reports the clock it ran on, so the number
of calls, the rate of each clock and the path each field took are all visible
in the output.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr

from jem.base.component import (
    Component,
    CoupledCarry,
    SupportsBind,
    SupportsCheckpoint,
    SupportsXarray,
)
from jem.base.coupler import Coupler, nested_carry, with_nested_carry
from jem.utils.checkpoints import (
    COUPLED_STEP_FILENAME,
    load_coupled_carry,
    save_coupled_carry,
)

DAY = 86400.0
HOUR = 3600.0
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")
FAST_TIMESTEP = jdt.to_timedelta(1, "hour")
START_DATE = jdt.to_datetime("2001-01-01")


class Counter:
    """Counts up by one plus what it was given, and reports its clock."""

    def __init__(self, name):
        """Name the component."""
        self.name = name

    def initialize(self):
        return {"value": jnp.float32(0.0), "received": jnp.float32(0.0)}

    def step(self, carry, time):
        new_carry = dict(carry, value=carry["value"] + 1.0 + carry["received"])
        return new_carry, {
            "value": new_carry["value"],
            "sim_time": time.sim_time,
            "dt": jnp.float32(time.dt),
        }

    def to_xarray(self, diagnostics, time):
        return xr.Dataset(
            {
                "value": ("time", np.asarray(diagnostics["value"])),
                "sim_time": ("time", np.asarray(diagnostics["sim_time"])),
            },
            coords={"time": time.datetimes()},
        )


class CheckpointingCounter(Counter):
    """A ``Counter`` that writes its own carry, as ``VerosComponent`` does.

    The point of the toy is the *path*: it records every directory it is
    handed, so a test can show that the coupler passed it the directory named
    after it, inside the one named after the coupler it lives in. Its format
    is deliberately not a pickle, so a checkpoint that fell back to the
    default pytree path would be visible as a missing file rather than as a
    file that happens to work.
    """

    FILENAME = "custom_format.json"

    def __init__(self, name):
        """Name the component and start with no recorded directories."""
        super().__init__(name)
        self.saved_directories = []
        self.loaded_directories = []

    def save_state(self, carry, directory):
        self.saved_directories.append(directory)
        (directory / self.FILENAME).write_text(
            json.dumps({key: float(value) for key, value in carry.items()})
        )

    def load_state(self, directory):
        self.loaded_directories.append(directory)
        stored = json.loads((directory / self.FILENAME).read_text())
        return {key: jnp.float32(value) for key, value in stored.items()}


def atm_lnd_exchange(components, time):
    """Give the land what the atmosphere last produced (the fast loop)."""
    del time
    return dict(
        components,
        lnd=dict(components["lnd"], received=components["atm"]["value"]),
    )


def srf_ocn_exchange_flat(components, time):
    """Trade the land's value for the ocean's, with every carry at one level."""
    del time
    ocean = dict(components["ocn"], received=components["lnd"]["value"])
    atmosphere = dict(components["atm"], received=components["ocn"]["value"])
    return dict(components, ocn=ocean, atm=atmosphere)


def srf_ocn_exchange_nested(components, time):
    """Trade the same fields, reaching into the nested coupler's carry.

    This is what an exchanger in the outer coupler has to do: the fast model
    is one entry in the carries mapping, holding a whole ``CoupledCarry``.
    """
    del time
    land = nested_carry(components, "atm_lnd", "lnd")
    atmosphere = nested_carry(components, "atm_lnd", "atm")
    components = with_nested_carry(
        components,
        "atm_lnd",
        "atm",
        dict(atmosphere, received=components["ocn"]["value"]),
    )
    return dict(
        components, ocn=dict(components["ocn"], received=land["value"])
    )


def fast_coupler(**kwargs):
    """Build the hourly atmosphere/land model, on its own."""
    return Coupler(
        {"atm": Counter("atm"), "lnd": Counter("lnd")},
        {"atm_lnd_exchange": atm_lnd_exchange},
        coupling_timestep=FAST_TIMESTEP,
        start_date=START_DATE,
        name="atm_lnd",
        **kwargs,
    )


def nested_model():
    """Build the daily model whose surface is the hourly coupler."""
    return Coupler(
        {"atm_lnd": fast_coupler(), "ocn": Counter("ocn")},
        {"srf_ocn_exchange": srf_ocn_exchange_nested},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=["srf_ocn_exchange", "atm_lnd", "ocn"],
    )


def checkpointing_nested_model():
    """Build the nested model with an inner component that checkpoints itself.

    Returns the outer coupler and the inner one, so a test can reach the
    component that recorded the directories it was given.
    """
    inner = Coupler(
        {"atm": Counter("atm"), "lnd": CheckpointingCounter("lnd")},
        {"atm_lnd_exchange": atm_lnd_exchange},
        coupling_timestep=FAST_TIMESTEP,
        start_date=START_DATE,
        name="atm_lnd",
    )
    outer = Coupler(
        {"atm_lnd": inner, "ocn": Counter("ocn")},
        {"srf_ocn_exchange": srf_ocn_exchange_nested},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=["srf_ocn_exchange", "atm_lnd", "ocn"],
    )
    return outer, inner


def flat_model():
    """Build the same model as one coupler with a repeated workflow."""
    return Coupler(
        {"atm": Counter("atm"), "lnd": Counter("lnd"), "ocn": Counter("ocn")},
        {
            "atm_lnd_exchange": atm_lnd_exchange,
            "srf_ocn_exchange": srf_ocn_exchange_flat,
        },
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=[
            "srf_ocn_exchange",
            ["atm_lnd_exchange", "atm", "lnd"] * 24,
            "ocn",
        ],
    )


def assert_trees_equal(left, right):
    """Compare two pytrees leaf by leaf, exactly."""
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# A coupler satisfies the component contract
# ---------------------------------------------------------------------------


def test_a_coupler_is_a_component():
    """No wrapper class: a Coupler has the members the protocols require."""
    coupler = fast_coupler()
    assert isinstance(coupler, Component)
    assert isinstance(coupler, SupportsBind)
    assert isinstance(coupler, SupportsXarray)
    # It checkpoints itself too, which is what lets an outer coupler hand it
    # its own subdirectory instead of pickling its whole inner carry.
    assert isinstance(coupler, SupportsCheckpoint)
    assert coupler.name == "atm_lnd"
    assert Coupler(
        {}, coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE
    ).name == "coupled"


class ParameterizedCounter(Counter):
    """A ``Counter`` that can start from parameters it is handed.

    It stands for a slab model with an initial-condition parameter: the value
    the run starts from is read once, by ``initialize``.
    """

    def __init__(self, name, start=0.0):
        """Name the component and set the value it defaults to starting from."""
        super().__init__(name)
        self.params = jnp.float32(start)

    def initialize(self, params=None):
        params = self.params if params is None else params
        return {"value": jnp.asarray(params), "received": jnp.float32(0.0)}


def test_initialize_routes_params_through_a_nested_coupler():
    """The value for a nested coupler is a mapping over *its* components."""
    inner = Coupler(
        {"atm": ParameterizedCounter("atm"), "lnd": Counter("lnd")},
        {"atm_lnd_exchange": atm_lnd_exchange},
        coupling_timestep=FAST_TIMESTEP,
        start_date=START_DATE,
        name="atm_lnd",
    )
    outer = Coupler(
        {"atm_lnd": inner, "ocn": ParameterizedCounter("ocn")},
        {"srf_ocn_exchange": srf_ocn_exchange_nested},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=["srf_ocn_exchange", "atm_lnd", "ocn"],
    )

    carry = outer.initialize({"atm_lnd": {"atm": jnp.float32(7.0)}, "ocn": jnp.float32(3.0)})

    inner_carry = carry.components["atm_lnd"]
    assert float(inner_carry.components["atm"]["value"]) == 7.0
    # The inner component the inner mapping does not name is untouched, as is
    # the inner coupler's own step counter.
    assert float(inner_carry.components["lnd"]["value"]) == 0.0
    assert int(inner_carry.step) == 0
    assert float(carry.components["ocn"]["value"]) == 3.0


def test_initialize_of_a_nested_coupler_rejects_an_unknown_inner_name():
    """The inner coupler validates its own mapping, and says which name it is."""
    inner = Coupler(
        {"atm": ParameterizedCounter("atm")},
        coupling_timestep=FAST_TIMESTEP,
        start_date=START_DATE,
        name="atm_lnd",
    )
    outer = Coupler(
        {"atm_lnd": inner},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )

    with pytest.raises(ValueError, match=r"\['ocn'\].*'atm_lnd'"):
        outer.initialize({"atm_lnd": {"ocn": jnp.float32(1.0)}})


def test_registering_a_coupler_binds_it():
    coupler = fast_coupler()
    assert coupler.outer_ratio is None
    Coupler(
        {"atm_lnd": coupler},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    assert coupler.outer_ratio == 24


def test_bind_refuses_a_timestep_that_does_not_divide():
    with pytest.raises(ValueError, match="whole multiple"):
        Coupler(
            {"atm_lnd": fast_coupler()},
            coupling_timestep=jdt.to_timedelta(90, "minute"),
            start_date=START_DATE,
        )


def test_bind_refuses_a_different_start_date_or_calendar():
    with pytest.raises(ValueError, match="Start-date mismatch"):
        Coupler(
            {"atm_lnd": fast_coupler()},
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=jdt.to_datetime("2002-01-01"),
        )
    with pytest.raises(ValueError, match="Calendar mismatch"):
        Coupler(
            {"atm_lnd": fast_coupler()},
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=START_DATE,
            calendar="gregorian",
        )


def test_rebinding_is_a_no_op_for_the_same_clock_and_refused_for_another():
    coupler = fast_coupler()
    coupler.bind(
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE, calendar="365_day"
    )
    coupler.bind(
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE, calendar="365_day"
    )
    assert coupler.outer_ratio == 24

    with pytest.raises(ValueError, match="already bound"):
        coupler.bind(
            coupling_timestep=jdt.to_timedelta(2, "day"),
            start_date=START_DATE,
            calendar="365_day",
        )
    assert coupler.outer_ratio == 24


def test_stepping_an_unbound_coupler_as_a_component_is_an_error():
    coupler = fast_coupler()
    with pytest.raises(RuntimeError, match="has not been bound"):
        coupler.step(coupler.initialize(), coupler.coupling_time(0))


def test_a_clock_from_another_coupler_is_refused():
    """Only the static fields can be checked, and they are enough."""
    coupler = fast_coupler()
    coupler.bind(
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE, calendar="365_day"
    )
    other = Coupler(
        {"ocn": Counter("ocn")},
        coupling_timestep=jdt.to_timedelta(2, "day"),
        start_date=START_DATE,
    )
    with pytest.raises(ValueError, match="dt="):
        coupler.step(coupler.initialize(), other.coupling_time(0))


def test_a_nested_coupler_of_equal_timestep_adds_no_axis():
    """``r == 1`` mirrors multiplicity 1: one inner step, no extra axis."""
    inner = Coupler(
        {"atm": Counter("atm")},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        name="surface",
    )
    outer = Coupler(
        {"surface": inner},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    assert inner.outer_ratio == 1

    carry, diagnostics = outer.generate_trajectory_function(3)(outer.initialize())

    assert diagnostics["surface"]["atm"]["value"].shape == (3,)
    assert int(carry.components["surface"].step) == 3
    datasets = outer.to_xarray(diagnostics)
    assert set(datasets) == {"atm"}
    assert datasets["atm"].sizes["time"] == 3


# ---------------------------------------------------------------------------
# The GFDL pattern: an hourly surface inside a daily coupling
# ---------------------------------------------------------------------------


def test_the_inner_components_advance_once_per_inner_step():
    """24 hourly inner steps per outer step, on a continuous inner clock."""
    model = nested_model()
    carry, diagnostics = model.generate_trajectory_function(2)(model.initialize())

    atmosphere = diagnostics["atm_lnd"]["atm"]
    # (outer steps, inner steps per outer step, ...)
    assert atmosphere["value"].shape == (2, 24)
    np.testing.assert_allclose(np.asarray(atmosphere["dt"]), np.full((2, 24), HOUR))
    np.testing.assert_allclose(
        np.asarray(atmosphere["sim_time"]).ravel(), np.arange(48) * HOUR
    )
    # The ocean is on the outer clock.
    np.testing.assert_allclose(np.asarray(diagnostics["ocn"]["dt"]), [DAY, DAY])

    # Two clocks, both in the carry: the outer counts coupled days, the inner
    # counts its own hours.
    assert int(carry.step) == 2
    assert int(carry.components["atm_lnd"].step) == 48


def test_to_xarray_flattens_the_inner_datasets_onto_the_inner_axis():
    model = nested_model()
    _, diagnostics = model.generate_trajectory_function(2)(model.initialize())

    datasets = model.to_xarray(diagnostics)

    # The nested coupler's own registered name does not appear; its
    # components' names do.
    assert set(datasets) == {"atm", "lnd", "ocn"}
    hourly = np.datetime64("2001-01-01", "ns") + np.arange(1, 49) * np.timedelta64(
        1, "h"
    )
    assert datasets["atm"].sizes["time"] == 48
    np.testing.assert_array_equal(datasets["atm"].time.values, hourly)
    np.testing.assert_array_equal(datasets["lnd"].time.values, hourly)
    np.testing.assert_array_equal(
        datasets["ocn"].time.values,
        np.array(["2001-01-02", "2001-01-03"], dtype="datetime64[ns]"),
    )


def test_to_xarray_of_a_chunk_labels_the_inner_axis_from_the_outer_step():
    model = nested_model()
    _, diagnostics = model.generate_trajectory_function(2)(model.initialize())

    datasets = model.to_xarray(diagnostics, first_step=2)

    hourly = np.datetime64("2001-01-01", "ns") + np.arange(49, 97) * np.timedelta64(
        1, "h"
    )
    np.testing.assert_array_equal(datasets["atm"].time.values, hourly)
    np.testing.assert_array_equal(
        datasets["ocn"].time.values,
        np.array(["2001-01-04", "2001-01-05"], dtype="datetime64[ns]"),
    )


def test_to_xarray_takes_a_time_axis_or_a_first_step_but_not_both():
    model = nested_model()
    _, diagnostics = model.generate_trajectory_function(1)(model.initialize())
    with pytest.raises(ValueError, match="not both"):
        model.to_xarray(diagnostics, model.time_axis(0, 1), first_step=1)


def test_a_name_collision_between_two_components_output_is_refused():
    """Flattening must not silently overwrite one component's output."""
    model = Coupler(
        {"atm_lnd": fast_coupler(), "lnd": Counter("lnd")},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    _, diagnostics = model.generate_trajectory_function(1)(model.initialize())
    with pytest.raises(ValueError, match="already written"):
        model.to_xarray(diagnostics)


# ---------------------------------------------------------------------------
# The two ways of writing the same model
# ---------------------------------------------------------------------------


def test_nested_and_flat_forms_are_the_same_run():
    """The nested pair and the repeated workflow agree bit for bit."""
    nested = nested_model()
    flat = flat_model()

    nested_carry_out, nested_diagnostics = nested.generate_trajectory_function(3)(
        nested.initialize()
    )
    flat_carry, flat_diagnostics = flat.generate_trajectory_function(3)(
        flat.initialize()
    )

    # Same state, component by component (the nested form holds two of them
    # one level down, which is the only difference between the two carries).
    assert int(nested_carry_out.step) == int(flat_carry.step) == 3
    for name in ("atm", "lnd"):
        assert_trees_equal(
            nested_carry_out.components["atm_lnd"].components[name],
            flat_carry.components[name],
        )
        assert_trees_equal(
            nested_diagnostics["atm_lnd"][name], flat_diagnostics[name]
        )
    assert_trees_equal(nested_carry_out.components["ocn"], flat_carry.components["ocn"])
    assert_trees_equal(nested_diagnostics["ocn"], flat_diagnostics["ocn"])

    # And the same output, including the time axes.
    nested_datasets = nested.to_xarray(nested_diagnostics)
    flat_datasets = flat.to_xarray(flat_diagnostics)
    assert set(nested_datasets) == set(flat_datasets) == {"atm", "lnd", "ocn"}
    for name, dataset in nested_datasets.items():
        xr.testing.assert_identical(dataset, flat_datasets[name])

    # The run is not trivially zero: the exchanges moved something.
    assert float(flat_datasets["ocn"].value.values[-1]) > 3.0


def test_checkpoint_round_trip_of_a_nested_run(tmp_path):
    """The inner CoupledCarry pickles like any other carry, and both clocks resume."""
    model = nested_model()
    initial = model.initialize()
    continuous_carry, _ = model.generate_trajectory_function(4)(initial)

    two = model.generate_trajectory_function(2)
    carry, _ = two(initial)
    save_coupled_carry(carry, tmp_path / "checkpoint")
    loaded = load_coupled_carry(tmp_path / "checkpoint", model.components)

    assert isinstance(loaded.components["atm_lnd"], CoupledCarry)
    assert int(loaded.step) == 2
    assert int(loaded.components["atm_lnd"].step) == 48

    resumed, diagnostics = two(loaded)
    assert_trees_equal(resumed, continuous_carry)
    # The inner clock continues from hour 48, not from zero.
    np.testing.assert_allclose(
        np.asarray(diagnostics["atm_lnd"]["atm"]["sim_time"]).ravel(),
        np.arange(48, 96) * HOUR,
    )


# ---------------------------------------------------------------------------
# Checkpointing the coupled model through the components' own capabilities
# ---------------------------------------------------------------------------


def test_save_state_hands_each_component_its_own_directory(tmp_path):
    """A component that writes itself is delegated to, however deeply nested.

    The savers are derived from the components, so the driver names none of
    them: the outer coupler delegates the inner coupled carry to the inner
    coupler, which in turn delegates its land to the component that knows how
    to write it.
    """
    outer, inner = checkpointing_nested_model()
    land = inner.components["lnd"]
    carry, _ = outer.generate_trajectory_function(2)(outer.initialize())

    root = tmp_path / "checkpoint"
    outer.save_state(carry, root)

    assert land.saved_directories == [root / "atm_lnd" / "lnd"]
    assert (root / "atm_lnd" / "lnd" / CheckpointingCounter.FILENAME).exists()
    # Its sibling has an ordinary pytree carry and is still pickled, next to
    # it, under the name the default path uses.
    assert (root / "atm_lnd" / "atm_carry.pkl").exists()
    assert (root / "ocn_carry.pkl").exists()
    # The delegating saver replaces the pickle rather than accompanying it,
    # for the component and for the nested coupler alike.
    assert not (root / "atm_lnd" / "lnd_carry.pkl").exists()
    assert not (root / "atm_lnd_carry.pkl").exists()
    # Both coupled models are complete checkpoints in their own right: the
    # inner one wrote its own clock and its own completion marker.
    assert (root / COUPLED_STEP_FILENAME).exists()
    assert (root / "atm_lnd" / COUPLED_STEP_FILENAME).exists()


def test_load_state_round_trips_a_nested_checkpoint(tmp_path):
    """The loaders are derived the same way, so the carry comes back whole."""
    outer, inner = checkpointing_nested_model()
    land = inner.components["lnd"]
    carry, _ = outer.generate_trajectory_function(2)(outer.initialize())

    root = tmp_path / "checkpoint"
    outer.save_state(carry, root)
    loaded = outer.load_state(root)

    assert land.loaded_directories == [root / "atm_lnd" / "lnd"]
    assert isinstance(loaded, CoupledCarry)
    assert isinstance(loaded.components["atm_lnd"], CoupledCarry)
    assert_trees_equal(loaded, carry)
    # Both clocks, spelled out: the outer counts coupled days, the inner its
    # own hours, and neither is reconstructed from the other.
    assert int(loaded.step) == 2
    assert int(loaded.components["atm_lnd"].step) == 48
    assert loaded.step.dtype == jnp.int32
    assert loaded.components["atm_lnd"].step.dtype == jnp.int32


def test_a_nested_run_resumed_from_load_state_continues_identically(tmp_path):
    """Four steps, or two then a checkpoint then two, are the same run."""
    outer, _ = checkpointing_nested_model()
    initial = outer.initialize()
    continuous_carry, _ = outer.generate_trajectory_function(4)(initial)

    two = outer.generate_trajectory_function(2)
    carry, _ = two(initial)
    outer.save_state(carry, tmp_path / "checkpoint")

    resumed, diagnostics = two(outer.load_state(tmp_path / "checkpoint"))

    assert_trees_equal(resumed, continuous_carry)
    # The inner clock continues from hour 48, not from zero.
    np.testing.assert_allclose(
        np.asarray(diagnostics["atm_lnd"]["atm"]["sim_time"]).ravel(),
        np.arange(48, 96) * HOUR,
    )


def test_save_state_matches_the_explicit_helpers_for_an_all_pytree_model(tmp_path):
    """With nothing to delegate to, the capability writes the same checkpoint.

    The flat model has no component with a checkpoint capability of its own,
    so ``Coupler.save_state`` derives an empty savers mapping and is exactly
    the call a driver used to write by hand.
    """
    model = flat_model()
    carry, _ = model.generate_trajectory_function(2)(model.initialize())

    model.save_state(carry, tmp_path / "capability")
    save_coupled_carry(carry, tmp_path / "explicit")

    assert sorted(path.name for path in (tmp_path / "capability").iterdir()) == sorted(
        path.name for path in (tmp_path / "explicit").iterdir()
    )
    assert_trees_equal(
        model.load_state(tmp_path / "capability"),
        load_coupled_carry(tmp_path / "explicit", model.components),
    )


# ---------------------------------------------------------------------------
# The helpers an exchanger reaches into a nested carry with
# ---------------------------------------------------------------------------


def test_nested_carry_reads_through_the_inner_coupled_carry():
    model = nested_model()
    carries = model.initialize().components
    assert (
        nested_carry(carries, "atm_lnd", "atm")
        is carries["atm_lnd"].components["atm"]
    )


def test_with_nested_carry_replaces_immutably():
    model = nested_model()
    carries = model.initialize().components
    before = carries["atm_lnd"]
    replacement = dict(nested_carry(carries, "atm_lnd", "lnd"), received=jnp.float32(7.0))

    updated = with_nested_carry(carries, "atm_lnd", "lnd", replacement)

    assert float(nested_carry(updated, "atm_lnd", "lnd")["received"]) == 7.0
    # Nothing the coupler is carrying was touched.
    assert carries["atm_lnd"] is before
    assert float(nested_carry(carries, "atm_lnd", "lnd")["received"]) == 0.0
    assert updated["ocn"] is carries["ocn"]
    assert isinstance(updated["atm_lnd"], CoupledCarry)
    assert int(updated["atm_lnd"].step) == int(before.step)
    # The untouched sibling is the same object, not a copy.
    assert nested_carry(updated, "atm_lnd", "atm") is nested_carry(
        carries, "atm_lnd", "atm"
    )


def test_the_helpers_say_so_when_the_carry_is_not_a_nested_model():
    model = nested_model()
    carries = model.initialize().components
    with pytest.raises(TypeError, match="does not hold a nested coupled model"):
        nested_carry(carries, "ocn", "atm")
    with pytest.raises(TypeError, match="does not hold a nested coupled model"):
        with_nested_carry(carries, "ocn", "atm", {})


def test_with_nested_carry_survives_a_new_coupled_carry_field():
    """It replaces a field rather than rebuilding the struct positionally."""
    model = nested_model()
    carries = model.initialize().components
    updated = with_nested_carry(
        carries, "atm_lnd", "atm", nested_carry(carries, "atm_lnd", "atm")
    )
    assert dataclasses.fields(updated["atm_lnd"]) == dataclasses.fields(
        carries["atm_lnd"]
    )


def test_an_inner_carry_from_another_point_in_the_run_is_reported(caplog):
    """The inner step counter is checked against the outer one every step.

    A nested checkpoint paired with an outer step from elsewhere in the run
    would date the inner components differently from the rest of the model;
    the mismatch is logged at ERROR (not raised: the check runs inside the
    outer scan), and a consistent run logs nothing.
    """
    import logging

    model = nested_model()
    run = model.generate_trajectory_function(2)

    with caplog.at_level(logging.ERROR, logger="jem.base.coupler"):
        run(model.initialize())
    assert not [r for r in caplog.records if "nested clock" in r.getMessage()]

    carry = model.initialize()
    inner = carry.components["atm_lnd"]
    skewed = dataclasses.replace(
        carry,
        components={**carry.components,
                    "atm_lnd": dataclasses.replace(inner, step=inner.step + 24)},
    )
    with caplog.at_level(logging.ERROR, logger="jem.base.coupler"):
        run(skewed)
    messages = [r.getMessage() for r in caplog.records if "nested clock" in r.getMessage()]
    # 24 inner steps ahead of outer step 0, whose consistent inner step is 0.
    assert any("inner step 24" in m and "inner step 0" in m for m in messages)
