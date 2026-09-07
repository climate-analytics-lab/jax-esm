"""Tests for ``jem.base.coupler``.

Everything here runs on toy components defined in this module: a source whose
value counts up, a sink that accumulates what it is given, and a clock
watcher that reports the ``CouplingTime`` it saw. They make the coupler's own
behaviour - the clock, the workflow order, the immutability of the carry, the
scan and the output plumbing - visible in exact integers, which a real slab
or atmosphere component would bury under physics.

The exchanger here rebuilds the dicts it is handed rather than assigning into
them, which is what every exchanger must do: the coupler hands out the carries
of a `lax.scan` and cannot tolerate them being written to in place.
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr

from jem.base.component import CoupledCarry, TimeAxis
from jem.base.coupler import Coupler

DAY = 86400.0
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")
START_DATE = jdt.to_datetime("2001-01-01")


# ---------------------------------------------------------------------------
# Toy components and exchangers
# ---------------------------------------------------------------------------


class SourceComponent:
    """Counts up by one per coupled step; the thing the sink is coupled to."""

    def __init__(self, name="source"):
        """Name the component."""
        self.name = name

    def initialize(self):
        return {"value": jnp.float32(0.0)}

    def step(self, carry, time):
        del time
        new_carry = {"value": carry["value"] + 1.0}
        return new_carry, {"value": new_carry["value"]}


class SinkComponent:
    """Accumulates whatever an exchanger has put in ``received``."""

    def __init__(self, name="sink"):
        """Name the component."""
        self.name = name

    def initialize(self):
        return {"received": jnp.float32(0.0), "total": jnp.float32(0.0)}

    def step(self, carry, time):
        del time
        new_carry = dict(carry, total=carry["total"] + carry["received"])
        return new_carry, {"received": carry["received"], "total": new_carry["total"]}


class ClockWatcher:
    """Reports the clock it was handed, so tests can see what a component sees."""

    def __init__(self, name="clock"):
        """Name the component."""
        self.name = name

    def initialize(self):
        return {"sim_time": jnp.float32(0.0)}

    def step(self, carry, time):
        del carry
        return (
            {"sim_time": time.sim_time},
            {
                "sim_time": time.sim_time,
                "step": time.step,
                "year_fraction": time.year_fraction,
            },
        )


class DampedComponent:
    """A smooth nonlinear step, so a gradient through it is worth checking."""

    def __init__(self, name="damped"):
        """Name the component."""
        self.name = name

    def initialize(self):
        return {"value": jnp.array([1.0, 2.0], dtype=jnp.float32)}

    def step(self, carry, time):
        del time
        value = 0.9 * carry["value"] + 0.1 * jnp.sin(carry["value"])
        return {"value": value}, {"value": value}


class XarrayComponent(SourceComponent):
    """A source that can label its own output; records the axis it was given."""

    def __init__(self, name="source"):
        """Record every time axis handed to `to_xarray`."""
        super().__init__(name=name)
        self.time_axes = []

    def to_xarray(self, diagnostics, time):
        self.time_axes.append(time)
        return xr.Dataset({"value": ("time", np.asarray(diagnostics["value"]))})


class ClockRecorder:
    """Records the clock of every call it receives and writes it out.

    Unlike `ClockWatcher` it also reports `dt`, which is what makes a
    sub-stepped call distinguishable from a coupled one, and it can label its
    own output, so the time axis a repeated component is given is visible.
    """

    def __init__(self, name="recorder"):
        """Name the component and start with no recorded time axes."""
        self.name = name
        self.time_axes = []

    def initialize(self):
        return {"calls": jnp.int32(0)}

    def step(self, carry, time):
        return (
            {"calls": carry["calls"] + 1},
            {
                "step": time.step,
                "sim_time": time.sim_time,
                "dt": jnp.float32(time.dt),
                "year_fraction": time.year_fraction,
            },
        )

    def to_xarray(self, diagnostics, time):
        self.time_axes.append(time)
        return xr.Dataset(
            {"sim_time": ("time", np.asarray(diagnostics["sim_time"]))},
            coords={"time": time.datetimes()},
        )


class BindRecorder(SourceComponent):
    """A `SupportsBind` component that records every clock it is bound to."""

    def __init__(self, name="bound"):
        """Name the component and start unbound."""
        super().__init__(name=name)
        self.binds = []

    def bind(self, *, coupling_timestep, start_date, calendar):
        self.binds.append(
            {
                "coupling_timestep": coupling_timestep,
                "start_date": start_date,
                "calendar": calendar,
            }
        )


def feed(components, time):
    """Copy the source's value into the sink's ``received`` field."""
    del time
    sink = dict(components["sink"], received=components["source"]["value"])
    return dict(components, sink=sink)


def _coupler(workflow=None, **kwargs):
    """Build the standard source/sink toy coupler."""
    return Coupler(
        {"source": SourceComponent(), "sink": SinkComponent()},
        {"feed": feed},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=workflow,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction, workflow validation and repr
# ---------------------------------------------------------------------------


def test_default_workflow_is_exchangers_then_components():
    """Exchange first, then every component, each in registration order."""
    assert _coupler().workflow == ("feed", "source", "sink")


def test_explicit_workflow_is_used():
    coupler = _coupler(workflow=["source", "feed", "sink"])
    assert coupler.workflow == ("source", "feed", "sink")


def test_unknown_workflow_name_raises():
    with pytest.raises(ValueError, match="ocean"):
        _coupler(workflow=["feed", "ocean"])


def test_duplicate_name_raises():
    """A name may be a component or an exchanger, not both."""
    with pytest.raises(ValueError, match="source"):
        Coupler(
            {"source": SourceComponent()},
            {"source": feed},
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=START_DATE,
        )


def test_workflow_revalidated_when_a_component_is_removed():
    """A workflow that was valid at construction is checked again at trace time."""
    coupler = _coupler(workflow=["feed", "source", "sink"])
    coupler.remove_component("sink")
    with pytest.raises(ValueError, match="sink"):
        coupler.step_function()


def test_repr_names_the_model():
    text = repr(_coupler())
    assert "source" in text
    assert "feed" in text
    assert "365_day" in text


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


def test_initialize_starts_at_step_zero():
    carry = _coupler().initialize()
    assert isinstance(carry, CoupledCarry)
    assert int(carry.step) == 0
    assert set(carry.components) == {"source", "sink"}


def test_clock_persists_across_trajectory_calls():
    """Two five-step calls continue the run; the scan index is not the clock."""
    coupler = Coupler(
        {"clock": ClockWatcher()},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    trajectory = coupler.generate_trajectory_function(5)

    carry = coupler.initialize()
    carry, first = trajectory(carry)
    carry, second = trajectory(carry)

    assert int(carry.step) == 10
    np.testing.assert_allclose(first["clock"]["sim_time"], np.arange(0, 5) * DAY)
    np.testing.assert_allclose(second["clock"]["sim_time"], np.arange(5, 10) * DAY)

    # The next step - the one the persisted counter is for - sees 10 * dt.
    _, tenth = coupler.step_function()(carry)
    assert float(tenth["clock"]["sim_time"]) == pytest.approx(10 * DAY)


def test_components_share_clock():
    """Two components in the same step see the identical time."""
    coupler = Coupler(
        {"first": ClockWatcher("first"), "second": ClockWatcher("second")},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    _, diagnostics = coupler.generate_trajectory_function(4)(coupler.initialize())

    np.testing.assert_array_equal(
        diagnostics["first"]["sim_time"], diagnostics["second"]["sim_time"]
    )
    np.testing.assert_array_equal(diagnostics["first"]["step"], np.arange(4))


def test_year_fraction_wraps():
    """The annual cycle wraps at the year end rather than running past 1."""
    # A start date one day before the year end on a 365-day calendar: the run
    # starts at 364/365 through the year and step 1 is New Year's Day. 2000 is
    # a Gregorian leap year, so a real-calendar day count would say 365 days
    # and wrap a day early; the offset is counted in the model calendar.
    coupler = Coupler(
        {"clock": ClockWatcher()},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=jdt.to_datetime("2000-12-31"),
        calendar="365_day",
    )
    assert float(coupler.coupling_time(0).year_fraction) == pytest.approx(364 / 365)
    assert float(coupler.coupling_time(1).year_fraction) == pytest.approx(0.0, abs=1e-6)
    assert float(coupler.coupling_time(2).year_fraction) == pytest.approx(
        1 / 365, rel=1e-6
    )


def test_year_fraction_uses_the_calendar_year_length():
    """The Gregorian calendar's 365.2425-day year is used when it is selected."""
    coupler = Coupler(
        {"clock": ClockWatcher()},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar="gregorian",
    )
    assert coupler.days_per_year == pytest.approx(365.2425)
    assert float(coupler.coupling_time(1).year_fraction) == pytest.approx(
        1 / 365.2425, rel=1e-5
    )


def test_clock_facts_are_exposed():
    coupler = _coupler()
    assert coupler.dt_seconds == DAY
    assert coupler.calendar == "365_day"
    assert coupler.days_per_year == 365.0
    assert coupler.year_offset_seconds == 0.0
    assert coupler.coupling_timestep == COUPLING_TIMESTEP


def test_time_axis_starts_at_the_requested_step():
    axis = _coupler().time_axis(7, 3)
    assert isinstance(axis, TimeAxis)
    np.testing.assert_array_equal(axis.steps, [7, 8, 9])
    assert len(axis) == 3
    assert axis.calendar == "365_day"


# ---------------------------------------------------------------------------
# The step function
# ---------------------------------------------------------------------------


def test_step_does_not_mutate_input():
    """A step rebuilds the carry; the one it was given is still the old one."""
    coupler = _coupler()
    carry = coupler.initialize()
    components_before = carry.components
    structure_before = jax.tree_util.tree_structure(carry)
    leaves_before = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(carry)]

    new_carry, _ = coupler.step_function()(carry)

    assert carry.components is components_before
    assert jax.tree_util.tree_structure(carry) == structure_before
    for before, after in zip(leaves_before, jax.tree_util.tree_leaves(carry), strict=True):
        np.testing.assert_array_equal(before, np.asarray(after))
    assert int(carry.step) == 0
    assert int(new_carry.step) == 1


def test_exchanger_runs_before_components_by_default():
    """With the default workflow the sink sees the value the source had last step."""
    _, diagnostics = _coupler().generate_trajectory_function(3)(_coupler().initialize())

    np.testing.assert_allclose(diagnostics["source"]["value"], [1.0, 2.0, 3.0])
    # Lagged by one coupled step: the exchange at step n moves what the source
    # produced during step n-1, and step 0 exchanges the initialized value.
    np.testing.assert_allclose(diagnostics["sink"]["received"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(diagnostics["sink"]["total"], [0.0, 1.0, 3.0])


def test_workflow_order_is_respected():
    """Running the exchanger between the two components removes the lag ..."""
    coupler = _coupler(workflow=["source", "feed", "sink"])
    _, diagnostics = coupler.generate_trajectory_function(3)(coupler.initialize())
    np.testing.assert_allclose(diagnostics["sink"]["received"], [1.0, 2.0, 3.0])

    # ... and a component placed before the exchanger sees the un-exchanged
    # value, one further step behind.
    coupler = _coupler(workflow=["sink", "feed", "source"])
    _, diagnostics = coupler.generate_trajectory_function(3)(coupler.initialize())
    np.testing.assert_allclose(diagnostics["sink"]["received"], [0.0, 0.0, 1.0])


def test_treedef_change_raises():
    """An exchanger that changes the carry structure is named, not left to scan."""

    def adds_a_key(components, time):
        del time
        sink = dict(components["sink"], extra=jnp.float32(0.0))
        return dict(components, sink=sink)

    coupler = Coupler(
        {"source": SourceComponent(), "sink": SinkComponent()},
        {"adds_a_key": adds_a_key},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    with pytest.raises(RuntimeError, match="adds_a_key"):
        coupler.step_function()(coupler.initialize())


def test_exchanger_must_return_a_mapping():
    """Returning something other than the carries dict is a clear error."""

    def returns_nothing(components, time):
        del components, time

    coupler = Coupler(
        {"source": SourceComponent()},
        {"returns_nothing": returns_nothing},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    with pytest.raises(TypeError, match="returns_nothing"):
        coupler.step_function()(coupler.initialize())


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------


def _concatenate(*chunks):
    """Join per-chunk diagnostics along their leading time axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)


def _assert_carries_close(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)


def test_continuous_equals_chunked():
    """Ten steps, five twice and two five times give the same run."""
    coupler = _coupler()
    initial = coupler.initialize()

    continuous_carry, continuous = coupler.generate_trajectory_function(10)(initial)

    five = coupler.generate_trajectory_function(5)
    carry, first = five(initial)
    carry, second = five(carry)
    _assert_carries_close(carry, continuous_carry)
    _assert_carries_close(_concatenate(first, second), continuous)

    two = coupler.generate_trajectory_function(2)
    carry = initial
    chunks = []
    for _ in range(5):
        carry, chunk = two(carry)
        chunks.append(chunk)
    _assert_carries_close(carry, continuous_carry)
    _assert_carries_close(_concatenate(*chunks), continuous)

    assert int(continuous_carry.step) == 10


def test_jit_false_matches_jit_true():
    coupler = _coupler()
    initial = coupler.initialize()
    jitted_carry, jitted = coupler.generate_trajectory_function(4, jit=True)(initial)
    eager_carry, eager = coupler.generate_trajectory_function(4, jit=False)(initial)

    _assert_carries_close(jitted_carry, eager_carry)
    _assert_carries_close(jitted, eager)


def test_remat_matches_plain_trajectory():
    """`remat` only trades memory for recomputation; the numbers are unchanged."""
    coupler = _coupler()
    initial = coupler.initialize()
    plain_carry, plain = coupler.generate_trajectory_function(4)(initial)
    remat_carry, remat = coupler.generate_trajectory_function(4, remat=True)(initial)

    _assert_carries_close(plain_carry, remat_carry)
    _assert_carries_close(plain, remat)


def test_diagnostics_have_a_leading_time_axis():
    coupler = _coupler()
    _, diagnostics = coupler.generate_trajectory_function(6)(coupler.initialize())
    assert diagnostics["source"]["value"].shape == (6,)


def test_gradient_through_a_trajectory_matches_finite_differences():
    """The coupled trajectory is differentiable end to end."""
    coupler = Coupler(
        {"damped": DampedComponent()},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    trajectory = coupler.generate_trajectory_function(3)
    initial = coupler.initialize()

    def objective(value):
        carry = CoupledCarry(components={"damped": {"value": value}}, step=initial.step)
        final, _ = trajectory(carry)
        return jnp.sum(final.components["damped"]["value"])

    value = initial.components["damped"]["value"]
    gradient = np.asarray(jax.grad(objective)(value))
    assert np.all(np.isfinite(gradient))

    epsilon = 1e-2
    for index in range(value.shape[0]):
        shift = jnp.zeros_like(value).at[index].set(epsilon)
        finite_difference = float(
            (objective(value + shift) - objective(value - shift)) / (2 * epsilon)
        )
        assert gradient[index] == pytest.approx(finite_difference, abs=1e-4)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_to_xarray_uses_time_axis():
    """Each capable component is handed the run's own time axis, and only it."""
    component = XarrayComponent()
    coupler = Coupler(
        {"source": component, "sink": SinkComponent()},
        {"feed": feed},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    _, diagnostics = coupler.generate_trajectory_function(3)(coupler.initialize())

    datasets = coupler.to_xarray(diagnostics, first_step=5)

    # The sink cannot write output, so it simply has none.
    assert set(datasets) == {"source"}
    assert isinstance(datasets["source"], xr.Dataset)

    (axis,) = component.time_axes
    assert isinstance(axis, TimeAxis)
    np.testing.assert_array_equal(axis.steps, [5, 6, 7])
    assert axis.start_date == START_DATE
    assert axis.dt == COUPLING_TIMESTEP


def test_to_xarray_skips_components_without_diagnostics():
    component = XarrayComponent()
    coupler = Coupler(
        {"source": component},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    assert coupler.to_xarray({}) == {}


def test_step_function_snapshots_the_model():
    """A generated step describes the model as it was; later edits do not leak in."""
    coupler = _coupler()
    carry = coupler.initialize()
    step = coupler.step_function()

    coupler.remove_component("sink")

    _, diagnostics = step(carry)
    assert set(diagnostics) == {"source", "sink"}


@pytest.mark.parametrize("seconds", [0, -86400])
def test_non_positive_coupling_timestep_is_rejected(seconds):
    """A slab-only coupler has no component that would refuse a zero or negative step."""
    with pytest.raises(ValueError, match="coupling_timestep must be positive"):
        Coupler(
            {"source": SourceComponent()},
            coupling_timestep=jdt.to_timedelta(seconds, "second"),
            start_date=START_DATE,
        )


# ---------------------------------------------------------------------------
# Nested workflows and multiplicity
# ---------------------------------------------------------------------------

HOUR = 3600.0


def _hourly_coupler():
    """Build a daily coupler in which ``fast`` runs hourly and ``slow`` daily."""
    return Coupler(
        {"fast": ClockRecorder("fast"), "slow": ClockRecorder("slow")},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=[["fast"] * 24, "slow"],
    )


def _assert_trees_equal(left, right):
    """Compare two pytrees leaf by leaf, exactly."""
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_nested_workflow_is_flattened():
    """A workflow may be written nested; what runs is the flat sequence."""
    coupler = _coupler(workflow=[["feed", "source"], ["sink"]])
    assert coupler.workflow == ("feed", "source", "sink")


def test_nested_workflow_with_repeats_flattens_to_the_running_order():
    coupler = _coupler(workflow=[["feed", "source"] * 2, "sink"])
    assert coupler.workflow == ("feed", "source", "feed", "source", "sink")
    assert coupler.multiplicities() == {"feed": 2, "source": 2, "sink": 1}


@pytest.mark.parametrize("workflow", [["feed", 3], [["feed", None]], ["feed", {"sink"}]])
def test_non_string_workflow_leaf_raises(workflow):
    """Only strings are leaves; anything else would be iterated by accident."""
    with pytest.raises(TypeError, match="strings"):
        _coupler(workflow=workflow)


def test_unknown_name_in_a_nested_workflow_raises():
    with pytest.raises(ValueError, match="ocean"):
        _coupler(workflow=[["feed", "source"], ["ocean"]])


def test_a_repeated_component_runs_on_a_faster_clock():
    """24 calls per coupled step, each on its own hour of the day."""
    coupler = _hourly_coupler()
    _, diagnostics = coupler.generate_trajectory_function(2)(coupler.initialize())

    fast = diagnostics["fast"]
    # The extra leading axis is the multiplicity, inside the scanned steps.
    assert fast["sim_time"].shape == (2, 24)
    np.testing.assert_allclose(np.asarray(fast["dt"]), np.full((2, 24), HOUR))
    # Hourly and continuous across the coupled step boundary.
    np.testing.assert_allclose(
        np.asarray(fast["sim_time"]).ravel(), np.arange(48) * HOUR
    )
    np.testing.assert_array_equal(np.asarray(fast["step"]).ravel(), np.arange(48))
    # The seasonal cycle is still reduced in exact integer arithmetic at the
    # sub-rate: an hour divides a 365-day year 8760 times.
    np.testing.assert_allclose(
        np.asarray(fast["year_fraction"]).ravel(), np.arange(48) / 8760, rtol=1e-6
    )

    # A component listed once still gets the coupled daily clock.
    slow = diagnostics["slow"]
    assert slow["sim_time"].shape == (2,)
    np.testing.assert_allclose(np.asarray(slow["dt"]), [DAY, DAY])
    np.testing.assert_allclose(np.asarray(slow["sim_time"]), [0.0, DAY])


def test_a_repeated_component_is_bound_with_its_own_timestep():
    """A bindable component is bound once, with the step it actually advances by."""
    fast = BindRecorder("fast")
    slow = BindRecorder("slow")
    Coupler(
        {"fast": fast, "slow": slow},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=[["fast"] * 24, "slow"],
    )

    assert len(fast.binds) == 1
    assert fast.binds[0]["coupling_timestep"] == jdt.to_timedelta(1, "hour")
    assert fast.binds[0]["start_date"] == START_DATE
    assert fast.binds[0]["calendar"] == "365_day"

    assert len(slow.binds) == 1
    assert slow.binds[0]["coupling_timestep"] == COUPLING_TIMESTEP


def test_a_component_the_workflow_omits_is_never_bound_or_run():
    """A count of zero means no clock and no step, not a silent daily one."""
    unused = BindRecorder("unused")
    coupler = Coupler(
        {"unused": unused, "source": SourceComponent()},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=["source"],
    )
    assert unused.binds == []

    _, diagnostics = coupler.step_function()(coupler.initialize())
    assert set(diagnostics) == {"source"}


def test_a_component_added_after_construction_is_bound_with_the_full_timestep():
    """It is not in the explicit workflow, so it has no faster clock to adopt."""
    component = BindRecorder()
    coupler = _coupler(workflow=["feed", "source", "sink"])
    coupler.add_component("bound", component)
    assert len(component.binds) == 1
    assert component.binds[0]["coupling_timestep"] == COUPLING_TIMESTEP


def test_indivisible_multiplicity_is_rejected():
    """A day does not divide into 7 whole seconds' worth of sub-steps."""
    with pytest.raises(ValueError, match="'source' appears 7 times"):
        _coupler(workflow=["feed", ["source"] * 7, "sink"])


def test_repr_collapses_a_repeated_block():
    text = repr(_coupler(workflow=[["feed", "source"] * 3, "sink"]))
    assert "['feed', 'source'] * 3" in text
    assert "'sink'" in text


# ---------------------------------------------------------------------------
# Multiplicity: output
# ---------------------------------------------------------------------------


def test_to_xarray_labels_a_repeated_component_at_the_sub_rate():
    """24 hourly records per coupled step, stamped at the end of each hour."""
    coupler = _hourly_coupler()
    _, diagnostics = coupler.generate_trajectory_function(2)(coupler.initialize())

    datasets = coupler.to_xarray(diagnostics)

    hourly = np.datetime64("2001-01-01", "ns") + (
        np.arange(1, 49) * np.timedelta64(1, "h")
    )
    assert datasets["fast"].sizes["time"] == 48
    np.testing.assert_array_equal(datasets["fast"].time.values, hourly)
    # The records are in run order: the flattened sub-step clock.
    np.testing.assert_allclose(
        datasets["fast"].sim_time.values, np.arange(48) * HOUR
    )

    assert datasets["slow"].sizes["time"] == 2
    np.testing.assert_array_equal(
        datasets["slow"].time.values,
        np.array(["2001-01-02", "2001-01-03"], dtype="datetime64[ns]"),
    )

    axis = coupler.components["fast"].time_axes[-1]
    assert axis.dt == jdt.to_timedelta(1, "hour")
    np.testing.assert_array_equal(axis.steps, np.arange(48))


def test_to_xarray_first_step_is_in_coupled_steps_for_every_component():
    """A chunked run labels the fast component from its own sub-step count."""
    coupler = _hourly_coupler()
    _, diagnostics = coupler.generate_trajectory_function(2)(coupler.initialize())

    datasets = coupler.to_xarray(diagnostics, first_step=2)

    hourly = np.datetime64("2001-01-01", "ns") + (
        np.arange(49, 97) * np.timedelta64(1, "h")
    )
    np.testing.assert_array_equal(datasets["fast"].time.values, hourly)
    np.testing.assert_array_equal(
        datasets["slow"].time.values,
        np.array(["2001-01-04", "2001-01-05"], dtype="datetime64[ns]"),
    )


def test_to_xarray_rejects_diagnostics_that_are_not_the_run_s():
    """Diagnostics without the multiplicity axis cannot be labelled."""
    coupler = _hourly_coupler()
    single = coupler.step_function()(coupler.initialize())[1]
    # One step's diagnostics are (24, ...), not (steps, 24, ...).
    with pytest.raises(ValueError, match="runs 24 times"):
        coupler.to_xarray(single)


# ---------------------------------------------------------------------------
# Multiplicity: nothing changes when every name appears once
# ---------------------------------------------------------------------------


def test_a_workflow_without_multiplicity_is_the_run_it_always_was():
    """Flat, nested-but-single and default workflows are the same model."""
    default = _coupler()
    flat = _coupler(workflow=["feed", "source", "sink"])
    nested = _coupler(workflow=[["feed"], ["source", "sink"]])
    assert nested.workflow == flat.workflow == default.workflow

    runs = [
        coupler.generate_trajectory_function(3)(coupler.initialize())
        for coupler in (default, flat, nested)
    ]
    for carry, diagnostics in runs[1:]:
        _assert_trees_equal(carry, runs[0][0])
        _assert_trees_equal(diagnostics, runs[0][1])
    # No extra axis, and the diagnostics are the component's own pytree.
    assert runs[0][1]["source"]["value"].shape == (3,)


def test_the_substep_clock_of_a_single_element_is_the_coupled_clock():
    """`multiplicity == 1` is not a special case of the sub-step arithmetic."""
    coupler = _coupler()
    coupled = coupler.coupling_time(3)
    substep = coupler.coupling_time_at_substep(3, 0, 1)
    assert int(substep.step) == int(coupled.step)
    assert float(substep.sim_time) == float(coupled.sim_time)
    assert substep.dt == coupled.dt
    assert substep.year_offset_seconds == coupled.year_offset_seconds
    assert substep.days_per_year == coupled.days_per_year


def test_a_repeated_exchanger_sees_the_sub_stepped_clock():
    """An exchanger may be repeated too, and is told which sub-step it is on."""
    seen = []

    def record(components, time):
        seen.append((float(time.sim_time), time.dt, int(time.step)))
        return components

    coupler = Coupler(
        {"fast": ClockRecorder("fast")},
        {"record": record},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=[["record", "fast"] * 4],
    )
    step = coupler.step_function()

    carry, _ = step(coupler.initialize())
    assert [entry[1] for entry in seen] == [DAY / 4] * 4
    assert [entry[0] for entry in seen] == [0.0, 21600.0, 43200.0, 64800.0]
    assert [entry[2] for entry in seen] == [0, 1, 2, 3]

    # The next coupled step continues the sub-step count, because it is
    # derived from the coupled counter in the carry.
    seen.clear()
    step(carry)
    assert [entry[2] for entry in seen] == [4, 5, 6, 7]


def test_checkpoint_round_trip_of_a_run_with_multiplicity(tmp_path):
    """The carry counts COUPLED steps, so a resumed run continues both clocks."""
    from jem.utils.checkpoints import load_coupled_carry, save_coupled_carry

    coupler = _hourly_coupler()
    initial = coupler.initialize()
    continuous_carry, continuous = coupler.generate_trajectory_function(4)(initial)

    two = coupler.generate_trajectory_function(2)
    carry, first = two(initial)
    save_coupled_carry(carry, tmp_path / "checkpoint")
    loaded = load_coupled_carry(tmp_path / "checkpoint", coupler.components)

    # Two coupled steps, not 48 sub-steps: the counter is the coupled clock.
    assert int(loaded.step) == 2

    resumed, second = two(loaded)
    _assert_carries_close(resumed, continuous_carry)
    _assert_carries_close(_concatenate(first, second), continuous)
    np.testing.assert_allclose(
        np.asarray(second["fast"]["sim_time"]).ravel(), np.arange(48, 96) * HOUR
    )
