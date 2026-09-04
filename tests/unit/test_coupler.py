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


def test_repeated_workflow_entry_raises():
    """Each element runs exactly once per coupled step."""
    with pytest.raises(ValueError, match="more than once"):
        _coupler(workflow=["feed", "source", "source"])


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
    # starts at 364/365 through the year and step 1 is New Year's Day. 2001 is
    # not a leap year, so the real-calendar offset jax_datetime computes and
    # the model's 365-day year agree (see `_seconds_since_new_year`).
    coupler = Coupler(
        {"clock": ClockWatcher()},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=jdt.to_datetime("2001-12-31"),
        calendar="365_day",
    )
    assert float(coupler.coupling_time(0).year_fraction) == pytest.approx(364 / 365)
    assert float(coupler.coupling_time(1).year_fraction) == pytest.approx(0.0, abs=1e-6)
    # Loose tolerance on purpose: `year_fraction` divides a large second count
    # by the year length in float32, so just after a wrap the surviving
    # precision is that of the *large* number, ~1e-7 relative to 1.0.
    assert float(coupler.coupling_time(2).year_fraction) == pytest.approx(
        1 / 365, rel=1e-4
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
