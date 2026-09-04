"""Tests for the component contract in ``jem.base.component``.

These cover what the coupler *requires* of a component and what it may
optionally use: the required protocol, the three optional capabilities, the
clock handed to a component that asks for it at registration, and the output
time axis every component labels its diagnostics with.

The components here are deliberately trivial toys. A slab or JCM component
would drag a grid and boundary data into a test of the contract, and the
contract has nothing to do with either.
"""

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest

from jem.base.component import (
    Component,
    SupportsBind,
    SupportsCheckpoint,
    SupportsXarray,
    TimeAxis,
)
from jem.base.coupler import Coupler
from jem.utils.time import time_coordinate

COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")
START_DATE = jdt.to_datetime("2001-01-01")


class MinimalComponent:
    """The smallest thing the coupler accepts: a name, initialize and step."""

    def __init__(self, name="minimal"):
        """Name the component; the coupler keys everything else by the dict key."""
        self.name = name

    def initialize(self):
        return {"value": jnp.zeros(())}

    def step(self, carry, time):
        del time
        return carry, {"value": carry["value"]}


class ComponentWithoutStep:
    """Satisfies everything except ``step`` - the coupler must refuse it."""

    def __init__(self, name="no_step"):
        """Name the component."""
        self.name = name

    def initialize(self):
        return {"value": jnp.zeros(())}


class ComponentWithXarray(MinimalComponent):
    """A component that can label its own diagnostics."""

    def to_xarray(self, diagnostics, time):
        del diagnostics, time
        return None


class ComponentWithCheckpoint(MinimalComponent):
    """A component whose carry it saves and loads itself (the Veros case)."""

    def save_state(self, carry, directory):
        del carry, directory

    def load_state(self, directory):
        del directory


class ComponentWithBind(MinimalComponent):
    """A component that records the clock it is bound to."""

    def __init__(self, name="bound"):
        """Start unbound, so a test can tell binding apart from construction."""
        super().__init__(name=name)
        self.bound = None

    def bind(self, *, coupling_timestep, start_date, calendar):
        self.bound = {
            "coupling_timestep": coupling_timestep,
            "start_date": start_date,
            "calendar": calendar,
        }


def test_component_protocol_accepts_a_minimal_component():
    """A name, an initialize and a step are the whole required contract."""
    assert isinstance(MinimalComponent(), Component)


def test_component_protocol_rejects_missing_step():
    """A component without ``step`` is refused, and the error names what is missing."""
    incomplete = ComponentWithoutStep()
    assert not isinstance(incomplete, Component)

    with pytest.raises(TypeError, match="step"):
        Coupler(
            {"broken": incomplete},
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=START_DATE,
        )


def test_optional_capabilities_detected():
    """The optional capabilities are recognised by ``isinstance``, one by one."""
    plain = MinimalComponent()
    assert not isinstance(plain, SupportsXarray)
    assert not isinstance(plain, SupportsCheckpoint)
    assert not isinstance(plain, SupportsBind)

    assert isinstance(ComponentWithXarray(), SupportsXarray)
    assert not isinstance(ComponentWithXarray(), SupportsBind)

    assert isinstance(ComponentWithCheckpoint(), SupportsCheckpoint)
    assert not isinstance(ComponentWithCheckpoint(), SupportsXarray)

    assert isinstance(ComponentWithBind(), SupportsBind)
    assert not isinstance(ComponentWithBind(), SupportsXarray)

    # An optional capability never affects the required contract.
    for component in (ComponentWithXarray(), ComponentWithCheckpoint(), ComponentWithBind()):
        assert isinstance(component, Component)


def test_bind_receives_clock():
    """Registration hands a binding component the coupler's clock definition."""
    component = ComponentWithBind()
    Coupler(
        {"bound": component},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar="365_day",
    )

    assert component.bound is not None
    assert component.bound["coupling_timestep"] == COUPLING_TIMESTEP
    assert component.bound["start_date"] == START_DATE
    assert component.bound["calendar"] == "365_day"


def test_bind_is_called_for_components_added_later():
    """A component registered after construction is bound too, or it has no clock."""
    component = ComponentWithBind()
    coupler = Coupler(
        {},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    assert component.bound is None

    coupler.add_component("bound", component)
    assert component.bound is not None
    assert component.bound["coupling_timestep"] == COUPLING_TIMESTEP


def test_registered_component_is_the_object_passed_in():
    """There is no wrapper: the coupler holds the user's own object."""
    component = MinimalComponent()
    coupler = Coupler(
        {"minimal": component},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    assert coupler.components["minimal"] is component


# ---------------------------------------------------------------------------
# The output time axis
# ---------------------------------------------------------------------------


def test_datetimes_label_the_end_of_each_interval():
    """Record ``k`` holds the interval that ENDS at ``start + (k+1) dt``."""
    axis = TimeAxis(
        start_date=jdt.to_datetime("2001-01-01"),
        steps=np.arange(3),
        dt=COUPLING_TIMESTEP,
        calendar="365_day",
    )
    np.testing.assert_array_equal(
        axis.datetimes(),
        np.array(["2001-01-02", "2001-01-03", "2001-01-04"], dtype="datetime64[ns]"),
    )
    assert axis.datetimes().dtype == np.dtype("datetime64[ns]")


def test_datetimes_reproduce_jcm_arithmetic_bit_for_bit():
    """The labels are JCM's float64-days product, not an exact ns count.

    Both models have to be inexact in the SAME way for ``xr.merge`` to align
    them, so this pins the arithmetic and not just the answer.
    """
    start = jdt.to_datetime("2001-03-01")
    steps = np.arange(5)
    axis = TimeAxis(start, steps, jdt.to_timedelta(6, "hour"), "365_day")

    nanoseconds_per_day = np.timedelta64(1, "D") / np.timedelta64(1, "ns")
    start_days = float(np.asarray(start.delta.days))
    expected = (
        (start_days + 0.25 * (steps.astype(np.float64) + 1.0)) * nanoseconds_per_day
    ).astype("datetime64[ns]")

    np.testing.assert_array_equal(axis.datetimes(), expected)


def test_time_coordinate_delegates_to_the_time_axis():
    """The slab helper is a call site, not a second implementation."""
    axis = TimeAxis(START_DATE, np.arange(4), COUPLING_TIMESTEP, "365_day")

    values, attrs = time_coordinate(axis)

    np.testing.assert_array_equal(values, axis.datetimes())
    assert attrs == axis.attrs
    # A fresh dict each call: xarray keeps what it is handed.
    assert attrs is not axis.attrs


class ComponentRejectingClock(MinimalComponent):
    """A component whose ``bind`` refuses every clock, as JCM does on a mismatch."""

    def bind(self, *, coupling_timestep, start_date, calendar):
        del coupling_timestep, start_date, calendar
        raise ValueError("clock rejected")


def test_a_component_that_rejects_the_clock_is_not_registered():
    """A failed bind leaves the coupler as it was, including an earlier component of that name."""
    from jem.base.coupler import Coupler

    good = ComponentWithBind(name="ocn")
    coupler = Coupler(
        {"ocn": good},
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=jdt.to_datetime("2001-01-01"),
    )
    with pytest.raises(ValueError, match="clock rejected"):
        coupler.add_component("ocn", ComponentRejectingClock(name="ocn"))
    assert coupler.components["ocn"] is good
    with pytest.raises(ValueError, match="clock rejected"):
        coupler.add_component("ice", ComponentRejectingClock(name="ice"))
    assert "ice" not in coupler.components


def test_seconds_since_new_year_counts_in_the_model_calendar():
    """The day of year follows the calendar: no leap day on ``365_day``."""
    from jem.base.component import seconds_since_new_year

    day = 86400.0
    december_31_leap_year = jdt.to_datetime("2000-12-31")
    assert seconds_since_new_year(december_31_leap_year, "365_day") == 364 * day
    assert seconds_since_new_year(december_31_leap_year, "gregorian") == 365 * day
    # 1 March is day 59 (31 + 28) without a leap day, day 60 with one.
    assert seconds_since_new_year(jdt.to_datetime("2000-03-01"), "365_day") == 59 * day
    assert seconds_since_new_year(jdt.to_datetime("2000-03-01"), "gregorian") == 60 * day
    with pytest.raises(ValueError, match="29 February"):
        seconds_since_new_year(jdt.to_datetime("2000-02-29"), "365_day")
    with pytest.raises(ValueError, match="calendar"):
        seconds_since_new_year(december_31_leap_year, "360_day")
