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
    FORCING_VARIABLE_PREFIX,
    Component,
    SupportsBind,
    SupportsCheckpoint,
    SupportsXarray,
    TimeAxis,
    forcing_variable,
)
from jem.base.coupler import Coupler

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

    def save_carry(self, carry, directory):
        del carry, directory

    def load_carry(self, directory):
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


def test_datetimes_label_the_midpoint_of_each_interval():
    """Record ``k`` holds the interval ``[start + k dt, start + (k+1) dt)``,
    labelled at its MIDPOINT, ``start + (k + 1/2) dt``.

    jax-gcm PR 878 moved an averaged record's label from the end of its
    interval to its midpoint (``docs/source/v2_to_v3.rst``, "One real
    datetime clock"), and ``TimeAxis.datetimes`` follows suit so a coupled
    run's other components merge with the atmosphere's output on one time
    axis (see the class docstring).
    """
    axis = TimeAxis(
        start_date=jdt.to_datetime("2001-01-01"),
        steps=np.arange(3),
        dt=COUPLING_TIMESTEP,
        calendar="365_day",
    )
    np.testing.assert_array_equal(
        axis.datetimes(),
        np.array(
            ["2001-01-01T12:00:00", "2001-01-02T12:00:00", "2001-01-03T12:00:00"],
            dtype="datetime64[ms]",
        ),
    )
    assert axis.datetimes().dtype == np.dtype("datetime64[ms]")


def test_datetimes_call_jcms_own_output_time_labels():
    """The labels are ``jcm.predictions.output_time_labels``'s, exactly.

    Both models have to compute the SAME conversion for ``xr.merge`` to align
    them, so this pins that ``TimeAxis.datetimes`` calls jax-gcm's own public
    conversion (closing jax-gcm#862) rather than reimplementing it -- and, in
    particular, that an odd-length interval's midpoint lands on the exact
    half second ``output_time_labels`` promises, which a naive
    ``jax_datetime.Timedelta`` (whole-seconds-only) computation could not
    represent.
    """
    from jcm.predictions import output_time_labels

    start = jdt.to_datetime("2001-03-01")
    steps = np.arange(5)
    dt = jdt.to_timedelta(6, "hour")
    axis = TimeAxis(start, steps, dt, "365_day")

    dt_seconds = 6 * 3600
    start_seconds = steps.astype(np.int64) * dt_seconds
    end_seconds = start_seconds + dt_seconds

    def _exact(seconds):
        days, secs = np.divmod(seconds, 86_400)
        return start + jdt.Timedelta(
            days=jnp.asarray(days, dtype=jnp.int32),
            seconds=jnp.asarray(secs, dtype=jnp.int32),
        )

    bounds_start = output_time_labels(_exact(start_seconds))
    bounds_end = output_time_labels(_exact(end_seconds))
    expected = bounds_start + (bounds_end - bounds_start) // 2

    np.testing.assert_array_equal(axis.datetimes(), expected)
    assert axis.datetimes().dtype == np.dtype("datetime64[ms]")


def test_datetimes_gives_an_exact_half_second_midpoint():
    """An odd-length interval's midpoint is exact to the millisecond.

    A 1-second coupling step covers an odd number of seconds only through its
    ``multiplicity`` (sub-timestep) form -- here, three coupled steps of a
    single second each, checked as a length-3 sub-axis -- and the middle one,
    ``[1, 2)`` seconds after the start, must land exactly on the half second
    (``docs/source/v2_to_v3.rst``: "the millisecond output unit only serves
    external half-second midpoints").
    """
    start = jdt.to_datetime("2001-01-01")
    axis = TimeAxis(start, np.arange(3), jdt.to_timedelta(1, "second"), "365_day")

    labels = axis.datetimes()
    assert labels.dtype == np.dtype("datetime64[ms]")
    np.testing.assert_array_equal(
        labels,
        np.array(
            ["2001-01-01T00:00:00.500", "2001-01-01T00:00:01.500",
             "2001-01-01T00:00:02.500"],
            dtype="datetime64[ms]",
        ),
    )


def test_datetimes_refuses_a_step_whose_day_count_would_wrap_on_every_calendar():
    """A step the int32 step counter holds, but whose date wraps, is refused on both calendars.

    The labels are proleptic Gregorian whatever the axis's calendar is, so a
    ``"365_day"`` axis is as exposed as a ``"gregorian"`` one. At
    ``2**31 - 1`` daily steps from 2001 the step itself fits int32, but the
    day count of the end of its interval does not: the same arithmetic,
    ``gregorian_instant`` at the interval's end, wraps to a negative day
    (a date millions of years before the start), and ``datetimes`` refuses
    rather than write it.
    """
    from jem.base.calendar import gregorian_instant

    step_int32_max = 2**31 - 1
    start = jdt.to_datetime("2001-01-01")
    dt = jdt.to_timedelta(1, "day")
    end_days, _ = gregorian_instant(
        jnp.int32(step_int32_max), 86_400, int(start.delta.days),
        int(start.delta.seconds), offset_seconds=86_400,
    )
    assert int(end_days) < 0  # the day count this step would be labelled with wraps
    for calendar in ("365_day", "gregorian"):
        axis = TimeAxis(
            start_date=start, steps=np.array([step_int32_max], dtype=np.int64),
            dt=dt, calendar=calendar,
        )
        with pytest.raises(ValueError, match="would wrap"):
            axis.datetimes()


def test_datetimes_accepts_exactly_at_and_refuses_one_past_its_day_count_limit():
    """The bound is exact: the last safe step is labelled, the next is refused, on every calendar.

    The last safe step's label is also checked against the exact day it
    must fall on -- the midpoint of its interval, one day before the last
    day an int32 day count holds -- so the accepted boundary is a correct
    label, not merely an unrefused one.
    """
    from jem.base.calendar import max_safe_record

    start = jdt.to_datetime("2001-01-01")
    dt = jdt.to_timedelta(1, "day")
    dt_seconds = 86_400
    limit = max_safe_record(
        dt_seconds, offset_seconds=dt_seconds,
        start_seconds=int(start.delta.seconds), start_days=int(start.delta.days),
    )
    expected = np.datetime64(0, "ms") + np.timedelta64(2**31 - 2, "D") + np.timedelta64(12, "h")
    for calendar in ("365_day", "gregorian"):
        accepted = TimeAxis(
            start_date=start, steps=np.array([limit], dtype=np.int64),
            dt=dt, calendar=calendar,
        )
        assert accepted.datetimes()[0] == expected

        refused = TimeAxis(
            start_date=start, steps=np.array([limit + 1], dtype=np.int64),
            dt=dt, calendar=calendar,
        )
        with pytest.raises(ValueError, match="would wrap"):
            refused.datetimes()


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.int16])
def test_datetimes_checks_the_largest_step_of_any_integer_dtype(dtype):
    """The bound is checked against the largest step, whatever integer dtype ``steps`` has.

    The step past the bound is placed in the middle of the array, so a check
    that read only the first or last step would miss it.
    """
    from jem.base.calendar import max_safe_record

    start = jdt.to_datetime("2001-01-01")
    dt = jdt.to_timedelta(1, "day")
    limit = max_safe_record(
        86_400, offset_seconds=86_400,
        start_seconds=int(start.delta.seconds), start_days=int(start.delta.days),
    )
    if limit + 1 > np.iinfo(dtype).max:
        # A dtype too narrow to hold a step past the bound cannot ask for one;
        # its own largest step is labelled.
        steps = np.array([0, np.iinfo(dtype).max, 1], dtype=dtype)
        assert len(TimeAxis(start, steps, dt, "365_day").datetimes()) == 3
        return
    steps = np.array([0, limit + 1, 1], dtype=dtype)
    with pytest.raises(ValueError, match="would wrap"):
        TimeAxis(start, steps, dt, "365_day").datetimes()


def test_datetimes_of_an_empty_axis_is_empty():
    """An axis with no records has no step to bound, and returns no labels."""
    axis = TimeAxis(
        start_date=jdt.to_datetime("2001-01-01"), steps=np.array([], dtype=np.int64),
        dt=jdt.to_timedelta(1, "day"), calendar="365_day",
    )
    assert len(axis.datetimes()) == 0


def test_datetimes_is_exact_at_the_bound_for_a_start_before_the_epoch():
    """A pre-epoch start labels its last safe step exactly, although its relative day count passes int32.

    From 1850 with a two-day step, the last step
    :func:`jem.base.calendar.max_safe_record` allows ends more than
    ``2**31 - 1`` days after the start while its absolute date still fits
    int32. ``datetimes`` bounds the absolute date only, so this step must be
    both accepted and labelled with the exact day -- the midpoint of its
    interval, one day before the last day an int32 day count holds.
    """
    from jem.base.calendar import max_safe_record

    start = jdt.to_datetime("1850-01-01")
    dt = jdt.to_timedelta(2, "day")
    start_days = int(start.delta.days)
    limit = max_safe_record(
        2 * 86_400, offset_seconds=2 * 86_400,
        start_seconds=int(start.delta.seconds), start_days=start_days,
    )
    assert 2 * (limit + 1) > 2**31 - 1  # the relative day count is past int32
    labels = TimeAxis(start, np.array([limit]), dt, "gregorian").datetimes()
    expected_days = start_days + 2 * limit + 1
    assert labels[0] == np.datetime64(0, "ms") + np.timedelta64(expected_days, "D")
    with pytest.raises(ValueError, match="would wrap"):
        TimeAxis(start, np.array([limit + 1]), dt, "gregorian").datetimes()


def test_time_axis_attrs_is_a_fresh_dict_per_access():
    """Hand every caller its own dict, because xarray keeps what it is given.

    ``to_xarray`` passes ``attrs`` straight to :class:`xarray.Dataset`, so a
    shared dict would let one dataset's edit reach every other component's.
    """
    axis = TimeAxis(START_DATE, np.arange(4), COUPLING_TIMESTEP, "365_day")

    assert axis.attrs == axis.attrs
    assert axis.attrs is not axis.attrs


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


def test_forcing_variable_prefixes_once():
    """A name that already carries the prefix is not prefixed again."""
    assert forcing_variable("total_heat_flux") == "forcing_total_heat_flux"
    assert forcing_variable("forcing_shortwave_flux") == "forcing_shortwave_flux"
    assert forcing_variable("total_heat_flux").startswith(FORCING_VARIABLE_PREFIX)
