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
