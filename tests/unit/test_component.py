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
from jcm.date import fraction_of_year_elapsed

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

    def bind(self, *, coupling_timestep, start_date):
        self.bound = {
            "coupling_timestep": coupling_timestep,
            "start_date": start_date,
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
    )

    assert component.bound is not None
    assert component.bound["coupling_timestep"] == COUPLING_TIMESTEP
    assert component.bound["start_date"] == START_DATE


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


def test_datetimes_label_each_interval_midpoint():
    """Record ``k`` holds the interval ``[start + k dt, start + (k+1) dt)``.

    and is labelled at its MIDPOINT, JCM's convention
    (``jcm.predictions.output_time_labels``).
    """
    axis = TimeAxis(
        start_date=jdt.to_datetime("2001-01-01"),
        steps=np.arange(3),
        dt=COUPLING_TIMESTEP,
    )
    np.testing.assert_array_equal(
        axis.datetimes(),
        np.array(
            ["2001-01-01T12:00", "2001-01-02T12:00", "2001-01-03T12:00"],
            dtype="datetime64[ms]",
        ),
    )
    assert axis.datetimes().dtype == np.dtype("datetime64[ms]")


def test_datetimes_reproduce_jcm_arithmetic_bit_for_bit():
    """The labels are JCM's floor-divided integer-millisecond midpoint.

    Both models have to be inexact in the SAME way for ``xr.merge`` to align
    them, so this pins the arithmetic and not just the answer: an odd-length
    interval's half-millisecond midpoint must round down, exactly as
    ``jcm.predictions.ModelPredictions.time_labels`` does.
    """
    start = jdt.to_datetime("2001-03-01")
    steps = np.arange(5)
    dt = jdt.to_timedelta(6, "hour")
    axis = TimeAxis(start, steps, dt)

    start_ms = int(np.asarray(start.delta.days)) * 86_400_000
    dt_ms = 6 * 3600 * 1000
    lower = start_ms + steps.astype(np.int64) * dt_ms
    upper = lower + dt_ms
    expected = (lower + (upper - lower) // 2).astype("datetime64[ms]")

    np.testing.assert_array_equal(axis.datetimes(), expected)


def test_time_axis_attrs_is_a_fresh_dict_per_access():
    """Hand every caller its own dict, because xarray keeps what it is given.

    ``to_xarray`` passes ``attrs`` straight to :class:`xarray.Dataset`, so a
    shared dict would let one dataset's edit reach every other component's.
    """
    axis = TimeAxis(START_DATE, np.arange(4), COUPLING_TIMESTEP)

    assert axis.attrs == axis.attrs
    assert axis.attrs is not axis.attrs


class ComponentRejectingClock(MinimalComponent):
    """A component whose ``bind`` refuses every clock, as JCM does on a mismatch."""

    def bind(self, *, coupling_timestep, start_date):
        del coupling_timestep, start_date
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


def test_year_fraction_equals_jcm_fraction_of_year_elapsed():
    """``CouplingTime.year_fraction`` is exactly ``jcm.date.fraction_of_year_elapsed``.

    Calling the same function the atmosphere's own seasonal physics does is
    what makes a slab's seasonal cycle and JCM's agree by construction --
    there is nothing of JEM's own to test beyond that delegation.
    """
    from jem.base.component import CouplingTime

    for when in ("2000-03-01", "2000-12-31", "2001-07-04T18:00:00"):
        time = jdt.to_datetime(when)
        coupling_time = CouplingTime(
            step=jnp.int32(0), time=time, sim_time=jnp.float32(0.0), dt=86400.0
        )
        assert float(coupling_time.year_fraction) == pytest.approx(
            float(fraction_of_year_elapsed(time))
        )


def test_start_year_fraction_matches_year_fraction_at_step_zero():
    """A slab's ``start_year_fraction`` and a step-0 ``CouplingTime`` agree.

    Both read the run's position in the annual cycle from the same start
    date, so a component that samples a climatology in ``initialize()`` and
    one that samples it in ``step()`` cannot disagree about where the run
    starts.
    """
    from jem.base.component import CouplingTime, start_year_fraction

    start = jdt.to_datetime("2001-07-01")
    coupling_time = CouplingTime(
        step=jnp.int32(0), time=start, sim_time=jnp.float32(0.0), dt=86400.0
    )
    assert start_year_fraction(start) == pytest.approx(
        float(coupling_time.year_fraction)
    )


def test_forcing_variable_prefixes_once():
    """A name that already carries the prefix is not prefixed again."""
    assert forcing_variable("total_heat_flux") == "forcing_total_heat_flux"
    assert forcing_variable("forcing_shortwave_flux") == "forcing_shortwave_flux"
    assert forcing_variable("total_heat_flux").startswith(FORCING_VARIABLE_PREFIX)
