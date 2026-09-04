"""The component contract and the coupled-model state types.

This module is the whole of the interface between the :class:`Coupler` and
the things it couples. It is deliberately small:

- :class:`Component` is a :class:`typing.Protocol`: a component is any object
  with a ``name``, an ``initialize()`` and a ``step(carry, time)``. There is
  no base class to inherit from, so an external model (JCM, Veros) is adapted
  by a thin wrapper class rather than by monkey-patching methods onto it.
- :class:`SupportsXarray`, :class:`SupportsCheckpoint` and
  :class:`SupportsBind` are *optional* capabilities. The coupler tests for
  them with ``isinstance`` (the protocols are runtime-checkable, which for a
  Protocol means "has these attributes"), never with ``hasattr`` at random
  call sites.
- :class:`CoupledCarry` is the scanned state of the coupled model: one carry
  per component plus the authoritative step counter. The counter lives in the
  carry, not in the ``lax.scan`` index, so the clock survives chunked runs and
  checkpoint restarts (the scan index restarts at zero on every call; the
  carry does not).
- :class:`CouplingTime` is what every ``Component.step`` receives instead of
  a bare step index: the step, the simulation time in seconds and the static
  calendar facts needed to turn that into a position in the seasonal cycle.
  Components therefore hold **no clock state of their own**; the coupler owns
  the one clock, and two components can never disagree about the date.
- :data:`Exchanger` is the type of the functions that move information
  between components. They were called "mappers" before v1.0; the name was
  changed because "mapper" reads as a regridding operation, whereas an
  exchanger may regrid, compute fluxes, convert units or simply copy a field.
  It is the *only* place where one component's carry is read by another.

The design is recorded in ``docs/source/design/api_hardening_plan.md`` (T1.1
and T1.3) and ``docs/source/design/architecture.md``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr
from flax import struct
from jcm.date import days_per_year as jcm_days_per_year

# A component's carry is an arbitrary pytree; by convention the slab models
# and the JCM wrapper use a dict with "state", "forcing" and "derived" keys
# (see architecture.md), but the coupler never looks inside it.
Carry = Any
# What a component returns per step for output; also an arbitrary pytree.
# The coupler stacks it over the scanned steps, so every leaf gains a leading
# time axis of length ``iterations``.
Diagnostics = Any

SECONDS_PER_DAY = 86400.0

#: Nanoseconds in a day, as a float64 -- the exact factor JCM multiplies its
#: float64 day counts by when it builds a ``datetime64[ns]`` time axis.
NANOSECONDS_PER_DAY = np.timedelta64(1, "D") / np.timedelta64(1, "ns")


def seconds_since_new_year(start_date: jdt.Datetime) -> float:
    """Return the seconds from 1 January of ``start_date``'s year to ``start_date``.

    This offset is what turns simulation time (seconds since the start of the
    run) into a position in the annual cycle, so a run that starts in July
    reads the July record of a monthly climatology on its first step. The
    coupler puts it on every :class:`CouplingTime` as ``year_offset_seconds``.

    The subtraction uses ``jax_datetime``'s proleptic-Gregorian arithmetic
    while the annual cycle is closed with the *model* calendar's
    ``days_per_year`` (see :func:`start_year_fraction`). The two disagree by
    one day for a start date after 29 February of a leap year under a
    ``365_day`` calendar. That is inherent in describing a real start date on
    an idealised calendar; it is left visible here rather than hidden by a
    second, home-made calendar.
    """
    year = start_date.to_pydatetime().year
    new_year = jdt.to_datetime(f"{year:d}-01-01")
    return float((start_date - new_year) / jdt.to_timedelta(1, "second"))


def start_year_fraction(start_date: jdt.Datetime, calendar: str) -> float:
    """Return the position of ``start_date`` in the annual cycle, in ``[0, 1)``.

    Zero is 00:00 on 1 January. This is the same quantity
    :attr:`CouplingTime.year_fraction` reports at step 0, computed from the
    same two facts (the offset into the year and the calendar's year length),
    so a component that samples a climatology in ``initialize()`` and one that
    samples it in ``step()`` cannot disagree about where the run starts.

    Parameters
    ----------
    start_date : jax_datetime.Datetime
        The run's start date.
    calendar : str
        Calendar name as JCM spells it (``"365_day"``, ``"gregorian"``).

    Returns
    -------
    float

    """
    seconds_per_year = SECONDS_PER_DAY * float(jcm_days_per_year(calendar))
    # The modulo matters for the leap-day case documented on
    # `seconds_since_new_year`: 31 December of a Gregorian leap year is a full
    # 365 days into a "365_day" year, which would otherwise report 1.0.
    return (seconds_since_new_year(start_date) / seconds_per_year) % 1.0


def _timedelta_days(delta: Any) -> float:
    """Return a ``jax_datetime`` days/seconds pair as a float64 count of days.

    ``jdt.Timedelta`` (and the ``.delta`` of a ``jdt.Datetime``) stores whole
    days and the seconds within the day separately. Dividing by a one-day
    ``Timedelta`` would work but goes through jax; this stays in numpy so the
    result is a plain float64 usable in the output-labelling arithmetic.
    """
    return (
        float(np.asarray(delta.days))
        + float(np.asarray(delta.seconds)) / SECONDS_PER_DAY
    )


@struct.dataclass
class CouplingTime:
    """The coupler's clock as seen by one component step.

    Attributes
    ----------
    step : jax.Array
        int32 scalar; number of coupling steps completed before this one
        (0 on the first step). Copied from :attr:`CoupledCarry.step`.
    sim_time : jax.Array
        Seconds since ``start_date``; equals ``step * dt``. Float64 when
        ``jax_enable_x64`` is on, float32 otherwise.
    dt : float
        Coupling timestep in seconds. Static (not a pytree leaf).
    year_offset_seconds : float
        Seconds from 1 January of the start year to ``start_date``. Static.
    days_per_year : float
        Length of the year in days for the run's calendar, from
        ``jcm.date.days_per_year``. Static.

    """

    step: jax.Array
    sim_time: jax.Array
    dt: float = struct.field(pytree_node=False)
    year_offset_seconds: float = struct.field(pytree_node=False)
    days_per_year: float = struct.field(pytree_node=False)

    def end_of_step(self) -> "CouplingTime":
        """Return the clock as it reads at the end of this step (one step later).

        Both ``step`` and ``sim_time`` advance together; a component that
        needs a boundary condition at both ends of a step (the slab models
        measure an anomaly against the climatology at the start and add it
        back at the end) must use this rather than adding ``dt`` to
        ``sim_time`` by hand, because :attr:`year_fraction` is computed from
        ``step`` whenever the step divides the year.
        """
        advanced: CouplingTime = self.replace(  # type: ignore[attr-defined]
            step=self.step + 1, sim_time=self.sim_time + self.dt
        )
        return advanced

    @property
    def seconds_per_year(self) -> float:
        """Length of the model year in seconds."""
        return SECONDS_PER_DAY * self.days_per_year

    @property
    def year_fraction(self) -> jax.Array:
        """Position in the annual cycle in ``[0, 1)`` at the *start* of this step.

        Zero is 00:00 on 1 January. This is what a monthly climatology is
        interpolated with (``jem.utils.cycles.evaluate_cyclic_linear``).

        Precision note: ``sim_time`` is a float32 array unless x64 is enabled,
        and float32 resolves only ~7 digits, so after a century of simulated
        time (3e9 s) it is quantised to hundreds of seconds. When the
        coupling step divides the year exactly (the usual case: daily steps
        in a 365-day year) the step count is reduced modulo the steps per
        year in exact integer arithmetic first, so the fraction keeps full
        float32 precision (a few seconds) for runs of any length. Otherwise
        the seconds are used directly and precision degrades with run length.
        """
        steps_per_year = self.seconds_per_year / self.dt
        if float(steps_per_year).is_integer():
            seconds_into_year = self.year_offset_seconds + (
                jnp.mod(self.step, int(steps_per_year)) * self.dt
            )
        else:
            seconds_into_year = self.year_offset_seconds + self.sim_time
        return jnp.mod(seconds_into_year / self.seconds_per_year, 1.0)


@struct.dataclass
class CoupledCarry:
    """The scanned state of the whole coupled model.

    Attributes
    ----------
    components : dict[str, Carry]
        One carry per component, keyed by component name.
    step : jax.Array
        int32 scalar; number of coupling steps completed. The coupler
        increments it once per coupled step and builds :class:`CouplingTime`
        from it, so it is the single source of truth for the model clock.

    """

    components: dict[str, Carry]
    step: jax.Array


@dataclasses.dataclass(frozen=True)
class TimeAxis:
    """The output records of a run and how they are labelled in time.

    Built by ``Coupler.time_axis(first_step, n)``; handed to
    :meth:`SupportsXarray.to_xarray` so every component labels its output
    with the same ``time`` coordinate and ``xr.merge`` of two components'
    datasets is an N-long join rather than a 2N-long union.

    The labelling convention is JCM's, which JAX-ESM cannot change from the
    outside (jax-gcm#758): record ``k`` is the average over
    ``[start_date + k dt, start_date + (k+1) dt)`` and is labelled with the
    **end** of that interval, ``start_date + (k+1) dt``, as a
    ``datetime64[ns]`` on the proleptic Gregorian calendar whatever the
    model calendar is (a ``365_day`` run still writes real dates; the
    calendar governs only the seasonal cycle and forcing selection).
    :meth:`datetimes` implements exactly that and is the one place the
    convention is written down.

    Attributes
    ----------
    start_date : jdt.Datetime
        The run's start date.
    steps : numpy.ndarray
        int array of coupled-step indices, one per record.
    dt : jdt.Timedelta
        Coupling timestep.
    calendar : str
        Calendar name as JCM spells it (``"365_day"``, ``"gregorian"``).

    """

    start_date: jdt.Datetime
    steps: Any
    dt: jdt.Timedelta
    calendar: str

    def __len__(self) -> int:
        """Return the number of output records."""
        return len(self.steps)

    def datetimes(self) -> np.ndarray:
        """Return the record labels as ``datetime64[ns]`` (end of each interval).

        The arithmetic, not just the answer, is JCM's
        (``jcm.predictions.ModelPredictions._trajectory_dataset``)::

            times = start_date.delta.days + save_interval * (arange(n) + 1)
            time  = (times * NANOSECONDS_PER_DAY).astype("datetime64[ns]")

        that is: a float64 count of **days** since the 1970 epoch, multiplied
        into nanoseconds at the end. That product overflows the exactly
        representable range of float64 (a 2001 date is ~9.5e17 ns, whose ulp
        is 128 ns), so the instants are not exact to the nanosecond -- but
        they are *identically* inexact for every component that comes through
        here, which is the property that makes ``xr.merge`` align two
        components' output on one time axis. Computing the exact integer
        nanosecond count instead would be more accurate and would merge with
        nothing.

        Sub-day start dates are the one deliberate difference from JCM's own
        output path, which takes ``start_date.delta.days`` and drops
        ``.seconds`` entirely, so a JCM run starting at 06:00 labels its
        records from midnight. Reproducing that here would mislabel a slab
        dataset by up to a day, so the seconds are kept. For a start date at
        midnight -- every configuration JAX-ESM ships -- the two agree bit for
        bit, because the added term is exactly 0.0.
        """
        start_days = _timedelta_days(self.start_date.delta)
        step_days = _timedelta_days(self.dt)
        steps = np.asarray(self.steps, dtype=np.float64)
        days = start_days + step_days * (steps + 1.0)
        return (days * NANOSECONDS_PER_DAY).astype("datetime64[ns]")

    @property
    def attrs(self) -> dict[str, str]:
        """CF attributes JCM writes on its ``time`` coordinate.

        ``units`` is deliberately absent: xarray owns it through the datetime
        encoding it chooses on write, and setting it here collides with that.
        """
        return {"standard_name": "time", "axis": "T", "long_name": "time"}


@runtime_checkable
class Component(Protocol):
    """What the coupler requires of anything it steps.

    A component holds its *configuration* (grid, parameters, boundary data)
    on ``self`` and its *evolving state* in the carry it returns from
    ``initialize`` and threads through ``step``. ``step`` must be a pure
    function of ``(carry, time)`` and must return a carry with exactly the
    pytree structure, shapes and dtypes it received, or ``lax.scan`` rejects
    it.
    """

    name: str

    def initialize(self) -> Carry:
        """Build the initial carry. Must not integrate the model."""
        ...

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance one coupling timestep; return the new carry and the step's output."""
        ...


@runtime_checkable
class SupportsXarray(Protocol):
    """Optional: convert stacked diagnostics to an ``xarray.Dataset``."""

    def to_xarray(self, diagnostics: Diagnostics, time: TimeAxis) -> xr.Dataset: ...


@runtime_checkable
class SupportsCheckpoint(Protocol):
    """Optional: components whose carry cannot be pickled as a plain pytree (Veros)."""

    def save_state(self, carry: Carry, directory: Path) -> None: ...

    def load_state(self, directory: Path) -> Carry: ...


@runtime_checkable
class SupportsBind(Protocol):
    """Optional: receive the coupler's clock definition at registration.

    A component that has its own internal timestep (JCM, Veros) needs to know
    the coupling timestep to decide how many internal steps make one coupled
    step, and needs to agree with the coupler about the start date and
    calendar. The coupler calls ``bind`` once, from its constructor, for
    every component that provides it. Raise ``ValueError`` on a mismatch.
    """

    def bind(
        self,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str,
    ) -> None: ...


# An exchanger moves information between components. It receives the mapping
# of component carries (a fresh dict, so adding or replacing entries never
# mutates the coupler's input) and the current clock, and returns the mapping
# to continue with. It must not mutate the carries it receives in place;
# build new ones with ``dataclasses.replace`` / ``.replace`` and return them.
# The clock is passed so an exchanger can implement time-dependent coupling
# (lagged exchange, ramped forcing) without keeping state of its own.
Exchanger = Callable[[dict[str, Carry], CouplingTime], dict[str, Carry]]
