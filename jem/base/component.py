"""The component contract and the coupled-model state types.

This module is the whole of the interface between the :class:`Coupler` and
the things it couples. It is deliberately small:

- :class:`Component` is a :class:`typing.Protocol`: a component is any object
  with a ``name``, an ``initialize()`` and a ``step(carry, time)``. There is
  no base class to inherit from, so an external model (JCM, Veros) is adapted
  by a thin wrapper class rather than by monkey-patching methods onto it.
- :class:`SupportsXarray`, :class:`SupportsCheckpoint`, :class:`SupportsBind`
  and :class:`SupportsInternalStepping` are *optional* capabilities. The
  coupler tests for them with ``isinstance`` (the protocols are
  runtime-checkable, which for a Protocol means "has these attributes"),
  never with ``hasattr`` at random call sites.
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

The design is recorded in ``docs/source/design/architecture.md`` and, for the
task numbering (T1.1, T1.3), in the API hardening plan, which lives on the
review branch rather than in this repository:
https://github.com/climate-analytics-lab/jax-esm/blob/claude/jax-esm-api-review-jv7j7u/docs/source/design/api_hardening_plan.md
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, get_args, runtime_checkable

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr
from flax import struct

# A component's carry is an arbitrary pytree; by convention the slab models
# and the JCM wrapper use a dict with "state", "forcing" and "derived" keys
# (see architecture.md), but the coupler never looks inside it.
Carry = Any
# What a component returns per step for output; also an arbitrary pytree.
# The coupler stacks it over the scanned steps, so every leaf gains a leading
# time axis of length ``iterations``.
Diagnostics = Any

SECONDS_PER_DAY = 86400.0

#: Prefix on the output name of a field a component was *given*, as opposed to
#: one it computed. See :func:`forcing_variable`.
FORCING_VARIABLE_PREFIX = "forcing_"


def forcing_variable(name: str) -> str:
    """Return the output-variable name for a field the component was forced with.

    A coupled run writes one dataset per component and they are meant to be
    read -- and merged -- together. A field a component *received* through the
    coupler is already written, unprefixed, by the component that produced it,
    so writing the received copy under the same name puts two different
    variables with one name into the merged dataset: different because coupling
    is lagged, so the copy is one step behind the original. ``xr.merge`` then
    refuses the two datasets outright.

    Prefixing the received copy with ``forcing_`` fixes that and says the more
    accurate thing anyway: ``forcing_total_heat_flux`` in the land model's
    output is the flux the land was driven with, not a flux the land computed.
    A component's own state and its derived diagnostics keep their plain names
    -- they are that component's output, and match what JCM calls them.

    This is a convention of the *packaged* output, not part of the component
    protocol: the coupler never inspects a dataset, and a wrapper around an
    external model may name its variables however that model does. The slab
    models and the Veros adapter follow it so that their datasets merge with
    each other's and with JCM's; a component added later only needs to follow
    it if its output is meant to merge the same way. It lives here, next to
    the rest of the contract, so that there is one definition of the prefix.

    A name that already carries the prefix is returned unchanged, so a model
    whose own field is called ``forcing_shortwave_flux`` does not come out as
    ``forcing_forcing_shortwave_flux``.

    Parameters
    ----------
    name : str
        The physical field's name, as the component that produces it writes it.

    Returns
    -------
    str

    """
    if name.startswith(FORCING_VARIABLE_PREFIX):
        return name
    return f"{FORCING_VARIABLE_PREFIX}{name}"


#: Name of the variable attribute that records which part of a component's
#: carry an output variable came from. See :func:`role_attrs`.
ROLE_ATTRIBUTE = "jem_role"

#: What a variable's role may be: the three sections of the carry layout the
#: packaged components share (``jem.exchangers`` addresses fields by them).
Role = Literal["state", "derived", "forcing"]

#: The roles, as a tuple, for validation and for iterating in a test.
ROLES: tuple[str, ...] = get_args(Role)


def role_attrs(role: Role) -> dict[str, str]:
    """Return the variable attributes marking an output variable's role.

    A packaged component's output says which part of its carry a variable
    came from in **two** ways, and they are not redundant:

    - The ``forcing_`` name prefix (:func:`forcing_variable`) exists to stop
      an ``xr.merge`` collision. A field one component computed and the copy
      another was given through the coupler are *different* variables --
      coupling is lagged, so the copy is a step behind -- and under one name
      ``xr.merge`` refuses the two datasets outright. Renaming is the only
      thing that fixes that, so the prefix stays.
    - This attribute exists so that nothing has to *parse* names to find out
      what a variable is. ``ds.filter_by_attrs(jem_role="forcing")`` is the
      whole query; the alternative, matching a prefix, cannot tell a received
      ``forcing_q_flux`` from a model whose own field happens to be called
      ``forcing_shortwave_flux``, and says nothing at all about the rest --
      whether ``total_heat_flux`` in the ocean's output is state the ocean
      integrated or a diagnostic it computed.

    So the prefix is a naming rule and this is metadata; every packaged
    component sets both. The roles are the sections of the carry layout the
    packaged components share and that :mod:`jem.exchangers` addresses:
    ``state`` is what the component integrates, ``derived`` what it diagnosed
    for others to read, ``forcing`` what it was given. A variable that is
    none of those -- a grid mask, a layer thickness, anything time-invariant
    that came from the component's configuration rather than its carry -- is
    left untagged, which is a meaningful answer and not an omission.

    A fresh dict is returned on every call, because xarray keeps the dict it
    is handed: two variables sharing one attrs dict would share any later
    edit to it.

    Parameters
    ----------
    role : {"state", "derived", "forcing"}
        Which section of the carry the variable was read from.

    Returns
    -------
    dict[str, str]
        ``{"jem_role": role}``, ready to merge into a variable's attributes.

    Raises
    ------
    ValueError
        If ``role`` is not one of the three.

    """
    if role not in ROLES:
        raise ValueError(
            f"Unknown variable role {role!r}; it must be one of {list(ROLES)!r}."
        )
    return {ROLE_ATTRIBUTE: role}


#: Days per year for each calendar JEM's own annual-cycle bookkeeping
#: supports (:class:`CouplingTime`'s ``year_fraction``, the slab models'
#: climatology sampling via :func:`start_year_fraction`, and
#: ``jem.accumulate.monthly_mean``'s fixed month-length table).
#:
#: Through jax-gcm PR 877 this table lived in jax-gcm itself
#: (``jcm.date.days_per_year``), because jax-gcm's own ``Model.calendar``
#: selected between the same two conventions and JEM deliberately deferred to
#: it so the two packages could not disagree about a year's length. jax-gcm
#: PR 878 (the v3 exact datetime clock) made jax-gcm's clock unconditionally
#: proleptic Gregorian and removed both the calendar concept and this table
#: -- ``jcm.date`` has no ``days_per_year`` any more. JEM's own components
#: still need a *fixed-length* "days per year" for windows and seasonal-cycle
#: bookkeeping that has nothing to do with the atmosphere's clock -- a
#: Veros- or slab-only coupled run has no jax-gcm component at all -- so the
#: table moves here unchanged rather than disappearing with jax-gcm's copy of
#: it. See :data:`jem.components.jcm.contract.JCM_SUPPORTED_REV`.
#:
#: A ``Coupler`` bound to a real ``jcm.model.Model`` must use ``"gregorian"``
#: -- :meth:`jem.components.jcm.component.JCMComponent.bind` enforces this --
#: because jax-gcm's atmosphere physics and forcing selection are
#: unconditionally Gregorian now, and any other choice here would silently
#: put the atmosphere's seasonal cycle out of phase with every other
#: component's. ``"365_day"`` remains for a coupled model with no atmosphere.
_DAYS_PER_YEAR_BY_CALENDAR: dict[str, float] = {
    "gregorian": 365.2425,
    "365_day": 365.0,
}

#: Duration units whose length in days does not depend on the calendar,
#: mirroring the fixed-unit table ``jcm.date`` carried before PR 878 (which
#: replaced it with an equivalent whole-seconds table of its own,
#: ``jcm.date._FIXED_UNIT_SECONDS`` -- not imported here, so that JEM's own
#: duration parsing does not depend on jax-gcm's internal names, and so that
#: it keeps accepting durations at plain floating-point precision rather than
#: jax-gcm v3's stricter whole-second requirement, which JEM's own scheduling
#: was never written to).
_FIXED_UNIT_DAYS: dict[str, float] = {
    "sec": 1.0 / 86400.0, "secs": 1.0 / 86400.0,
    "second": 1.0 / 86400.0, "seconds": 1.0 / 86400.0,
    "min": 1.0 / 1440.0, "mins": 1.0 / 1440.0,
    "minute": 1.0 / 1440.0, "minutes": 1.0 / 1440.0,
    "h": 1.0 / 24.0, "hr": 1.0 / 24.0, "hrs": 1.0 / 24.0,
    "hour": 1.0 / 24.0, "hours": 1.0 / 24.0,
    "d": 1.0, "day": 1.0, "days": 1.0,
    "w": 7.0, "wk": 7.0, "wks": 7.0, "week": 7.0, "weeks": 7.0,
}
_MONTH_ALIASES = {"mo", "mon", "mons", "month", "months"}
_YEAR_ALIASES = {"y", "yr", "yrs", "year", "years"}


def days_per_year(calendar: str) -> float:
    """Return the days-per-year JEM's own annual-cycle bookkeeping uses for ``calendar``.

    ``"gregorian"`` is the true average Gregorian year, ``365.2425`` days
    (the astronomical value the calendar is calibrated to: 400 years contain
    exactly 97 leap years, so ``365 + 97/400 = 365.2425``) -- **not** the
    exact length of any *particular* year, which is 365 or 366 depending
    which one. Every caller of this value (:class:`CouplingTime`'s
    ``year_fraction``, :func:`seconds_since_new_year` /
    :func:`start_year_fraction`, and ``jem.accumulate``'s bin-sizing
    arithmetic) treats a calendar's year as one fixed length for the whole
    run, so there is no way to plug in "365 or 366, whichever this
    particular year is" without changing that shared assumption -- doing so
    is a larger redesign than this migration's scope (see
    ``jem.accumulate.monthly_mean``'s own ``NotImplementedError`` for the
    sharpest edge of the same limit: a *fixed* calendar-month table cannot
    exist for a calendar whose year is not a fixed number of days at all).
    The consequence is a **bounded, non-accumulating** phase error of at most
    a fraction of a day within any given year (as opposed to ``"365_day"``,
    whose error against a real Gregorian atmosphere *accumulates* -- about a
    day every four years, since it never has a 29 February at all) -- see
    :data:`jem.components.jcm.contract.JCM_SUPPORTED_REV`'s "Why a `dev`
    revision" note for why ``"gregorian"`` is nonetheless the required choice
    for a jax-gcm-coupled run. This value is unchanged from the one
    ``jcm.date.days_per_year("gregorian")`` returned before jax-gcm v3
    removed it (jax-gcm PR 878) -- moved here, not reconsidered, because nothing
    about the migration bears on what the right approximation is.

    Parameters
    ----------
    calendar : str
        ``"gregorian"`` or ``"365_day"``.

    Raises
    ------
    ValueError
        If ``calendar`` is neither.

    """
    try:
        return _DAYS_PER_YEAR_BY_CALENDAR[calendar]
    except KeyError as exc:
        raise ValueError(
            f"Unknown calendar {calendar!r}; expected one of "
            f"{tuple(_DAYS_PER_YEAR_BY_CALENDAR)}."
        ) from exc


def parse_duration_days(value: str | float, calendar: str) -> float:
    """Parse a duration spec into a float number of days, on ``calendar``.

    Numeric input (int / float) is returned as-is -- assumed to be days.
    Strings are parsed as ``<number> <unit>``, e.g. ``'1 month'``,
    ``'5 years'``, ``'30 days'``, ``'12 hours'``. Months and years are mapped
    through :func:`days_per_year` (so under ``'365_day'``, ``'1 month'`` is
    ``365/12`` days; under ``'gregorian'`` it is ``365.2425/12``).

    This is JEM's own duration parser (jax-gcm's ``jcm.date.parse_duration_days``
    dropped both the calendar argument and month/year units in PR 878, since a
    fixed-duration model clock has no use for either) -- it schedules JEM's own
    coupled-run and output-windowing durations (``coupled_run.total_time``,
    ``jem.accumulate``'s windows), which are calendar concepts independent of
    whatever clock the atmosphere -- if there is one -- runs on.

    Parameters
    ----------
    value : str or float
        The duration, as a number of days or a ``'<number> <unit>'`` string.
    calendar : str
        Calendar name as JCM used to spell it (``"365_day"``, ``"gregorian"``),
        used only to resolve a month/year unit.

    Returns
    -------
    float

    """
    if isinstance(value, (int, float)):
        return float(value)

    import re
    s = str(value).strip().lower()
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-z]+)\s*$", s)
    if not m:
        raise ValueError(
            f"Cannot parse duration {value!r}. Expected '<number> <unit>' "
            "with unit in {seconds, minutes, hours, days, weeks, months, years}."
        )
    n = float(m.group(1))
    unit = m.group(2)

    if unit in _FIXED_UNIT_DAYS:
        return n * _FIXED_UNIT_DAYS[unit]
    if unit in _MONTH_ALIASES:
        return n * days_per_year(calendar) / 12.0
    if unit in _YEAR_ALIASES:
        return n * days_per_year(calendar)

    raise ValueError(
        f"Unknown duration unit {unit!r} in {value!r}. Expected one of "
        f"{sorted(_FIXED_UNIT_DAYS)} ∪ {sorted(_MONTH_ALIASES | _YEAR_ALIASES)}."
    )


def seconds_since_new_year(start_date: jdt.Datetime, calendar: str) -> float:
    """Return the seconds from 1 January of ``start_date``'s year to ``start_date``.

    This offset is what turns simulation time (seconds since the start of the
    run) into a position in the annual cycle, so a run that starts in July
    reads the July record of a monthly climatology on its first step. The
    coupler puts it on every :class:`CouplingTime` as ``year_offset_seconds``.

    The day of year is counted in the *model* calendar. On a ``365_day``
    calendar there is no 29 February, so a Gregorian date after it is one
    day earlier in the model year than the real-calendar subtraction would
    say (31 December is day 364, not day 365, so the seasonal cycle does
    not wrap a day early); a date that does not exist in that calendar is
    rejected. On the ``gregorian`` calendar the real subtraction applies.

    Parameters
    ----------
    start_date : jax_datetime.Datetime
        The run's start date.
    calendar : str
        Calendar name as JCM used to spell it (``"365_day"``, ``"gregorian"``).

    Raises
    ------
    ValueError
        If ``start_date`` does not exist in ``calendar`` (29 February on a
        365-day calendar), or the calendar is unknown (see :func:`days_per_year`).

    """
    when = start_date.to_pydatetime()
    if float(days_per_year(calendar)) == 365.0:
        if when.month == 2 and when.day == 29:
            raise ValueError(
                f"{when.date()} does not exist in the {calendar!r} calendar, "
                "which has no 29 February."
            )
        # Count the day of year in a year without a leap day: any non-leap
        # reference year gives the same month/day -> day-of-year mapping.
        reference = datetime.datetime(
            2001, when.month, when.day, when.hour, when.minute, when.second,
            when.microsecond,
        )
        return (reference - datetime.datetime(2001, 1, 1)).total_seconds()
    new_year = jdt.to_datetime(f"{when.year:d}-01-01")
    return float((start_date - new_year) / jdt.to_timedelta(1, "second"))


def start_year_fraction(start_date: jdt.Datetime, calendar: str) -> float:
    """Return the position of ``start_date`` in the annual cycle, in ``[0, 1)``.

    Zero is 00:00 on 1 January. This is the same quantity
    :attr:`CouplingTime.year_fraction` reports at step 0, so a component that
    samples a climatology in ``initialize()`` and one that samples it in
    ``step()`` cannot disagree about where the run starts.

    On ``"gregorian"`` this computes the exact real-calendar day-of-year and
    leap-year status of ``start_date`` on the host, with Python's
    ``datetime`` -- rather than reusing :func:`seconds_since_new_year`'s
    fixed-average-year division, which is what :attr:`CouplingTime
    .year_fraction` itself no longer does either (see that property's
    docstring for the phase-drift this was found to cause). The two are the
    same computation at ``step == 0`` -- one on the host in Python, one
    in-jit via :func:`jem.base.calendar.gregorian_instant` -- so this function
    still gives the same value ``year_fraction`` gives at step 0, up to the
    float32-vs-float64 rounding between a host Python float and a traced JAX
    array (the same precision gap that existed before this fix), which is the
    property this function exists to keep. ``"365_day"`` is unchanged: its
    year has no leap day, so the fixed-average division was already exact.
    (``"360_day"`` is not a calendar this function, or any other part of
    jem, accepts by name -- see :func:`days_per_year`.)

    Parameters
    ----------
    start_date : jax_datetime.Datetime
        The run's start date.
    calendar : str
        Calendar name as JCM spells it: ``"gregorian"`` or ``"365_day"``.

    Returns
    -------
    float

    """
    if float(days_per_year(calendar)) == 365.2425:
        when = start_date.to_pydatetime()
        is_leap = when.year % 4 == 0 and (when.year % 100 != 0 or when.year % 400 == 0)
        day_of_year = (when - datetime.datetime(when.year, 1, 1)).days
        seconds_into_day = when.hour * 3600 + when.minute * 60 + when.second
        year_length = 366.0 if is_leap else 365.0
        return float(day_of_year + seconds_into_day / SECONDS_PER_DAY) / year_length
    seconds_per_year = SECONDS_PER_DAY * float(days_per_year(calendar))
    # `seconds_since_new_year` already counts in the model calendar, so this
    # is strictly below 1; the modulo only guards the boundary against
    # rounding.
    return (seconds_since_new_year(start_date, calendar) / seconds_per_year) % 1.0


def _timedelta_seconds(delta: Any) -> int:
    """Return a ``jax_datetime`` days/seconds pair as an exact int64 count of seconds.

    ``jdt.Timedelta`` (and the ``.delta`` of a ``jdt.Datetime``) stores whole
    days and the seconds within the day separately, so this is exact -- unlike
    the float64-days arithmetic :meth:`TimeAxis.datetimes` used before
    jax-gcm#862 published :func:`jcm.predictions.output_time_labels`. Plain
    ``int`` (arbitrary precision), not ``int32``: the caller multiplies this by
    a record count that can run to the millions over a multi-decade run, which
    would overflow ``int32`` well before it overflows an unbounded seconds
    count.
    """
    return int(np.asarray(delta.days)) * int(SECONDS_PER_DAY) + int(
        np.asarray(delta.seconds)
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
        Length of the year in days for the run's calendar, from this module's
        own :func:`days_per_year`. Static.
    start_day, start_second : int
        Days and seconds since the Unix epoch of ``start_date`` (a
        ``jax_datetime.Datetime``'s ``.delta.days``/``.delta.seconds``).
        Static. Used only by :attr:`year_fraction` on the ``"gregorian"``
        calendar (identified by ``days_per_year == 365.2425``, the same
        sentinel :func:`seconds_since_new_year` tests) to compute the exact
        Gregorian day-of-year and leap-year status of this step, via
        :mod:`jem.base.calendar`. The ``365_day`` calendar does not read
        these fields at all -- its year has no leap day, so the existing
        modular-arithmetic path below is already exact.

    """

    step: jax.Array
    sim_time: jax.Array
    dt: float = struct.field(pytree_node=False)
    year_offset_seconds: float = struct.field(pytree_node=False)
    days_per_year: float = struct.field(pytree_node=False)
    start_day: int = struct.field(pytree_node=False, default=0)
    start_second: int = struct.field(pytree_node=False, default=0)

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

        The upper bound is enforced explicitly, not just arithmetically: an
        instant a single second before a year boundary can be exactly ``1``
        short of the excluded value by less than the array's own float32
        precision at ``1.0`` (its ULP, ``2**-24``), so the correctly rounded
        float32 result of the true, sub-``1`` fraction is ``1.0`` itself --
        confirmed at ``2000-12-31T23:59:59`` on ``"gregorian"``. No
        reordering of the arithmetic below avoids this (it is a
        representable-range limit of the dtype, not a computation bug), so
        the very end of this property clamps against it once, for both
        branches.

        Zero is 00:00 on 1 January. This is what a monthly climatology is
        interpolated with (``jem.utils.cycles.evaluate_cyclic_linear``) -- the
        slab ocean's ``sst_climatology``/``q_flux``, the sea-ice model's
        ``ice_climatology``, and the slab land model's surface temperature,
        snow and soil water at both ends of a step.

        **On ``"gregorian"``** (``days_per_year == 365.2425``, the same
        sentinel :func:`seconds_since_new_year` tests -- the true average
        Gregorian year, not any particular year's length; see
        :func:`days_per_year`), this is computed from the **exact** proleptic
        Gregorian date of this step -- real leap years, not the 365.2425-day
        average -- via :func:`jem.base.calendar.gregorian_instant` and
        :func:`~jem.base.calendar.gregorian_day_of_year`. This closes a
        confirmed phase-drift bug (the 2026-09 jax-gcm-878 migration review):
        dividing elapsed seconds by the *average* year length, as every
        calendar here used to, is only ever exactly right at a handful of
        instants and drifts by up to a full day within the run (peaking at
        every year boundary, since the atmosphere's real Gregorian calendar
        and this fixed-average one fall on opposite sides of 31 December for
        most of the year) -- up to +1.48 days over 400 years, and +0.757 days
        within the single leap year 2000. Every ``"gregorian"``-calendar
        consumer of ``year_fraction`` (the slab models, listed above) reads
        this property, so the fix applies to all of them without their own
        code changing.

        This intentionally does **not** match jax-gcm's own ``wrap_year``
        step function bit for bit: jax-gcm samples a climatology by
        interpolating between the two nearest of a fixed number of samples
        *within whichever year length the current year actually has*, while
        this property (and the slab models built on it) keep sampling by a
        single ``[0, 1)`` fraction of the year that never itself changes
        length -- interpolating a climatology by ``year_fraction`` some 0.001
        further into 2000's 366 days is a very slightly different instant
        than the equivalent step of a 365-day year, whereas jax-gcm's own
        ``wrap_year`` is calibrated per actual year length so that "day 60"
        always means 1 March regardless of leap years. Changing the slabs to
        match would change which day of the climatology they sample on a leap
        year -- a science change, not a bug fix -- so it is deliberately out
        of scope here: this property only removes the *drift*, not the
        (separate, and much smaller) day-of-climatology convention
        difference from jax-gcm's own scheme.

        **On ``"365_day"``** (the only other calendar name a
        :class:`~jem.base.coupler.Coupler` accepts -- ``"360_day"`` is not,
        and never has been, a calendar any part of jem supports by name; see
        :func:`days_per_year`) this is unchanged from before the 2026-09
        review: that calendar has no leap day, so a fixed average-year-length
        division was already exact, and the modular integer-step reduction
        below (kept for its float32-precision benefit over many decades of
        simulated time -- see the note in its own branch) still applies.
        """
        if self.days_per_year == 365.2425:
            fraction = self._gregorian_year_fraction()
        else:
            steps_per_year = self.seconds_per_year / self.dt
            if float(steps_per_year).is_integer():
                # Precision note: `sim_time` is a float32 array unless x64 is
                # enabled, and float32 resolves only ~7 digits, so after a
                # century of simulated time (3e9 s) it is quantised to hundreds
                # of seconds. When the coupling step divides the year exactly
                # (the usual case: daily steps in a 365-day year) the step count
                # is reduced modulo the steps per year in exact integer
                # arithmetic first, so the fraction keeps full float32 precision
                # (a few seconds) for runs of any length. Otherwise the seconds
                # are used directly and precision degrades with run length.
                seconds_into_year = self.year_offset_seconds + (
                    jnp.mod(self.step, int(steps_per_year)) * self.dt
                )
            else:
                seconds_into_year = self.year_offset_seconds + self.sim_time
            # Reduce in seconds before dividing: the modulo of a quotient near
            # 1.0 keeps only the absolute float32 precision of that quotient
            # (~1e-7), whereas the remainder in seconds is exact for
            # whole-second steps and the division then has full relative
            # precision.
            fraction = (
                jnp.mod(seconds_into_year, self.seconds_per_year)
                / self.seconds_per_year
            )
        # Clamp against the one edge float32 (the dtype throughout, unless
        # x64 is enabled) cannot represent: an instant a single second before
        # a year boundary can be `< 1` by less than float32's own precision
        # at 1.0 (ULP `2**-24`), so the CORRECTLY ROUNDED float32 value of
        # the true fraction is `1.0` even though the exact fraction never
        # reaches it (confirmed: `2000-12-31T23:59:59` on `"gregorian"`,
        # `(365*86400 + 86399) / (366*86400)` is `1` short of `1` by
        # `1 / (366*86400)`, well under half a ULP at 1.0). No reordering of
        # the arithmetic above changes this -- it is a representable-range
        # limit of the dtype, not a computation bug -- so it is clamped
        # explicitly here, in the one place both branches return through,
        # rather than chased separately in each. `nextafter` rather than a
        # fixed epsilon so this is exact for the array's own dtype, float32
        # or float64 alike.
        one = jnp.asarray(1.0, dtype=fraction.dtype)
        return jnp.minimum(fraction, jnp.nextafter(one, jnp.zeros_like(one)))

    def _gregorian_year_fraction(self) -> jax.Array:
        """Return the exact Gregorian ``year_fraction``; see that property.

        ``dt`` is static (never traced) and is always a whole number of
        seconds -- the coupler only ever builds whole-second clocks
        (``Coupler._element_timestep`` refuses a sub-timestep that is not) --
        so ``record_seconds`` below is exact. ``step`` is the only traced
        quantity, and :func:`~jem.base.calendar.gregorian_instant` is a limb
        multiply-then-divide, exact for any ``step`` an int32 can hold, up to
        the exact int32 day-count limit its own docstring derives (about 5.87
        million simulated years) -- ``jem.driver.run_chunked`` refuses a run
        past that limit before it ever reaches here (2026-09 migration
        review, round 2, finding B1); this property itself has no run length
        to check against, since ``step`` is traced.
        """
        from jem.base.calendar import (
            gregorian_day_of_year,
            gregorian_instant,
            gregorian_ymd_from_days,
            is_leap_year,
        )

        record_seconds = round(self.dt)
        days, seconds = gregorian_instant(
            self.step, record_seconds, self.start_day, self.start_second
        )
        year, month, day = gregorian_ymd_from_days(days)
        day_of_year = gregorian_day_of_year(year, month, day)
        year_length = jnp.where(is_leap_year(year), 366, 365)
        return (day_of_year + seconds / SECONDS_PER_DAY) / year_length


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

    The labelling convention is JCM's: record ``k`` is the average over
    ``[start_date + k dt, start_date + (k+1) dt)`` and is labelled with the
    **midpoint** of that interval, ``start_date + (k + 1/2) dt``, as an exact
    ``datetime64[ms]`` on the proleptic Gregorian calendar whatever the
    coupler's own calendar is (a ``365_day`` coupler still writes real dates;
    the calendar governs only the seasonal cycle and forcing selection, not
    the output labels -- see :func:`jem.base.component.days_per_year`).
    :meth:`datetimes` implements exactly that and is the one place the
    convention is written down.

    Through jax-gcm PR 877 this reimplemented JCM's *labelling* arithmetic
    (an end-of-interval label, via a float64 days-since-epoch product that is
    not exact past a 128 ns ulp) rather than calling JCM, because the only
    public conversion at the time (``Model.date_from_sim_time``, jax-gcm#824)
    was JCM's *model clock* conversion -- exact but for a different quantity
    (forcing/physics dates from elapsed seconds) -- and not what the output
    files were labelled with, so adopting it here would only have merged with
    JCM's output when the coupling step happened to be a power-of-two fraction
    of a day. jax-gcm PR 878 published the labelling conversion itself,
    ``jcm.predictions.output_time_labels`` (closing jax-gcm#862), and moved
    JCM's own averaged output from an end-of-interval label to a
    **midpoint-of-interval** one (`docs/source/v2_to_v3.rst`, "One real
    datetime clock"). :meth:`datetimes` now calls that function directly
    instead of reimplementing it: every component computes its record's exact
    interval bounds with ``jax_datetime`` (whole-second arithmetic, so the
    bounds themselves are always exactly representable), converts them to
    ``datetime64[ms]`` with :func:`~jcm.predictions.output_time_labels`, and
    takes the midpoint by plain NumPy ``datetime64``/``timedelta64``
    arithmetic on the millisecond values -- the same two-step recipe
    ``ModelPredictions.to_xarray`` uses for its own averaged output, which is
    what lets a midpoint fall on a half second for an odd-length interval
    without ``jax_datetime.Timedelta`` (whole-seconds-only) ever having to
    represent one. This is exact for *any* coupling step, not merely a
    power-of-two fraction of a day, so the jax-gcm#862 gap this class existed
    to work around is closed rather than merely documented.

    The leap-day / calendar-months caveat this class used to carry (a
    ``365_day`` coupler's *labels* falling a day behind the model calendar
    from the first 29 February the run reaches) is unaffected by the
    midpoint change and is not repeated here in full; see
    :func:`jem.accumulate.monthly_mean`'s **Leap days** section, which is
    where a user actually meets it.

    Attributes
    ----------
    start_date : jdt.Datetime
        The run's start date.
    steps : numpy.ndarray
        int array of coupled-step indices, one per record.
    dt : jdt.Timedelta
        Coupling timestep.
    calendar : str
        Calendar name as JCM used to spell it (``"365_day"``, ``"gregorian"``).

    """

    start_date: jdt.Datetime
    steps: Any
    dt: jdt.Timedelta
    calendar: str

    def __len__(self) -> int:
        """Return the number of output records."""
        return len(self.steps)

    def datetimes(self) -> np.ndarray:
        """Return the record labels as exact ``datetime64[ms]`` (interval midpoints).

        See the class docstring for the convention and why this calls
        :func:`jcm.predictions.output_time_labels` rather than reimplementing
        it. The interval bounds are computed in exact int64 seconds (never a
        floating day count, and never an ``int32`` total that a
        many-thousand-record run could overflow) before being split back into
        the whole day/second pair ``jax_datetime.Timedelta`` needs.
        """
        from jcm.predictions import output_time_labels

        dt_seconds = _timedelta_seconds(self.dt)
        steps = np.asarray(self.steps, dtype=np.int64)
        interval_start_seconds = steps * dt_seconds
        interval_end_seconds = interval_start_seconds + dt_seconds

        def _exact_datetime(seconds_since_start: np.ndarray) -> jdt.Datetime:
            days, seconds = np.divmod(seconds_since_start, int(SECONDS_PER_DAY))
            return self.start_date + jdt.Timedelta(
                days=jnp.asarray(days, dtype=jnp.int32),
                seconds=jnp.asarray(seconds, dtype=jnp.int32),
            )

        bounds_start = output_time_labels(_exact_datetime(interval_start_seconds))
        bounds_end = output_time_labels(_exact_datetime(interval_end_seconds))
        # Integer-divide the millisecond *bounds*, not a jax_datetime
        # Timedelta: a whole-second interval's length in milliseconds is
        # always even (a multiple of 1000), so this only ever produces a
        # half-second midpoint for an odd number of seconds, exactly as
        # jax-gcm's own `ModelPredictions.to_xarray` computes it.
        midpoints: np.ndarray = bounds_start + (bounds_end - bounds_start) // 2
        return midpoints

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
    """Optional: convert stacked diagnostics to ``xarray``.

    The return value is normally one :class:`xarray.Dataset`, keyed in the
    coupler's output under the component's registered name. A component that
    is itself a coupled model -- a :class:`~jem.base.coupler.Coupler` nested
    inside a slower one -- has no single dataset to return: it holds several
    components of its own, each with its own variables and its own sampling
    rate. It may therefore return a **mapping** of name to dataset instead,
    which the outer coupler flattens into its own result under those names
    (and refuses if one of them collides with a name already there). A
    wrapper around any other multi-model system can do the same.
    """

    def to_xarray(
        self, diagnostics: Diagnostics, time: TimeAxis
    ) -> xr.Dataset | Mapping[str, xr.Dataset]: ...


@runtime_checkable
class SupportsCheckpoint(Protocol):
    """Optional: components whose carry is not a plain pytree of arrays (Veros)."""

    def save_carry(self, carry: Carry, directory: Path) -> None: ...

    def load_carry(self, directory: Path) -> Carry: ...


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


@runtime_checkable
class SupportsInternalStepping(Protocol):
    """Optional: report how many of a component's own internal steps one ``step()`` call makes.

    A component with its own inner timestep (JCM, Veros) may keep raw
    counters of its own -- an internal step count, or a clock like JCM's own
    ``RunState`` -- that advance faster than the coupled step calling it,
    entirely inside that component's own implementation and invisible to the
    coupler's workflow structure (a multiplicity, a nested :class:`Coupler`).
    If such a counter is itself an ``int32``, or feeds a product that is
    (JCM's ``expected_step = time.step * self._inner_steps()`` in
    ``JCMComponent._report_authoritative_clock_drift``), it needs the same
    int32-overflow protection the coupler's own counters get -- but nothing
    outside that component can know the rate to protect it at without being
    told.

    ``internal_steps_per_call`` is that rate: how many of *this* component's
    own internal timesteps happen inside one call to :meth:`Component.step`.
    :func:`jem.driver._max_element_rate` multiplies it in for every element
    clock that calls this component (a workflow multiplicity, or -- via the
    recursion -- a nested coupler's own substep rate), so
    :func:`jem.driver.run_chunked`'s up-front int32 check
    (``_check_step_counters_fit_int32``) covers it too, the same way it
    covers a plain workflow multiplicity (2026-09 review, round 3 follow-up,
    finding 7). A component that does not implement this capability is
    assumed to advance no faster than the calls it receives (rate 1) -- the
    same as every component before this capability existed; a component that
    *does* keep such counters but does not report them here is simply not
    protected, the same way an unbound component's own clock mismatch is
    only ever caught if it implements :class:`SupportsBind`.
    """

    def internal_steps_per_call(self) -> int: ...


# An exchanger moves information between components. It receives the mapping
# of component carries (a fresh dict, so adding or replacing entries never
# mutates the coupler's input) and the current clock, and returns the mapping
# to continue with. It must not mutate the carries it receives in place;
# build new ones with ``dataclasses.replace`` / ``.replace`` and return them.
# The clock is passed so an exchanger can implement time-dependent coupling
# (lagged exchange, ramped forcing) without keeping state of its own.
Exchanger = Callable[[dict[str, Carry], CouplingTime], dict[str, Carry]]
