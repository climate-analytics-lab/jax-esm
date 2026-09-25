"""Exact proleptic-Gregorian calendar arithmetic, vendored and jit-safe.

``jem.base`` and ``jem.accumulate`` are deliberately jcm-free at import time
(a Veros- or slab-only coupled run never imports jax-gcm at all), so the
handful of Gregorian primitives both :func:`jem.accumulate.monthly_mean` (the
in-scan monthly-bin rule, item A of the 2026-09 jax-gcm-878 migration review)
and :class:`jem.base.component.CouplingTime` (the exact ``year_fraction``,
item B of the same review) need are **vendored here** rather than imported
from ``jcm.date`` -- the same reason ``jem.base.component.days_per_year``
already keeps its own copy of the days-per-year table rather than importing
jax-gcm's.

:func:`gregorian_ymd_from_days` and :func:`is_leap_year` are byte-for-byte
the algorithm in ``jcm/date.py``'s ``gregorian_ymd_from_days`` /
``is_leap_year`` (Fliegel, H. F., & Van Flandern, T. C. (1968); see also
https://aa.usno.navy.mil/faq/JD_formula) -- copied rather than reimplemented
so that the two packages can never silently disagree about what a given day
number means. ``gregorian_day_of_year`` is the same computation as jcm's
private ``_gregorian_day_of_year``, made public here because JEM's own
``year_fraction`` needs it outside this module.
``jem/base/calendar_test.py`` cross-checks all three against
``jcm.date`` and against ``pandas`` over 400+ years, century leap-year rules
included, so a divergence between the two copies is a test failure rather
than a silent drift.

:func:`gregorian_instant` is the one piece of arithmetic new to this module:
the int32-safe "reduce-before-multiply" decomposition that turns a record
counter that can run into the millions over a multi-century run into the
(days, seconds)-since-epoch of one instant within that record, without ever
forming a product that could overflow ``int32``. Both callers use it --
:func:`jem.accumulate._gregorian_month_rule` at the record's **midpoint**
(the bin a record counts in, matching the midpoint labelling convention
:class:`jem.base.component.TimeAxis` writes -- see that module's decision
record) and :class:`~jem.base.component.CouplingTime` at the record's
**start** (``year_fraction`` is defined at the start of a step, matching its
pre-existing convention on the ``365_day``/``360_day`` calendars).
"""

from __future__ import annotations

import math

import jax.numpy as jnp

SECONDS_PER_DAY = 86_400

# Julian Day Number of 1970-01-01 -- the Unix epoch. Vendored verbatim from
# `jcm.date._UNIX_EPOCH_JDN`.
_UNIX_EPOCH_JDN = 2440588


def gregorian_ymd_from_days(days_since_epoch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert days-since-1970 to a proper Gregorian (year, month, day).

    Uses the Fliegel & Van Flandern (1968) integer algorithm -- JAX-friendly
    (only int arithmetic, no Python ``datetime``) and exact for any year in
    the proleptic Gregorian calendar. References:
        - Fliegel, H. F., & Van Flandern, T. C. (1968).
        - https://aa.usno.navy.mil/faq/JD_formula

    Vendored verbatim (variable names included) from ``jcm.date`` -- see the
    module docstring for why this package keeps its own copy rather than
    importing jax-gcm's. Variable names match the published algorithm and are
    intentionally not renamed (``# noqa: E741`` for the lowercase ``l``).
    """
    jdn = days_since_epoch + _UNIX_EPOCH_JDN
    l = jdn + 68569                              # noqa: E741
    n = (4 * l) // 146097
    l = l - (146097 * n + 3) // 4                # noqa: E741
    i = (4000 * (l + 1)) // 1461001
    l = l - (1461 * i) // 4 + 31                 # noqa: E741
    j = (80 * l) // 2447
    day = l - (2447 * j) // 80
    l = j // 11                                  # noqa: E741
    month = j + 2 - 12 * l
    year = 100 * (n - 49) + i + l
    return year, month, day


def is_leap_year(year: jnp.ndarray) -> jnp.ndarray:
    """Gregorian leap-year predicate (returns a JAX boolean array).

    Vendored verbatim from ``jcm.date.is_leap_year``.
    """
    return ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)


def gregorian_day_of_year(year: jnp.ndarray, month: jnp.ndarray, day: jnp.ndarray) -> jnp.ndarray:
    """Return the zero-indexed day-of-year for a Gregorian date (Jan 1 -> 0).

    Same computation as jcm's private ``jcm.date._gregorian_day_of_year``, but
    generalized to a ``year``/``month`` of *any* shape (jcm's own version
    broadcasts a length-12 table against ``is_leap_year(year)`` and only works
    for a scalar ``year``, which is all jcm itself ever calls it with; this
    module's ``jem.accumulate`` cross-check in ``test_calendar.py`` needs it
    over whole arrays of dates, so the leap offset is selected by ``month``
    directly -- ``month >= 3`` (March on) is exactly the condition jcm's
    ``jnp.arange(12) >= 2`` selects once indexed by ``month - 1`` -- rather
    than built as a length-12 table first). Public here because
    :class:`jem.base.component.CouplingTime`'s exact ``year_fraction`` needs
    it outside this module.
    """
    days_in_month = jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    cum_no_leap = jnp.concatenate([jnp.array([0]), jnp.cumsum(days_in_month)[:-1]])
    leap_offset = jnp.where(month >= 3, is_leap_year(year).astype(jnp.int32), 0)
    return cum_no_leap[month - 1] + leap_offset + (day - 1)


def gregorian_instant(
    record: jnp.ndarray,
    record_seconds: int,
    start_days: int,
    start_seconds: int,
    offset_seconds: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the (days, seconds)-since-epoch of one instant within a record.

    ``record`` counts records of ``record_seconds`` each from the run's
    start (``start_days``/``start_seconds`` -- a ``jax_datetime.Datetime``'s
    ``.delta.days``/``.delta.seconds``, i.e. days and seconds since the Unix
    epoch); the instant returned is ``offset_seconds`` into record ``record``,
    e.g. ``0`` for the record's start or ``record_seconds // 2`` for its
    midpoint (see the module docstring for why the two callers use different
    offsets).

    The naive way to find it -- multiply the record counter by
    ``record_seconds`` and add the offsets -- costs one product that grows
    with the run: JAX indices are int32 by default, and a whole-run count of
    seconds passes 2**31 after a few decades of daily coupling (sooner for
    finer coupling). The fix, used throughout jax-gcm's own v3 clock
    (``jcm.date``) and here: reduce the **traced** record counter modulo a
    small, **static** period *before* multiplying, and let a whole number of
    *days* (unbounded, but growing only with elapsed simulated time in days,
    not with record count) carry the rest.

    Concretely: let ``g = gcd(record_seconds, 86400)``, ``P = 86400 // g``
    and ``D = record_seconds // g``; ``P`` records span exactly ``D`` days
    (``P * record_seconds == D * 86400`` by construction of ``g``), so
    ``record = q*P + r`` with ``0 <= r < P`` means record ``record`` sits
    ``q*D`` whole days plus ``r`` records into the run. Only ``r`` -- bounded
    by ``P <= 86400`` -- is ever multiplied by ``record_seconds``, so the
    product is bounded by ``86400 * D`` regardless of how long the run is
    (int32-safe for ``D`` up to about 24855, i.e. any coupling timestep up to
    tens of years -- see jax-gcm's own migration notes for the general form
    of this bound). ``q * D``, the day count, is a plain int32 multiply of
    two quantities that both grow only with elapsed *days*, which is what
    stays representable for a run of any realistic length.

    Parameters
    ----------
    record : jax.Array
        int32 (or castable) record counter, traced.
    record_seconds : int
        Length of one record, in seconds. Static (a Python int, not traced).
    start_days, start_seconds : int
        Days and seconds since the Unix epoch of record 0's start. Static.
    offset_seconds : int, optional
        Seconds into record ``record`` at which to evaluate the instant.
        Static; must satisfy ``0 <= offset_seconds <= record_seconds`` for
        the result to still describe a moment within (or at the boundary of)
        that record -- both callers pass ``0`` or ``record_seconds // 2``.

    Returns
    -------
    days, seconds : jax.Array
        int32 arrays: whole days since the Unix epoch, and the seconds within
        that day, of the requested instant.

    """
    g = math.gcd(record_seconds, SECONDS_PER_DAY)
    period_records = SECONDS_PER_DAY // g
    period_days = record_seconds // g
    record = jnp.asarray(record, dtype=jnp.int32)
    whole_periods, remainder_records = jnp.divmod(record, period_records)
    extra_days, seconds = jnp.divmod(
        remainder_records * record_seconds + offset_seconds + start_seconds,
        SECONDS_PER_DAY,
    )
    days = start_days + whole_periods * period_days + extra_days
    return days, seconds
