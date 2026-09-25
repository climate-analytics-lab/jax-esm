"""Reductions of a run's diagnostics computed *inside* the coupled scan.

``Coupler.generate_trajectory_function(iterations, accumulate=(init, update))``
runs ``update`` on every step's diagnostics inside the ``lax.scan`` body and
returns only the accumulator, never the stacked per-step output. This module
packages the reductions a long run almost always wants -- a mean over each
bin of a fixed set of bins -- as such a pair:

- :func:`monthly_mean`, the calendar months -- twelve bins that composite a
  multi-year run into a climatology, or (``total_time=`` / ``n_months=``) one
  bin per month the run passes through;
- :func:`windowed_mean`, ``n_windows`` windows measured from the run's own
  start date -- of one fixed length, which is what a sub-seasonal forecast is
  scored on (pentads, weeks), or of a repeating *pattern* of lengths.

Both are :func:`_build_binned_mean` with a different step-to-bin rule, so there is
one running-sum-and-count implementation and one :meth:`BinnedMean.finalize`;
and both rules are :func:`_variable_window_rule` with different boundaries,
so there is one piece of bin arithmetic. The two are **not**
interchangeable, and the arguments of that rule are exactly how they differ:
a calendar month is phased to where the run's start date falls *in the
calendar* and is closed at its start, so that it agrees with
``groupby("time.month")`` of the written output (for a run whose labels cross
no Gregorian 29 February -- the bins are the model calendar's months while the
labels are proleptic Gregorian, which :func:`monthly_mean` explains under
**Leap days**); a window is measured from the run's start with no phase at
all and is closed at its end, so that the first pentad is days 1 to 5.
Handing :func:`month_lengths` to :func:`windowed_mean` is therefore calendar
months only for a run starting at 00:00 on 1 January -- :func:`monthly_mean`
is the one that knows where in the calendar the run began.

Why it is in the scan at all. A twelve-month run reduced on the host must
hold every step's diagnostics until the chunk ends, which for an atmosphere
is the largest array in the run by a wide margin; and a run chunked *by
month* must compile a 28-, a 30- and a 31-day trajectory, because
``iterations`` is static. Accumulating inside the scan does neither: one
compiled trajectory of whatever length suits the machine, and a fixed-size
``(n_bins, ...)`` accumulator that a chunked run threads from call to call.

Both are ordinary JAX: the accumulator is a pytree in the scan carry, so
``jax.grad`` of a binned mean with respect to a component parameter flows
through the reduction exactly as it flows through the trajectory. See the
worked calibration example in ``docs/source/design/architecture.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, NamedTuple

import datetime
import math

import jax
import jax.numpy as jnp
import numpy as np

from jem.base.calendar import gregorian_instant, gregorian_ymd_from_days, max_safe_record
from jem.base.component import (
    CouplingTime,
    Diagnostics,
    days_per_year as jem_days_per_year,
    parse_duration_days,
)

#: The accumulator a :class:`BinnedMean` carries: an ``(n_bins, ...)`` running
#: sum shaped like one step's diagnostics, and the count of records that
#: landed in each bin. The counts are a single ``(n_bins,)`` array when every
#: component of the model produces one record per coupled step, and one array
#: per component -- shaped ``(n_bins, *the component's sub-step axes)`` --
#: when any of them produces more, because components that record at
#: different rates then fill different bins as a coupled step is folded in
#: (see :func:`_record_axes`).
BinnedAccumulator = tuple[Any, Any]

#: Number of calendar months in a year, and the number of bins
#: :func:`monthly_mean` accumulates into unless it is asked for one bin per
#: month of the run (``total_time=`` / ``n_months=``). In the twelve-bin form
#: the leading axis of everything :meth:`BinnedMean.finalize` returns is
#: January first; in the sequential form it starts with the month the run
#: starts in.
MONTHS_PER_YEAR = 12

_SECONDS_PER_DAY = 86400

#: Month lengths, in days, of the fixed-length calendars a static month table
#: can be built for, keyed by the length of their year. :func:`month_lengths`
#: is the public way to read it. :func:`jem.base.component.days_per_year`
#: defines only ``365_day`` (and ``gregorian``, whose leap years make no such
#: table possible), so ``360_day`` is here because the table is the same kind
#: of object, not because a ``Coupler`` can be built with it today.
_MONTH_LENGTHS: dict[int, tuple[int, ...]] = {
    365: (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    360: (30,) * MONTHS_PER_YEAR,
}


def month_lengths(calendar_or_coupler: Any) -> tuple[int, ...]:
    """Return the twelve month lengths, in days, of a fixed-length calendar.

    This is the table :func:`monthly_mean` bins on, made public so that an
    analysis can weight or label months without rebuilding it -- a monthly
    mean weighted into an annual one, say::

        from jem.accumulate import month_lengths, monthly_mean

        days = jnp.asarray(month_lengths(coupler))
        annual = jnp.sum(climatology * days[:, None, None], 0) / days.sum()

    The lengths are plain days, and always **January first**, whatever the
    run's start date. That is what makes them a poor window pattern: handed
    to :func:`windowed_mean` they are measured from the run's own start date
    with no phase, so ``windowed_mean(coupler, month_lengths(coupler), ...)``
    is calendar months only for a run starting at 00:00 on 1 January, and for
    a run starting on 1 July it bins the first 31 days together, then 28, and
    so on. Use ``monthly_mean(coupler, total_time=...)`` for one bin per
    calendar month of a run: it rotates this table to the month the run starts
    in and phases it to the start date.

    Parameters
    ----------
    calendar_or_coupler : Coupler or str or float
        The coupled model whose calendar to tabulate (anything carrying a
        ``days_per_year``, which is what a
        :class:`~jem.base.coupler.Coupler` carries), the name of a calendar
        (``"365_day"``), or a year length in days. A name is resolved through
        :func:`jem.base.component.days_per_year`, the single table every
        calendar-aware part of JEM (this one included) reads.

    Returns
    -------
    tuple of int
        Twelve month lengths in days, January first, summing to the year.

    Raises
    ------
    NotImplementedError
        If the calendar's year is not a fixed whole number of days with fixed
        month lengths -- ``gregorian``, whose leap years change the table from
        year to year, so no *table* of twelve lengths can describe it (this
        function only ever returns one fixed table). This is not a limit on
        binning calendar months on ``gregorian`` in general:
        :func:`monthly_mean` computes them directly from the exact Gregorian
        calendar instead of from a table, entirely in-scan, and does not call
        this function to do it -- see that function's own documentation. This
        function stays table-only because its whole purpose is a table other
        code (the weighting example above, :func:`windowed_mean`) can read.

    """
    # A coupler carries its year length; a bare number is one already.
    days_per_year = getattr(calendar_or_coupler, "days_per_year", calendar_or_coupler)
    if isinstance(days_per_year, str):
        days_per_year = jem_days_per_year(days_per_year)
    length = int(days_per_year)
    if length != days_per_year or length not in _MONTH_LENGTHS:
        raise NotImplementedError(
            f"Calendar months need a calendar whose year is a fixed whole "
            f"number of days with fixed month lengths, so that the table of "
            f"them is a constant; this year is {days_per_year} days. "
            f"Supported: {sorted(_MONTH_LENGTHS)} days. A Gregorian "
            "calendar's leap years change the table from year to year, which "
            "a static table cannot express -- but `monthly_mean(coupler, ...)` "
            "itself handles `calendar='gregorian'` directly (exactly, in-scan, "
            "real leap years included), without going through this table at "
            "all; call it directly rather than trying to build one here."
        )
    return _MONTH_LENGTHS[length]


def _variable_window_rule(
    boundaries_seconds: np.ndarray,
    offset_seconds: int,
    inclusive: Literal["left", "right"],
) -> Callable[[jnp.ndarray, int], jnp.ndarray]:
    """Return the ``bin_of_record`` rule for bins of the given lengths.

    :func:`windowed_mean`'s bin arithmetic: bins laid end to end, cycling for
    as long as the run lasts, a record belonging to the bin its interval's
    *end* falls in (in elapsed run-time, independent of what instant the
    record is actually labelled with -- see :func:`windowed_mean`'s own
    docstring for why the two are kept independent). This was also
    :func:`monthly_mean`'s bin rule before the 2026-09 migration review moved
    that function to bin by a record's **midpoint** instead
    (:func:`_midpoint_month_rule`, :func:`_gregorian_month_rule`), to keep
    pace with :class:`~jem.base.component.TimeAxis`'s own move to a
    midpoint output label -- a half-record shift this function's own
    boundaries-to-record-counts conversion (see the Notes below) cannot
    express, which is why that rebinding needed a new function rather than a
    third ``inclusive`` mode here.

    Parameters
    ----------
    boundaries_seconds : numpy.ndarray
        The **ends** of the bins, in whole seconds from the start of the
        pattern: the cumulative sum of the bin lengths, strictly increasing,
        one entry per bin. The last entry is where the bins end and the
        period they repeat with, which is what makes a run longer than the
        accumulator wrap; it must be a whole number of coupling steps, so a
        caller whose last bin does not end on one extends that bin itself
        (:func:`monthly_mean`'s sequential form does, and says why).
    offset_seconds : int
        Where the run's start date sits in that pattern -- 0 for windows the
        run itself defines, and the run's offset into the calendar year for
        calendar months, so that a run starting on 1 July fills the July bin
        first.
    inclusive : {"left", "right"}
        Which end of a bin is included in it, in the sense of
        :func:`pandas.date_range`'s ``inclusive``. A record labelled exactly
        on a boundary between two bins belongs to the bin that includes that
        end.

        Take 5-day bins. ``inclusive="left"`` makes the first bin the labels
        in ``[day 0, day 5)`` and the second ``[day 5, day 10)``, so a label
        of exactly day 5 opens the second bin. ``inclusive="right"`` makes
        the first bin ``(day 0, day 5]`` and the second ``(day 5, day 10]``,
        so a label of exactly day 5 closes the first bin.

        Only ``inclusive="right"`` is exercised in production now (by
        :func:`windowed_mean`): a record's interval-end position of exactly
        day 5 closes the first 5-day window rather than opening the second,
        so the first window is the records covering days 0-5. ``"left"``
        remains supported as the general case (and is what
        :func:`monthly_mean` used, on this same function, before the
        2026-09 migration moved it to :func:`_midpoint_month_rule` /
        :func:`_gregorian_month_rule` instead).

    Returns
    -------
    callable
        ``(record, record_seconds) -> int32 bin index``, total by
        construction (see :func:`_build_binned_mean`).

    Notes
    -----
    **Why the boundaries are converted to record counts.** Record ``k`` of
    length ``r`` is labelled at ``(k + 1)·r`` seconds from the start of the
    run, so the obvious rule -- multiply the record counter by ``r`` and look
    the result up in a table of seconds -- costs one multiplication whose
    product grows with the run. JAX indices are int32 by default, and a
    product of seconds passes 2^31 after 68 years of simulated time, at which
    point the bins would silently wrap to nonsense. Reducing the record
    counter modulo the records in one period *before* multiplying bounds the
    product by the period rather than by the run, which the previous
    fixed-window arithmetic did too; converting the boundaries to record
    counts on the host, in int64, removes the multiplication from the traced
    code altogether and so bounds nothing by int32 but the record counter
    itself. The conversion is exact because every boundary is a whole number
    of coupling steps and a record is a whole division of one.

    """
    if inclusive not in ("left", "right"):
        # `Literal` is a promise to the type checker, not a runtime check, so a
        # misspelling ("rigth") would otherwise be read as "left" by the
        # comparison below and silently shift every bin boundary by a record.
        raise ValueError(
            f'inclusive must be "left" or "right"; got {inclusive!r}.'
        )
    boundaries = np.asarray(boundaries_seconds, dtype=np.int64)
    period = int(boundaries[-1])
    # The two conventions differ by one second of the label. A record's label
    # sits at the END of its interval, and `bin_of_record` counts how many
    # boundaries lie at or before `label + shift_seconds`. With 5-day bins and
    # daily records (record k labelled at day k + 1):
    #
    #   inclusive="right": shift -1 s. A label of day 5 is looked up a second
    #     before the first boundary, so no boundary precedes it -> bin 0; it
    #     is the LAST record of (day 0, day 5]. Day 6 -> bin 1.
    #   inclusive="left": no shift. A label of day 5 is looked up at the
    #     boundary itself, which now counts -> bin 1; it is the FIRST record
    #     of [day 5, day 10).
    #
    # In calendar months from 1 January with daily records, the record
    # labelled 1 February 00:00 is therefore January's last record under
    # "right" and February's first under "left".
    shift_seconds = -1 if inclusive == "right" else 0

    def bin_of_record(record: jnp.ndarray, record_seconds: int) -> jnp.ndarray:
        """Return the 0-based bin a record of ``record_seconds`` counts in.

        ``record`` counts records of that length from the start of the run.
        With daily records, 5-day bins and ``inclusive="right"``: record 0 is
        labelled day 1 and lands in bin 0; record 4 (day 5) is the last of
        bin 0; record 5 (day 6) is the first of bin 1; and in a 73-bin
        accumulator record 365 -- labelled day 366, one year on -- wraps to
        bin 0 again. Under ``inclusive="left"`` record 4 (day 5) is instead
        the first of bin 1.
        """
        records_per_period, remainder = divmod(period, record_seconds)
        # An invariant of the callers, not a user error: every builder checks
        # that the bins' period is a whole number of coupling steps, and the
        # coupler refuses a workflow whose sub-timestep is not a whole
        # division of one (`Coupler._element_timestep`, re-checked in
        # `_record_seconds`), so this division is exact. It is asserted rather
        # than assumed because a silent rounding here would drift every bin
        # boundary by a fraction of a record.
        assert remainder == 0, (
            f"a period of {period} s is not a whole number of "
            f"{record_seconds} s records"
        )
        # Where the boundaries sit on this component's record grid. The run's
        # offset into the pattern need not be a whole number of records, so it
        # is split into whole records (`shifted`, folded into the counter) and
        # a remainder (`phase`, folded into the boundaries); the ceiling is
        # then the first record whose label reaches the boundary.
        shifted, phase = divmod(offset_seconds + shift_seconds, record_seconds)
        # The last entry lands exactly on `records_per_period`: the period is
        # a whole number of records and `phase` is less than one, so the
        # ceiling cannot overshoot it, and every wrapped record has a bin.
        in_records = _ceil_div(boundaries - phase, record_seconds)
        boundary_records = jnp.asarray(in_records, dtype=jnp.int32)
        # `record + 1` because record k is labelled at the END of its own
        # interval. The modulo is what wraps a run longer than the pattern
        # and, with it, keeps the index inside the accumulator whatever the
        # run's length; it is also what keeps the arithmetic in int32.
        label = jnp.mod(
            jnp.asarray(record, dtype=jnp.int32) + (1 + shifted), records_per_period
        )
        return jnp.searchsorted(boundary_records, label, side="right").astype(
            jnp.int32
        )

    return bin_of_record


def _ceil_div(numerator: Any, denominator: int) -> Any:
    """Return ``ceil(numerator / denominator)`` in integer arithmetic.

    Python's ``//`` rounds towards negative infinity, so negating the
    numerator, floor-dividing, and negating the result rounds *up* instead:
    ``ceil(7 / 2) == -((-7) // 2) == 4``. It stays exact for the int64 arrays
    and Python ints this module works in, where ``math.ceil(a / b)`` would go
    through a float and lose precision past 2**53.
    """
    return -((-numerator) // denominator)


def _midpoint_month_rule(
    boundaries_seconds: np.ndarray, offset_seconds: int
) -> Callable[[jnp.ndarray, int], jnp.ndarray]:
    """Return :func:`monthly_mean`'s ``bin_of_record`` rule, binning by MIDPOINT.

    This is the fixed-calendar (``365_day``/``360_day``) counterpart of
    :func:`_gregorian_month_rule`, and exists as a separate function from
    :func:`_variable_window_rule` -- rather than a third mode of that one --
    because the two do genuinely different arithmetic. A record's *label* is
    now its interval's midpoint (:class:`~jem.base.component.TimeAxis`'s
    convention since jax-gcm v3, PR 878), which sits half a record off every
    boundary; :func:`_variable_window_rule` converts the bin boundaries to
    **record counts** up front specifically so the traced arithmetic is a
    cheap integer comparison, and that conversion is only exact for a
    whole-record shift (the end or the start of a record), not a half-record
    one. Reworking it to carry a fractional record shift would give up the
    one thing it is for; comparing in **seconds** instead, after the same
    reduce-before-multiply step that keeps :func:`_variable_window_rule`
    int32-safe, keeps the exactness and needs no record-count conversion at
    all -- see the Notes below for why that reduction is still int32-safe.
    :func:`windowed_mean` still uses :func:`_variable_window_rule` unchanged:
    its own bins are measured from the run's start with no phase, and its
    docstring's own convention (closed at the window's end) is independent of
    what label :class:`~jem.base.component.TimeAxis` happens to write on a
    record -- see that function's docstring.

    Parameters
    ----------
    boundaries_seconds : numpy.ndarray
        The **ends** of the months, in whole seconds from the start of the
        pattern (the cumulative month lengths); see
        :func:`_variable_window_rule`'s identically-named parameter, which
        this mirrors exactly.
    offset_seconds : int
        Where the run's start date sits in that pattern.

    Returns
    -------
    callable
        ``(record, record_seconds) -> int32 bin index``, closed at the start
        of a month: a midpoint falling exactly on a month boundary counts in
        the month it opens, matching ``groupby("time.month")``'s own
        ``pandas`` convention for a timestamp falling exactly at midnight on
        the 1st.

    Notes
    -----
    **Int32 safety.** Comparing directly in seconds -- as this function did
    before the 2026-09 migration review's item 2 -- bounds the traced
    arithmetic by ``period`` (the pattern's own span: a year for the
    twelve-bin form, or the whole ``n_months``/``total_time`` span for the
    sequential one), which is fine for a year but **not** for a sequential
    accumulator spanning decades: a 100-year, 365-day-calendar
    ``monthly_mean`` has ``period`` around 3.15e9 s, already past ``2**31``,
    so building ``boundaries_int32`` at all raised ``OverflowError`` past
    about 68 years of sequential bins. The fix compares in **days** instead:
    every month boundary but (possibly) the pattern's very last -- see
    below -- falls exactly at midnight, so a boundary comparison needs no
    finer resolution than a day, and a day count stays int32-safe up to
    about 5.87 million years (``2**31`` DAYS) rather than 68 (``2**31``
    SECONDS). The record's own day is computed with
    :func:`~jem.base.calendar.gregorian_instant`'s own int32-safe block
    decomposition (``start_days=0``, ``start_seconds=0``, since only "days
    since the pattern's own start" is wanted here, not since the epoch) --
    still necessary, and not merely a seconds-vs-days relabelling, because
    ``record_mod * record_seconds`` alone can overflow int32 for a long
    enough pattern exactly as it can in :func:`gregorian_instant`'s own
    docstring; ``record`` is reduced modulo ``records_per_period`` (an
    ordinary Python ``divmod``, so this itself never overflows regardless of
    how long the run is) before either function ever sees it, which is what
    keeps the elapsed time within one call ``gregorian_instant`` is asked to
    resolve bounded by one *pattern*, not by the run.

    ``boundary_days`` rounds every boundary **up** (``-(-boundaries //
    86400)``, exact integer ceiling division, the same idiom
    :func:`_ceil_div` in this module uses) rather than down. Every boundary
    but the last is already an exact day multiple, for which ceiling and
    floor agree, so this changes nothing for them; the pattern's very last
    boundary, however, is sometimes **not** a whole number of days (see
    :func:`monthly_mean`'s own construction of a sequential-form
    accumulator's ``boundaries``, which extends the last one to the next
    whole coupled step, "by less than one step" -- a coupling step that does
    not itself divide a day exactly leaves a fractional day there). A
    record's own midpoint is always *strictly* before ``period`` seconds (the
    reduction above guarantees it), so its day can equal
    ``floor(period / 86400)`` when the last boundary has a fractional day
    left over -- rounding that boundary *up* instead keeps the record's day
    strictly less than it, so ``searchsorted`` can never place a record past
    the last valid bin (an off-by-one this function would otherwise commit
    only in that specific edge case).

    **The half-second floor.** ``record_seconds // 2`` truncates a
    midpoint's fractional half-second down for a record of odd length. Every
    month boundary is a whole number of seconds (there is no such thing as
    half past midnight on the 1st in this calendar), so flooring a midpoint
    can only move it a half-second *away* from a boundary it has not yet
    reached, never across one it would otherwise have crossed -- the floored
    midpoint and the true one are always on the same side of every boundary.
    Comparing by day rather than by second does not change this: the
    half-second truncation can only move an instant within the same second,
    let alone the same day.

    """
    boundaries = np.asarray(boundaries_seconds, dtype=np.int64)
    period = int(boundaries[-1])
    # Ceiling, not floor -- see the Notes above for why the pattern's last
    # boundary specifically needs it. `boundary_days` stays int32-safe for
    # any period a real run's pattern spans (millions of years), unlike the
    # `boundaries_int32` (SECONDS) array this replaces, which is what
    # overflowed for a multi-decade sequential accumulator.
    boundary_days = jnp.asarray(-(-boundaries // _SECONDS_PER_DAY), dtype=jnp.int32)

    def bin_of_record(record: jnp.ndarray, record_seconds: int) -> jnp.ndarray:
        records_per_period, remainder = divmod(period, record_seconds)
        # See `_variable_window_rule`'s identical assertion: an invariant of
        # the callers (the coupler only ever builds a sub-timestep that is a
        # whole division of the coupled one), not a user error.
        assert remainder == 0, (
            f"a period of {period} s is not a whole number of "
            f"{record_seconds} s records"
        )
        # `record_mod` ranges over `[0, records_per_period)`, and both that
        # bound and `record_seconds` are plain Python ints here (not
        # traced), so -- unlike `gregorian_instant`'s own `record` argument,
        # which is traced and open-ended (a run's length is not known in
        # advance) -- this IS a case where the maximum value
        # `gregorian_instant` will ever be asked to resolve for this pattern
        # is knowable up front. Checking it here, in plain Python, is what
        # "raise a clear error at construction" means for this reduction:
        # `monthly_mean`'s caller finds out its pattern does not fit BEFORE
        # a bin index is ever silently wrong, rather than after.
        bound = max_safe_record(
            record_seconds, offset_seconds=offset_seconds + record_seconds // 2
        )
        if records_per_period - 1 > bound:
            raise ValueError(
                f"This monthly_mean's pattern needs {records_per_period} "
                f"{record_seconds} s records per cycle ({period} s total), "
                f"but gregorian_instant can only resolve up to {bound} of "
                f"them exactly for a record this long (see "
                "jem.base.calendar.max_safe_record) -- this pattern is too "
                "long, or its records too long, to bin exactly."
            )
        record_mod = jnp.mod(jnp.asarray(record, dtype=jnp.int32), records_per_period)
        day, _ = gregorian_instant(
            record_mod, record_seconds, 0, 0,
            offset_seconds=offset_seconds + record_seconds // 2,
        )
        return jnp.searchsorted(boundary_days, day, side="right").astype(jnp.int32)

    return bin_of_record


def _gregorian_month_rule(
    start_days: int,
    start_seconds: int,
    *,
    sequential: bool,
    y0: int = 0,
    m0: int = 0,
    n_bins: int = 12,
) -> Callable[[jnp.ndarray, int], jnp.ndarray]:
    """Return :func:`monthly_mean`'s ``bin_of_record`` rule on the Gregorian calendar.

    Unlike :func:`_midpoint_month_rule` and :func:`_variable_window_rule`,
    this needs no fixed table of month lengths and no "period" to reduce a
    record counter modulo: :func:`jem.base.calendar.gregorian_instant`
    already reduces the traced record counter int32-safely (see its
    docstring), and the record's real Gregorian month is then read directly
    off the *exact* calendar date of its own midpoint via
    :func:`~jem.base.calendar.gregorian_ymd_from_days` -- real leap years,
    not a fixed 365- or 360-day table. This is what makes a jax-gcm-coupled
    run's ``monthly_mean`` bins equal ``groupby("time.month")`` (or
    ``groupby(["time.year", "time.month"])``) of its own written output *by
    construction*, at every boundary including a 29 February, rather than
    only when the run's labels happen to agree with a fixed model-calendar
    table (the residual mismatch :func:`monthly_mean`'s **Leap days** section
    describes for the ``365_day``/``360_day`` calendars, which do not have
    this luxury because they are not the calendar the labels are written in).

    Parameters
    ----------
    start_days, start_seconds : int
        Days and seconds since the Unix epoch of the run's start date (a
        ``jax_datetime.Datetime``'s ``.delta.days``/``.delta.seconds``).
    sequential : bool
        ``False`` for the twelve-bin climatology (bin = calendar month,
        0-indexed January first); ``True`` for the sequential form (bin =
        months since the run's first record, per ``y0``/``m0``).
    y0, m0 : int
        For the sequential form: the calendar year and month of the *first
        record's own midpoint* (not necessarily the run's start date -- see
        :func:`monthly_mean`'s sequential-form sizing, which computes both on
        the host with the same convention). Bin 0 is this month; ignored for
        the twelve-bin form.
    n_bins : int
        For the sequential form: the accumulator's size, which a run longer
        than wraps into modulo -- exactly as the fixed-calendar sequential
        form wraps at its own span (:func:`monthly_mean`'s **The sequential
        form wraps...** paragraph), except the wrap here is a wrap of
        *months* rather than of seconds, since there is no fixed span of
        seconds a whole number of Gregorian months corresponds to. Ignored
        for the twelve-bin form.

    Returns
    -------
    callable
        ``(record, record_seconds) -> int32 bin index``.

    """

    def bin_of_record(record: jnp.ndarray, record_seconds: int) -> jnp.ndarray:
        days, _ = gregorian_instant(
            record, record_seconds, start_days, start_seconds,
            offset_seconds=record_seconds // 2,
        )
        year, month, _ = gregorian_ymd_from_days(days)
        if sequential:
            index = (year - y0) * 12 + (month - m0)
            return jnp.mod(index, n_bins).astype(jnp.int32)
        return (month - 1).astype(jnp.int32)

    return bin_of_record


def _duration_to_seconds(duration: str | float, calendar: str, what: str) -> int:
    """Return ``duration`` as a whole positive number of seconds.

    The duration is parsed on the *coupler's own* calendar
    (:func:`jem.base.component.parse_duration_days` /
    :func:`jem.base.component.days_per_year`), which for ``"gregorian"`` is
    JEM's own fixed-average year, ``365.2425`` days -- so ``"1 year"`` is
    that many days exactly, not the number of days whatever real calendar
    year the run's own dates happen to fall in actually has. It must also be
    a whole number of seconds: everything downstream of here is integer
    arithmetic on seconds, because a float32 count of seconds since the
    start of a run stops being exact within a few decades of simulated time,
    and a fractional second is refused rather than rounded away.
    """
    days = float(parse_duration_days(duration, calendar))
    seconds = _exact_seconds(days * _SECONDS_PER_DAY, f"{what}={duration!r}")
    if seconds <= 0:
        raise ValueError(
            f"{what}={duration!r} is {seconds} s, which is not a positive "
            "duration."
        )
    return seconds


def _whole_coupling_steps(total_seconds: int, dt_seconds: int, total_time: str | float) -> int:
    """Return ``total_seconds`` as a whole, positive number of ``dt_seconds`` steps.

    ``monthly_mean(coupler, total_time=...)`` sizes a sequential accumulator
    to a run of this length, and that run -- were it actually integrated --
    would be refused by :func:`jem.driver.run_chunked`'s own
    ``total_time``/``chunk`` validation (``jem.driver._whole_steps``) unless
    it is a whole number of coupling steps: a run is only ever integrated in
    whole coupled steps, so a duration that is not one names no run for the
    accumulator to be sized to. Before this check existed, ``total_time=``
    was silently floor-divided by the coupling timestep on both calendar
    paths (this function's caller and :func:`_gregorian_monthly_mean`), so
    ``"36.5 hours"`` on a daily coupling built a one-bin accumulator as if it
    had been asked for exactly one day, and ``"10 years"`` on the default
    ``"gregorian"`` calendar -- whose duration parser uses the fixed-average
    365.2425-day year, so ``"10 years"`` is ``3652.425`` days, never a whole
    number of days -- silently discarded the leftover 0.425 of a day with no
    warning either.

    Unlike :func:`jem.driver._whole_steps`, which compares *days* as floats
    (a duration and a coupling timestep that need not themselves be whole
    seconds until parsed) and so needs a relative tolerance, this compares
    *seconds* that are already exact integers by the time either caller
    calls it (:func:`_duration_to_seconds` and ``_exact_seconds`` on the
    fixed-calendar path, ``_exact_seconds`` alone on the Gregorian one), so
    the check is an exact modulo -- no tolerance needed or wanted.

    Parameters
    ----------
    total_seconds : int
        ``total_time``, already converted to exact whole seconds.
    dt_seconds : int
        The coupling timestep, in exact whole seconds.
    total_time : str or float
        The original argument, for the error message only.

    Returns
    -------
    int
        ``total_seconds // dt_seconds``.

    Raises
    ------
    ValueError
        If ``total_seconds`` is not a whole, positive multiple of
        ``dt_seconds``.

    """
    n_steps, remainder = divmod(total_seconds, dt_seconds)
    if remainder or n_steps < 1:
        raise ValueError(
            f"total_time={total_time!r} is {total_seconds} s, which is "
            f"{total_seconds / dt_seconds:g} coupling steps of {dt_seconds} s "
            "-- not a whole number of them (or shorter than one). An "
            "accumulator is sized in whole coupled steps, exactly like a run "
            "itself (jem.driver.run_chunked's own total_time/chunk validates "
            "the same way), so there would be no well-defined record for a "
            "fractional step to hold."
        )
    return int(n_steps)


def _exact_seconds(value: float, what: str) -> int:
    """Return ``value`` as an ``int``, refusing a fractional second.

    Everything in this module is integer arithmetic on seconds, and the
    coupler only ever builds whole-second clocks (``Coupler._element_timestep``
    refuses a sub-timestep that is not), so a value that is not a whole
    number of seconds is a broken invariant or a duration the bins cannot
    represent. Either way it is refused rather than rounded: rounding would
    silently move every bin boundary.

    The comparison allows a float-rounding margin, because a duration given
    in days reaches here through a float multiplication (``11 / 86400`` days
    comes back as ``10.999999999999998`` s). A microsecond, plus a relative
    part for durations of centuries, is far below anything a clock or a
    duration can express and far above that rounding, so a genuine fraction
    of a second is still refused.
    """
    seconds = round(value)
    if not math.isclose(value, seconds, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            f"{what} is {value!r} s, which is not a whole number of seconds; "
            "the bins are laid out in whole seconds."
        )
    return int(seconds)


def _months_covering(
    rotated_months_seconds: np.ndarray, offset_seconds: int, total_seconds: int
) -> int:
    """Return how many months a run of ``total_seconds`` puts a record in.

    ``rotated_months_seconds`` are the month lengths starting with the month
    the run starts in, and ``offset_seconds`` is how far into that month the
    start date lies, so the run's labels run from ``offset + dt`` to
    ``offset + total_seconds``.

    The count is of the months those labels fall in, under the same
    closed-at-the-start convention the binning uses: a last label lying
    exactly on a month boundary belongs to the month it *opens*, which is why
    the boundary at that instant does not end the count. A one-month run from
    1 January therefore gets two bins, the second holding the single record
    at 00:00 on 1 February -- the same record ``groupby("time.month")`` of the
    output puts in February, whenever label and model calendar agree (see
    :func:`monthly_mean`'s **Leap days**), which is the whole reason the
    convention is what it is.
    """
    last_label = offset_seconds + total_seconds
    months, covered, cycle = 1, 0, len(rotated_months_seconds)
    # Terminates because every month is a positive number of seconds; the
    # count is bounded by the run's length in months, which is also the size
    # of the accumulator the caller is asking for.
    while covered + int(rotated_months_seconds[(months - 1) % cycle]) <= last_label:
        covered += int(rotated_months_seconds[(months - 1) % cycle])
        months += 1
    return months


def _record_axes(coupler: Any) -> dict[str, Any]:
    """Return the sub-step axes each component's diagnostics carry, by name.

    A component the workflow runs ``n > 1`` times per coupled step has its
    ``n`` diagnostics stacked on a leading axis of length ``n``, and a nested
    :class:`~jem.base.coupler.Coupler` that runs ``r`` of its own coupled
    steps per outer step stacks its inner components' diagnostics on a leading
    axis of length ``r`` -- so a step's diagnostics for such a component are
    not one record but several, each covering its own sub-interval.

    This is the map of that: ``{name: axes}``, where ``axes`` is the tuple of
    leading axis lengths one coupled step puts in front of the diagnostic's
    own shape (``()`` for a component that records once per coupled step),
    or -- for a nested coupler -- a ``{inner name: axes}`` mapping of its own
    components, since those may record at different rates from each other.
    The axes are in the order the leading axes appear, which is the order the
    records were produced in, so flattening them row-major gives the records
    of one coupled step in time order. That is the whole reason this map
    exists: :func:`_build_binned_mean` needs each record's own place in the coupled
    step to bin it by its own label rather than by the step's.

    The nested coupler is recognised by duck-typing rather than by an
    ``isinstance`` check, because importing :mod:`jem.base.coupler` here would
    pull in jax-gcm (and with it the whole atmosphere) just to import this
    module; :func:`_duration_to_seconds` defers its import for the same reason.
    """
    multiplicities = coupler.multiplicities()
    axes: dict[str, Any] = {}
    for name, component in coupler.components.items():
        runs = multiplicities.get(name, 0)
        if runs < 1:
            # Registered but not in the workflow: it never runs, so it
            # produces no diagnostics to bin.
            continue
        calls = (runs,) if runs > 1 else ()
        inner = _nested_record_axes(component)
        axes[name] = calls if inner is None else _prefixed(inner, calls)
    return axes


def _nested_record_axes(component: Any) -> Any:
    """Return a nested coupler's own record axes, or None if it is not one."""
    if not (hasattr(component, "components") and hasattr(component, "multiplicities")):
        return None
    # `outer_ratio` is None for a coupler that has not been bound -- which a
    # registered one always has been, so this is a guard rather than a case.
    ratio = getattr(component, "outer_ratio", None) or 1
    return _prefixed(_record_axes(component), (ratio,) if ratio > 1 else ())


def _prefixed(axes: Any, prefix: tuple[int, ...]) -> Any:
    """Return ``axes`` with ``prefix`` in front of every component's axes."""
    if isinstance(axes, dict):
        return {name: _prefixed(node, prefix) for name, node in axes.items()}
    return prefix + axes


def _records(axes: tuple[int, ...]) -> int:
    """Return how many records one coupled step produces on ``axes``."""
    return int(np.prod(axes, dtype=int))


def _record_counts(axes: Any) -> set[int]:
    """Return every distinct number of records per coupled step in ``axes``."""
    if isinstance(axes, dict):
        return set().union(*(_record_counts(node) for node in axes.values()), set())
    return {_records(axes)}


def _record_seconds(dt_seconds: int, records: int) -> int:
    """Return the interval one of ``records`` records per coupled step covers.

    The coupler refuses a workflow whose sub-timestep is not a whole number of
    seconds (``Coupler._element_timestep``), so this cannot fail for a model
    it built; it is checked again here because everything downstream is
    integer arithmetic on seconds, and a silent rounding would drift a bin
    boundary by a second per record.
    """
    seconds, remainder = divmod(dt_seconds, records)
    if remainder != 0 or seconds < 1:
        raise ValueError(
            f"A component recording {records} times per coupled step records "
            f"every {dt_seconds}/{records} s, which is not a whole number of "
            "seconds, so its records cannot be labelled exactly."
        )
    return seconds


def _zero_counts(axes: Any, n_bins: int) -> Any:
    """Return the zeroed counts of one component (or nested coupler)."""
    if isinstance(axes, dict):
        return {name: _zero_counts(node, n_bins) for name, node in axes.items()}
    return jnp.zeros((n_bins, *axes), dtype=jnp.int32)


def _all_flat(axes: Any) -> bool:
    """Return whether every component records exactly once per coupled step."""
    if isinstance(axes, dict):
        return all(_all_flat(node) for node in axes.values())
    return bool(axes == ())


def _divide_sums_by_counts(sums: Any, counts: Any) -> Any:
    """Return the running sums divided by the counts of their own bins.

    Recurses on ``counts`` rather than on ``sums``: the counts mirror the
    *components* of the model (one array per component when they record at
    different rates, one shared array when they do not), while a component's
    sums are an opaque pytree of its diagnostics, which may itself be a dict.
    """
    if isinstance(counts, dict):
        return {name: _divide_sums_by_counts(sums[name], node) for name, node in counts.items()}
    counts = jnp.asarray(counts)
    empty = counts == 0
    # The division is taken against a count of 1 where there is no data
    # and the result thrown away by the `where`. Dividing by zero and
    # masking the NaN afterwards would give the same value but a NaN
    # gradient, which propagates back through the whole trajectory.
    safe_counts = jnp.where(empty, 1, counts)

    def mean(total: jnp.ndarray) -> jnp.ndarray:
        total = jnp.asarray(total)
        # The counts are `(n_bins, *sub-step axes)` and the sum is those axes
        # followed by the diagnostic's own shape, so the trailing axes are
        # what has to be broadcast over.
        per_bin = counts.shape + (1,) * (total.ndim - counts.ndim)
        return jnp.where(
            empty.reshape(per_bin),
            jnp.nan,
            total / safe_counts.reshape(per_bin).astype(total.dtype),
        )

    return jax.tree_util.tree_map(mean, sums)


def fold_records(means: Any, counts: Any) -> Any:
    """Return one component's binned means with its sub-step axes folded away.

    :meth:`BinnedMean.finalize` **keeps** the axes a component that records
    more than once per coupled step accumulates into -- bin ``b``, slot ``j``
    is the mean of the records of call ``j`` that fell in ``b`` -- because
    that is a monthly-mean diurnal cycle, which folding destroys and which
    cannot be recovered afterwards. This is the fold, for when the plain mean
    over every record of a bin is what was wanted::

        monthly = monthly_mean(coupler)
        ...
        sums, counts = accumulator
        means = monthly.finalize(accumulator)
        per_month = fold_records(means["atm"], counts["atm"])

    It is a *weighted* mean, by each slot's own count, which is what makes it
    equal a ``groupby`` of the written output (on the bins' own terms -- for a
    monthly mean, under the leap-year condition in :func:`monthly_mean`'s
    **Leap days**): a straight mean over the slots is the same number only
    when every slot holds the same number of records, and what breaks that is
    exactly the bin boundary this binning exists to get right (23 of a day's
    hourly records in January, one in February).

    Parameters
    ----------
    means : pytree
        One component's entry in what :meth:`BinnedMean.finalize` returned:
        leaves shaped ``(n_bins, *sub-step axes, ...)``, empty bins NaN.
    counts : array
        That component's entry in the accumulator's counts, shaped
        ``(n_bins, *sub-step axes)``. A component that records once per
        coupled step has no sub-step axes, and folding it then returns its
        means unchanged -- exactly, not to within a float32 rounding -- so a
        caller need not know which kind of component it has.

    Returns
    -------
    pytree
        ``means`` with the sub-step axes summed away: leaves shaped
        ``(n_bins, ...)``, a bin no record fell in still NaN.

    """
    counts = jnp.asarray(counts)
    if counts.ndim == 1:
        # Nothing to fold: this component records once per coupled step, so a
        # bin already holds one mean over all of its records. Returning the
        # means untouched rather than multiplying by the count and dividing by
        # it again keeps this exactly the identity -- that round trip moves
        # the last bit of a float32 mean -- and keeps NaN for an empty bin.
        return means
    record_axes = tuple(range(1, counts.ndim))
    per_bin = jnp.sum(counts, axis=record_axes)

    def fold(leaf: jnp.ndarray) -> jnp.ndarray:
        leaf = jnp.asarray(leaf)
        # The counts are `(n_bins, *sub-step axes)` and a leaf is those axes
        # followed by the diagnostic's own shape, so the trailing axes are
        # what has to be broadcast over.
        weights = counts.reshape(counts.shape + (1,) * (leaf.ndim - counts.ndim))
        # An empty slot's mean is NaN, so its weight is applied by selection
        # rather than by multiplication: 0 * NaN is NaN, and one empty slot
        # would otherwise make the whole bin NaN (and its gradient NaN too).
        total = jnp.sum(jnp.where(weights == 0, 0.0, leaf) * weights, axis=record_axes)
        divisor = per_bin.reshape(per_bin.shape + (1,) * (total.ndim - per_bin.ndim))
        return jnp.where(
            divisor == 0,
            jnp.nan,
            total / jnp.where(divisor == 0, 1, divisor).astype(total.dtype),
        )

    return jax.tree_util.tree_map(fold, means)


class BinnedMean(NamedTuple):
    """The ``(init, update)`` pair for a binned mean, and its ``finalize``.

    It *is* the pair ``generate_trajectory_function(accumulate=...)`` takes --
    a two-field :class:`typing.NamedTuple`, so it both unpacks as
    ``init, update`` and carries the :meth:`finalize` that turns the
    accumulator into means::

        monthly = jem.accumulate.monthly_mean(coupler)
        trajectory = coupler.generate_trajectory_function(365, accumulate=monthly)
        carry, accumulator = trajectory(coupler.initialize())
        means = monthly.finalize(accumulator)

    Build one with :func:`monthly_mean` or :func:`windowed_mean`, which are
    what know the coupler's clock. The two differ only in how a step is
    mapped to a bin; what is accumulated, and how it is finalized, is this
    class either way.

    Attributes
    ----------
    init : callable
        ``() -> BinnedAccumulator``; the zeroed sums and counts.
    update : callable
        ``(accumulator, diagnostics, time) -> BinnedAccumulator``; adds one
        coupled step's diagnostics into their bins -- one bin for the step
        when every component records once per coupled step, and otherwise one
        per record (see :func:`_build_binned_mean`).

    """

    init: Callable[[], BinnedAccumulator]
    update: Callable[
        [BinnedAccumulator, dict[str, Diagnostics], CouplingTime],
        BinnedAccumulator,
    ]

    def finalize(self, accumulator: BinnedAccumulator) -> Any:
        """Return the binned means: the running sums divided by their counts.

        Parameters
        ----------
        accumulator : BinnedAccumulator
            What a trajectory built with this pair returned.

        Returns
        -------
        pytree
            The structure of one coupled step's diagnostics, every leaf with a
            leading axis of length ``n_bins`` -- for :func:`monthly_mean`,
            either the twelve calendar months, January first, or the
            ``n_months`` months the run passes through, its first month first;
            for :func:`windowed_mean`, the ``n_windows`` windows from the
            start of the run. A component
            that records ``n`` times per coupled step keeps that axis after
            the bins, so its leaves are ``(n_bins, n, ...)``: bin ``b``, slot
            ``j`` is the mean of the records of call ``j`` whose labels fell
            in bin ``b``. A bin **no record fell in is NaN**, not zero: a run
            of two months has ten empty monthly bins, and zero would be a
            value that plotted and averaged as if it were data.

        """
        sums, counts = accumulator
        # All of the arithmetic, and the bin count -- which is the
        # accumulator's own leading axis rather than anything captured when
        # the pair was built, so `finalize` is correct for whatever `init`
        # made and stays a plain function of its argument.
        return _divide_sums_by_counts(sums, counts)


def _build_binned_mean(
    coupler: Any,
    bin_of_record: Callable[[jnp.ndarray, int], jnp.ndarray],
    n_bins: int,
    carry: Any = None,
) -> BinnedMean:
    """Return the ``(init, update)`` pair that means each record into its bin.

    This is the whole of the reduction; :func:`monthly_mean` and
    :func:`windowed_mean` differ only in the ``bin_of_record`` they hand it,
    so a change to how sums are kept, to the dtype they are kept in, or to
    what an empty bin means, happens once.

    **A record, not a step.** A coupled step does not produce one record per
    component: a component the workflow runs ``n`` times per step produces
    ``n``, each covering its own sub-interval, so they need not all fall in
    the same bin. The 24 hourly records of the daily step that covers 31
    January have midpoints half an hour past each hour -- 23 of them (through
    ``22:00-23:00``) before midnight and one (``23:00-00:00``) after. Every
    record is therefore binned by its own interval -- its own midpoint, for
    :func:`monthly_mean`, which is the same instant
    :meth:`~jem.base.coupler.Coupler.to_xarray` labels that record with; its
    own end, for :func:`windowed_mean`, independent of that label (see that
    function's own docstring for why) -- read from the coupler's own sub-step
    clock (:meth:`~jem.base.coupler.Coupler.coupling_time_at_substep`).
    Binning them all by the coupled step instead would put a whole day of
    hourly records in the month the day *ended* in, and the accumulated mean
    would disagree with a ``groupby`` of the written output
    at every month boundary -- rather than only where the Gregorian labels and
    the model calendar themselves part company, which is the one residual
    disagreement and is :func:`monthly_mean`'s **Leap days**.

    The sub-step axis is **kept**, not folded: a component recording ``n``
    times per coupled step accumulates into ``(n_bins, n, ...)``, bin ``b``
    slot ``j`` holding the records of call ``j`` that fell in bin ``b`` -- a
    monthly-mean diurnal cycle for a component sub-cycling through the day.
    Folding it would be a mean of means, and it is one :func:`fold_records`
    away (weighted by the counts the accumulator carries) whereas the cycle
    cannot be recovered once folded.

    Because components recording at different rates fill different bins as one
    step is folded in, the counts are then **per component** rather than the
    one shared ``(n_bins,)`` array of a model whose components all record once
    per coupled step -- which keeps that (much more common) accumulator, and
    the reduction that fills it, exactly what it was.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for. The structure, shapes and
        dtypes of one step's diagnostics are taken from it, and so are the
        rates its components record at -- workflow multiplicity, and the inner
        steps of a nested coupler (see :func:`_record_axes`).
    bin_of_record : callable
        ``(record, record_seconds) -> int32 bin index in [0, n_bins)``,
        evaluated inside the scan on the traced index of a record counted from
        the start of the run in records of ``record_seconds`` each -- the
        coupled step counter and the coupling timestep for a component
        recording once per coupled step, the sub-step counter and the
        sub-timestep for one recording more often. It must be total: an index
        outside the range would be clipped by ``.at[].add`` and silently
        counted in the nearest bin, which is why :func:`_variable_window_rule`
        -- what both public builders use -- reduces the record counter modulo
        one period of the bins before looking a boundary up.
    n_bins : int
        Length of the accumulator's leading axis. Static, so that one
        compiled trajectory serves a run of any length.
    carry : CoupledCarry, optional
        A carry to take the diagnostics' shapes from. The shapes are obtained
        with ``jax.eval_shape`` of one coupled step, so the step is never run
        and nothing is computed; this argument only saves a driver that
        already has a carry the cost of ``coupler.initialize()``.

    Returns
    -------
    BinnedMean

    """
    if carry is None:
        carry = coupler.initialize()
    # `eval_shape` traces the coupled step abstractly: it gives the structure,
    # shapes and dtypes of a step's diagnostics without running the model, so
    # building the accumulator costs nothing and -- more importantly -- has no
    # side effects on components that have them.
    _, diagnostics_shapes = jax.eval_shape(coupler.generate_step_function(), carry)

    dt_seconds = _exact_seconds(coupler.dt_seconds, "the coupling timestep")
    axes = _record_axes(coupler)
    # True for a model whose every component records once per coupled step,
    # which is every model without workflow multiplicity or a sub-stepping
    # nested coupler. The reduction then has one bin index and one count array
    # for the whole step, as it did before rates could differ.
    flat = _all_flat(axes)
    # Validated here, at build time, rather than inside the traced scan body,
    # so that an impossible record rate is a construction error.
    record_seconds = {
        records: _record_seconds(dt_seconds, records)
        for records in _record_counts(axes)
    }

    def accumulator_dtype(dtype: Any) -> Any:
        """Return the dtype a running sum of ``dtype`` is kept in.

        A mean of integer or boolean diagnostics (a flag, a counter) is not an
        integer, so those accumulate in float32 rather than wrapping or
        truncating.
        """
        return dtype if jnp.issubdtype(dtype, jnp.floating) else jnp.float32

    def init() -> BinnedAccumulator:
        sums = jax.tree_util.tree_map(
            lambda leaf: jnp.zeros(
                (n_bins, *leaf.shape), accumulator_dtype(leaf.dtype)
            ),
            diagnostics_shapes,
        )
        if flat:
            return sums, jnp.zeros(n_bins, dtype=jnp.int32)
        return sums, {
            name: _zero_counts(axes.get(name, ()), n_bins)
            for name in diagnostics_shapes
        }

    def update(
        accumulator: BinnedAccumulator,
        diagnostics: dict[str, Diagnostics],
        time: CouplingTime,
    ) -> BinnedAccumulator:
        sums, counts = accumulator
        if flat:
            index = bin_of_record(time.step, dt_seconds)
            new_sums = jax.tree_util.tree_map(
                lambda total, value: total.at[index].add(value), sums, diagnostics
            )
            return new_sums, counts.at[index].add(1)

        # Every component recording at the same rate fills the same bins, so
        # the bins of a rate are built once per coupled step and shared.
        bins_by_rate: dict[int, jnp.ndarray] = {}

        def bins_of(records: int) -> jnp.ndarray:
            """Return the bin of each of one step's ``records`` records."""
            if records not in bins_by_rate:
                # The coupler's own sub-step clock, one call at a time, rather
                # than the same arithmetic written out again here: call `k` of
                # coupled step `s` is sub-step `s * records + k`, and
                # `bin_of_record` (below) is what turns that sub-step counter
                # into the bin of the record it produces.
                substeps = jnp.stack(
                    [
                        coupler.coupling_time_at_substep(time.step, call, records).step
                        for call in range(records)
                    ]
                )
                bins_by_rate[records] = bin_of_record(
                    substeps, record_seconds[records]
                )
            return bins_by_rate[records]

        def fold(
            axes_node: Any, sums_node: Any, counts_node: Any, diagnostics_node: Any
        ) -> tuple[Any, Any]:
            """Return one component's sums and counts with this step folded in."""
            if isinstance(axes_node, dict):
                # A nested coupler: its components record at their own rates,
                # so each is folded in on its own.
                folded = {
                    name: fold(
                        node,
                        sums_node[name],
                        counts_node[name],
                        diagnostics_node[name],
                    )
                    for name, node in axes_node.items()
                }
                return (
                    {name: node for name, (node, _) in folded.items()},
                    {name: node for name, (_, node) in folded.items()},
                )

            records = _records(axes_node)
            if records == 1:
                index = bin_of_record(time.step, dt_seconds)
                return (
                    jax.tree_util.tree_map(
                        lambda total, value: total.at[index].add(value),
                        sums_node,
                        diagnostics_node,
                    ),
                    counts_node.at[index].add(1),
                )

            record_bins = bins_of(records)
            slots = jnp.arange(records)
            depth = len(axes_node)

            def add(total: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
                """Scatter each of one step's records into its own bin."""
                total = jnp.asarray(total)
                value = jnp.asarray(value)
                # The sub-step axes are flattened to a single record axis, so
                # that one two-index scatter covers a component that runs `n`
                # times, a nested coupler that runs `r` inner steps, and a
                # component that does both. `.at[].add` accumulates repeated
                # indices, which is what the records of a step that stays
                # within one bin are.
                flattened = total.reshape(
                    (n_bins, records, *total.shape[1 + depth :])
                )
                updated = flattened.at[record_bins, slots].add(
                    value.reshape((records, *value.shape[depth:]))
                )
                return updated.reshape(total.shape)

            return (
                jax.tree_util.tree_map(add, sums_node, diagnostics_node),
                counts_node.reshape((n_bins, records))
                .at[record_bins, slots]
                .add(1)
                .reshape(counts_node.shape),
            )

        folded = {
            name: fold(
                axes.get(name, ()), sums[name], counts[name], component_diagnostics
            )
            for name, component_diagnostics in diagnostics.items()
        }
        return (
            {name: node for name, (node, _) in folded.items()},
            {name: node for name, (_, node) in folded.items()},
        )

    return BinnedMean(init=init, update=update)


def monthly_mean(
    coupler: Any,
    carry: Any = None,
    *,
    n_months: int | None = None,
    total_time: str | float | None = None,
) -> BinnedMean:
    """Build the in-scan accumulator of a coupled run's monthly means.

    Everything the reduction needs it takes from ``coupler``: the diagnostics
    a step produces (their structure, shapes and dtypes), the coupling
    timestep, the start date and the calendar. The caller supplies nothing but
    the coupler -- and, if the run's months are each to have a bin of their
    own, how many::

        monthly_mean(coupler)                           # (12, ...): a climatology
        monthly_mean(coupler, total_time="3650 days")   # (120, ...): every month
        monthly_mean(coupler, n_months=120)             # sized directly instead

    The two forms bin by the same rule and the same convention; they differ in
    what the accumulator *is*. Twelve bins are the calendar months, so a
    ten-year run composites its ten Januaries into bin 0 -- a climatology.
    ``n_months`` bins are the months the run passes through, in order, so the
    same run gives January of year 1 in bin 0 and December of year 10 in bin
    119, which is the monthly time series of the run -- and ``total_time``
    gives that same count (``120``, not ``121``) when it names exactly ten
    years: the run's *last* record's own midpoint, ``total_time - dt/2``, is
    still within December of year 10, not on the January-of-year-11 boundary
    a duration of exactly ``total_time`` would touch. (Before this migration,
    when a record's defining instant for both binning and sizing was its
    interval's *end* -- exactly ``total_time`` for a run of that length -- the
    same call needed a 121st bin to hold that single boundary record; the
    midpoint convention removes the boundary case entirely, not just relabels
    it, which is why the sizing arithmetic below no longer needs the
    corresponding "one more bin" adjustment either.)

    Every calendar a :class:`~jem.base.coupler.Coupler` supports is handled
    **exactly** and **in-scan**: ``"365_day"`` (the only fixed-length
    calendar a ``Coupler`` actually accepts -- ``"360_day"`` is a table
    :func:`month_lengths` can build, but not a calendar name any part of jem
    accepts; see that function's own docstring and
    :class:`~jem.base.coupler.Coupler`'s ``calendar`` parameter) bins against
    a fixed table of month lengths (:func:`month_lengths`), and
    ``"gregorian"`` -- since 2026-09, no longer a
    :class:`NotImplementedError` here -- bins against the real proleptic
    Gregorian calendar (real leap years, via
    :mod:`jem.base.calendar`), with no fixed table at all. This matters
    because a :class:`~jem.base.coupler.Coupler` bound to a real
    ``jcm.model.Model`` component (jax-gcm v3, PR 878) has no calendar of its
    own to choose and *must* be Gregorian
    (:meth:`~jem.components.jcm.component.JCMComponent.bind` enforces it),
    and is also the coupler's own default calendar (see
    :class:`~jem.base.coupler.Coupler`'s ``calendar`` parameter) -- so this is
    the common case, not a special one, and the two forms below apply to it
    exactly as they do to the fixed calendars.

    **Which month a record counts in.** A coupled step (or, for a component
    recording more than once per coupled step, one of its sub-steps -- see
    **Sub-steps** below) covers ``[start + k·dt, start + (k+1)·dt)``, and
    JEM's own output labels that record with the interval's **midpoint**,
    ``start + (k + 1/2)·dt`` (:class:`~jem.base.component.TimeAxis`, jax-gcm
    v3's own convention change -- see that class's docstring). The bin
    follows *that same instant*: a record is counted in the calendar month
    its own midpoint falls in, which is what makes
    ``monthly.finalize(accumulator)`` equal
    ``coupler.to_xarray(diagnostics).groupby("time.month").mean()`` of the
    same run **by construction**, leaf for leaf -- and, for the sequential
    form, the same grouped by year and month -- for *every* calendar, not
    only one whose labels happen to agree with a fixed model-calendar table.
    (For a component that records more than once per coupled step that
    equality holds after :func:`fold_records`, which folds the sub-step axis
    this reduction deliberately keeps; see **Sub-steps** below.)

    This is a **breaking change** from the binning this function used before
    the 2026-09 jax-gcm-878 migration review, which counted a record in the
    month its interval's *end* fell in (to match the *pre-878* end-of-interval
    output label). Once ``TimeAxis`` itself moved to midpoint labels, keeping
    the old end-of-interval bin rule would have made ``monthly_mean`` disagree
    with a ``groupby`` of its own coupler's written output at every month
    boundary -- exactly the disagreement this rebinding exists to prevent --
    so the bin rule moved to match, for **every** calendar
    (``"365_day"``/``"360_day"`` included, not only the newly-supported
    ``"gregorian"``). Bin *membership* near a month boundary therefore differs
    from a pre-migration run: a coupled step whose interval spans a boundary
    (only possible when the coupling step is not itself much shorter than a
    month, e.g. the daily case never spans one) is now counted by which half
    its midpoint falls in rather than always counting the step at its end.
    See the CHANGELOG's Breaking Changes for the same note in one line.

    A midpoint that lands on a fractional half-second (an odd-length record)
    is binned by the whole second it floors to, never the one after: every
    month boundary falls on a whole second, so flooring can only move a
    midpoint a half-second *away* from an upcoming boundary, never across one
    -- see :func:`_midpoint_month_rule` and
    :func:`jem.base.calendar.gregorian_instant` for exactly where this
    floor happens on each calendar.

    **Leap days on the fixed calendars.** ``"365_day"`` and ``"360_day"`` bin
    against their own fixed month-length table, which is not the calendar
    :meth:`~jem.base.component.TimeAxis.datetimes` labels with -- JEM writes
    every label as proleptic Gregorian whatever the model calendar is (JCM's
    convention, kept so a slab's output and the atmosphere's merge on one time
    axis; a known inconsistency, tracked as jax-gcm#449). On a ``"365_day"``
    run the two agree until the run's labels reach a Gregorian 29 February,
    after which each label sits one day *earlier* than the model-calendar date
    of the same instant -- one more day of drift per leap year the run
    passes -- so ``groupby("time.month")`` of the written output and this
    reduction's own bins part company by up to a few records near each
    affected month boundary, though both still hold the same *total* of
    records across the run. This residual mismatch is a property of the
    ``"365_day"``/``"360_day"`` calendars specifically -- their fixed table is
    not the calendar the labels are ever written in -- and does **not** arise
    on ``"gregorian"``, where the bins and the labels are now the same real
    calendar (see the paragraph above): the midpoint-vs-end rebinding closes
    the boundary-convention half of the old mismatch for every calendar, and
    using the real Gregorian calendar for ``"gregorian"`` runs closes the
    other (leap-day) half for the calendar essentially every atmosphere-coupled
    run now uses. Emitting calendar-consistent labels on the fixed calendars
    too -- ``cftime`` no-leap dates -- remains tracked as #118 and is
    unaffected by any of this.

    **The twelve-bin form wraps at the year**, because the bin is the calendar
    month and not the month since the run started: a three-year run's January
    bin holds all three Januaries, which is a climatology and is what the
    fixed ``(12, ...)`` accumulator is for.

    **The sequential form wraps at the span of its bins** -- the total length
    of the ``n_months`` months, not a number of months. A run that outlasts
    the accumulator is folded back modulo that span, which realigns with the
    calendar only when the span is a whole number of years, i.e. when
    ``n_months`` is a multiple of twelve. Otherwise a wrapped calendar month
    **straddles** two bins: with ``n_months=6`` from 1 January the bins span
    181 days, so the second August of the run puts 28 of its records in the
    February bin and 3 in the March bin. Wrapped bins of a sequential monthly
    mean are therefore only meaningful for a multiple of twelve; size the
    accumulator with ``total_time`` and it is never wrapped into at all, which
    is the form to prefer. Sizing it *larger* than the run is harmless -- the
    surplus bins stay NaN, like any bin no record fell in.

    (The wrap point is rounded up to the next whole coupled step, because the
    record counter is reduced modulo it and the span of a whole number of
    calendar months need not be a whole number of steps -- a 5-day coupling
    divides the 365-day year but not the 59 days of January and February. It
    moves the wrap by less than one coupling step and leaves every bin
    boundary exact, and since a wrapped bin is only calendar-aligned for a
    multiple of twelve and a ``total_time``-sized accumulator never wraps,
    there is nothing observable to pay for it.)

    The first and last bins of the sequential form are **partial** whenever
    the run's start (or end) does not itself sit at 00:00 on a calendar month
    boundary: a bin holds the records whose *midpoints* fall in it, and the
    run's first record's midpoint is ``start_date + dt/2``. A run starting
    exactly at 00:00 on 1 July with daily coupling therefore fills its July
    bin **completely** -- all 31 midpoints, 1 July 12:00 through 31 July
    12:00, fall in July, unlike the pre-migration end-labelled convention,
    under which the same run's last daily step of July (``[31 July, 1
    August)``) was labelled 1 August and counted in August, leaving July one
    record short (see the Breaking-change paragraph above). A run starting
    mid-month still gets a partial first bin -- only the midpoints from its
    own first record onward fall in that month -- and every bin is divided by
    its own count regardless, so a partial month is the mean of what fell in
    it.

    Both forms are the same twelve calendar-month lengths laid end to end and
    phased to the run's start date; only how many of them the accumulator
    holds, and therefore where it repeats, differs.

    **Sub-steps.** A component the workflow runs ``n > 1`` times per coupled
    step returns diagnostics with a leading sub-step axis of length ``n``, and
    that axis is kept: the mean for that component comes back as
    ``(12, n, ...)``, the monthly mean of each sub-step slot -- a
    monthly-mean diurnal cycle for a component sub-cycling through the day.
    Each of the ``n`` records is binned by **its own** midpoint, not by the
    coupled step's, so of the 24 hourly records the daily step covering 31
    January produces, the 23 whose midpoints (30 minutes past the hour) fall
    before midnight count in January and the one covering ``23:00-00:00``
    counts in February -- as ``groupby("time.month")`` of the written output
    does, exactly, on ``"gregorian"``, or up to the residual mismatch in
    **Leap days on the fixed calendars** above on ``"365_day"``/``"360_day"``.
    :func:`fold_records` folds that axis away,
    weighting each slot by its own count, when the plain monthly mean is what
    was wanted::

        sums, counts = accumulator                     # counts["atm"]: (12, n)
        means = monthly.finalize(accumulator)          # means["atm"]: (12, n, ...)
        per_month = fold_records(means["atm"], counts["atm"])   # (12, ...)

    -- a straight mean over the slot axis is the same number only when every
    slot holds the same number of records, which is what fails at the month
    boundaries this binning exists to get right.

    A nested :class:`~jem.base.coupler.Coupler` is treated the same way. Its
    diagnostics arrive as a mapping of *its* components, each with a leading
    axis of the inner coupled steps that ran within one outer step, and those
    inner records are labelled at the inner rate too -- so they are binned
    individually, at that rate, and the mapping is kept in the structure the
    step produced it in rather than flattened the way
    :meth:`~jem.base.coupler.Coupler.to_xarray` flattens it. The counts follow
    the same structure (one array per inner component), because inner
    components may sub-step at rates of their own.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for.
    carry : CoupledCarry, optional
        A carry to take the diagnostics' shapes from; see
        :func:`_build_binned_mean`.
    n_months : int, optional
        Build the **sequential** form with this many bins: the months the run
        passes through, in order, starting with the month the run's *first
        record's own midpoint* falls in (``start_date + dt/2`` -- see
        **Sequential-form bin 0** below for why this, rather than
        ``start_date`` itself, is what "the month the run starts in" means
        now). At most one of this and ``total_time``; neither gives the
        twelve-bin climatology.
    total_time : str or float, optional
        Build the sequential form sized to a run of this length -- a
        ``jem.base.component.parse_duration_days`` string (``"10 years"``,
        ``"400 days"``) or a number of days, parsed on the coupler's calendar.
        Must correspond to a whole number of coupling steps (as
        :func:`windowed_mean`'s own ``total_time`` does). On ``"365_day"``/
        ``"360_day"`` the bin count is the number of distinct calendar months
        the run's record midpoints touch, computed from the fixed month-length
        table; on ``"gregorian"`` it is computed on the **host**, with
        Python's own ``datetime`` (exact, and calendar arithmetic has no place
        in a bin-*count*, which is static): the calendar month of the run's
        first and last record's own midpoints, ``(last.year - first.year) *
        12 + (last.month - first.month) + 1``.

    **Sequential-form bin 0.** Bin 0 is defined as the calendar month of the
    run's *first record's own midpoint*, not of ``start_date`` itself, so
    that record always lands in bin 0 by construction even in the rare case
    where ``start_date`` sits within ``dt/2`` of a month's end and the first
    record's midpoint therefore falls in the *next* calendar month from
    ``start_date``'s own -- using ``start_date``'s month there would leave
    bin 0 permanently empty. The two agree for every ordinary case (any run
    whose coupling step is not itself comparable to a month in length), which
    is every shipped configuration.

    Returns
    -------
    BinnedMean
        Whose ``finalize`` returns ``(12, ...)`` leaves, January first, or
        ``(n_months, ...)`` leaves, the run's first month first.

    Raises
    ------
    ValueError
        If both ``n_months`` and ``total_time`` are given, if ``n_months`` is
        not a positive integer, if ``total_time`` is not positive or (on
        ``"gregorian"``) not a whole number of coupling steps, or -- on
        ``"365_day"``/``"360_day"`` only -- if the coupling timestep does not
        divide the year exactly, since the month of a record there is found
        from the step counter reduced modulo a whole number of steps per
        year. ``"gregorian"`` has no such restriction: a record's real
        calendar month is read directly off its exact date
        (:func:`_gregorian_month_rule`), which needs no period to reduce the
        step counter modulo at all.

    """
    if coupler.calendar == "gregorian":
        return _gregorian_monthly_mean(coupler, carry, n_months=n_months, total_time=total_time)

    # Whole seconds throughout: `jdt.Timedelta` is integer-backed and the year
    # offset is a difference of two dates, so this arithmetic is exact, which
    # float32 seconds-since-start would not be after a few decades of a run.
    month_seconds = (
        np.asarray(month_lengths(coupler), dtype=np.int64) * _SECONDS_PER_DAY
    )
    seconds_per_year = _exact_seconds(
        _SECONDS_PER_DAY * coupler.days_per_year, "the year"
    )
    dt_seconds = _exact_seconds(coupler.dt_seconds, "the coupling timestep")
    year_offset_seconds = _exact_seconds(
        coupler.year_offset_seconds, "the start date's offset into the year"
    )
    if dt_seconds <= 0 or seconds_per_year % dt_seconds:
        raise ValueError(
            f"A monthly mean needs the coupling timestep ({dt_seconds} s) to "
            f"divide the year ({seconds_per_year} s) exactly, so that the "
            "month of a step can be found from the step counter reduced modulo "
            "a whole number of steps per year. (This restriction is specific "
            f"to the {coupler.calendar!r} calendar's fixed month-length "
            "table; 'gregorian' has none.)"
        )
    if n_months is not None and total_time is not None:
        raise ValueError(
            "Give at most one of n_months and total_time: n_months sets the "
            "accumulator's size directly and total_time counts it from the "
            "run, while giving neither is the twelve-month climatology (got "
            f"n_months={n_months!r}, total_time={total_time!r})."
        )

    if n_months is None and total_time is None:
        # Twelve bins laid end to end, phased so that bin 0 is January
        # wherever in the year the run starts, and repeating with the year --
        # so a multi-year run composites its Januaries, which is what the
        # fixed `(12, ...)` accumulator is for. The month lengths sum to the
        # calendar's year by construction, so the period *is* the year.
        # `_midpoint_month_rule`, not `_variable_window_rule`: the bin has to
        # follow the record's MIDPOINT (see the docstring's Breaking-change
        # paragraph), which is a half-record shift that a record-count table
        # cannot express exactly -- see `_midpoint_month_rule`'s own
        # docstring for why that needs comparing in seconds instead.
        return _build_binned_mean(
            coupler,
            _midpoint_month_rule(np.cumsum(month_seconds), year_offset_seconds),
            MONTHS_PER_YEAR,
            carry,
        )

    # The sequential form. The table is rotated so that bin 0 is the month
    # the run's *first record's own midpoint* falls in (see the docstring's
    # **Sequential-form bin 0**) -- which, for the ordinary case of a
    # coupling step much shorter than a month, is the month the run starts
    # in, but is defined this way so that the first record always lands in
    # bin 0 even in the (rare) case where it does not. The phase
    # (`offset_seconds`) stays measured from the start date itself, not from
    # the first record's midpoint: `_midpoint_month_rule` adds the
    # per-record midpoint shift on its own, for every record, so adding it
    # again here would double it.
    month_starts = np.concatenate([[0], np.cumsum(month_seconds)[:-1]])
    first_record_midpoint = (year_offset_seconds + dt_seconds // 2) % seconds_per_year
    start_month = int(
        np.searchsorted(month_starts, first_record_midpoint, side="right") - 1
    )
    offset_seconds = year_offset_seconds - int(month_starts[start_month])
    rotated = np.concatenate(
        [month_seconds[start_month:], month_seconds[:start_month]]
    )

    if total_time is not None:
        total_seconds = _duration_to_seconds(total_time, coupler.calendar, "total_time")
        n_steps = _whole_coupling_steps(total_seconds, dt_seconds, total_time)
        # The elapsed time, from the start date, of the LAST record's own
        # midpoint -- `_months_covering` counts the calendar months from the
        # start date to this instant, inclusive, which is the accumulator
        # size the run needs. (`total_seconds` itself, the pre-878 argument,
        # was the elapsed time to the last record's END; using it here would
        # under- or over-count by a fraction of a step at the boundary,
        # exactly the mismatch the midpoint rebinding exists to remove.)
        last_record_midpoint = (n_steps - 1) * dt_seconds + dt_seconds // 2
        bins = _months_covering(rotated, offset_seconds, last_record_midpoint)
    elif isinstance(n_months, bool) or not isinstance(n_months, int) or n_months < 1:
        # `bool` is an `int`, and `n_months=True` would silently build a
        # one-month accumulator.
        raise ValueError(f"n_months must be a positive integer; got {n_months!r}.")
    else:
        bins = n_months

    boundaries = np.cumsum(
        [rotated[index % MONTHS_PER_YEAR] for index in range(bins)]
    )
    # The bins repeat where the last one ends, and the record counter is
    # reduced modulo that span, so it has to be a whole number of coupled
    # steps -- and the span of a whole number of calendar months need not be
    # one (a 5-day coupling divides the 365-day year but not the 59 days of
    # January and February). The LAST bin is therefore extended to the next
    # coupled step, by less than one step; every other boundary stays exact.
    # Nothing observable pays for it: an accumulator sized by `total_time` is
    # never wrapped into at all, and a wrapped `n_months` bin only lines up
    # with a calendar month when `n_months` is a multiple of twelve anyway
    # (see the wrap paragraph in the docstring). Refusing instead would reject
    # every `total_time` form on such a coupling, since counting months from a
    # run always gives 12N+1 of them, whose span is never a whole number of
    # years.
    boundaries[-1] = _ceil_div(int(boundaries[-1]), dt_seconds) * dt_seconds

    return _build_binned_mean(
        coupler,
        _midpoint_month_rule(boundaries, offset_seconds),
        bins,
        carry,
    )


def _gregorian_monthly_mean(
    coupler: Any,
    carry: Any,
    *,
    n_months: int | None,
    total_time: str | float | None,
) -> BinnedMean:
    """Return :func:`monthly_mean` on the ``"gregorian"`` calendar.

    Split out of :func:`monthly_mean` itself only to keep the two calendar
    families' construction code apart -- this is the whole of the
    ``"gregorian"`` path: no fixed month-length table, no divide-the-year
    check, and the sequential form's bin count is sized on the **host**, with
    Python's own ``datetime`` arithmetic, rather than in the seconds-only
    arithmetic the fixed calendars use (there being no fixed span of seconds
    a whole number of Gregorian months corresponds to in the first place).
    See :func:`_gregorian_month_rule` for the in-scan bin rule itself, and
    :func:`monthly_mean`'s own docstring for everything about what the two
    forms mean.
    """
    if n_months is not None and total_time is not None:
        raise ValueError(
            "Give at most one of n_months and total_time: n_months sets the "
            "accumulator's size directly and total_time counts it from the "
            "run, while giving neither is the twelve-month climatology (got "
            f"n_months={n_months!r}, total_time={total_time!r})."
        )

    dt_seconds = _exact_seconds(coupler.dt_seconds, "the coupling timestep")
    start_days = int(np.asarray(coupler.start_date.delta.days))
    start_seconds = int(np.asarray(coupler.start_date.delta.seconds))

    if n_months is None and total_time is None:
        return _build_binned_mean(
            coupler,
            _gregorian_month_rule(start_days, start_seconds, sequential=False),
            MONTHS_PER_YEAR,
            carry,
        )

    # The sequential form's bin 0 is the calendar month of the run's first
    # record's own midpoint (`start_date + dt/2`) -- computed here, on the
    # host, with Python's `datetime`, which does exact proleptic-Gregorian
    # arithmetic for free and needs no `gcd`/period trick at all, since
    # nothing here is traced. See `monthly_mean`'s **Sequential-form bin 0**.
    start = coupler.start_date.to_pydatetime()
    step = datetime.timedelta(seconds=dt_seconds)
    first_midpoint = start + step / 2
    y0, m0 = first_midpoint.year, first_midpoint.month

    if total_time is not None:
        total_seconds = _duration_to_seconds(total_time, coupler.calendar, "total_time")
        n_steps = _whole_coupling_steps(total_seconds, dt_seconds, total_time)
        # `_gregorian_month_rule.bin_of_record` evaluates `gregorian_instant`
        # at every record's own midpoint (`offset_seconds=dt_seconds // 2`)
        # for every record `0` through `n_steps - 1`; this is exactly the
        # same "check the maximum record a construction-time-known pattern
        # will ever ask for" this module already does for the fixed
        # calendars' own `_midpoint_month_rule` -- see that check's own
        # comment -- extended here to the calendar whose sequential form
        # this branch builds (2026-09 migration review, round 2, finding
        # B1's `run_chunked` refusal, mirrored here since `monthly_mean` is
        # its own construction site with its own known record count, not
        # something `run_chunked`'s check can see).
        last_record = n_steps - 1
        bound = max_safe_record(
            dt_seconds, offset_seconds=dt_seconds // 2,
            start_seconds=start_seconds, start_days=start_days,
        )
        if last_record > bound:
            raise ValueError(
                f"total_time={total_time!r} needs {n_steps} {dt_seconds} s "
                f"records, but gregorian_instant can only resolve up to "
                f"{bound + 1} of them (record {bound}) exactly for a "
                "coupling timestep and start date this long (see "
                "jem.base.calendar.max_safe_record) -- this run is too long "
                "to bin exactly."
            )
        last_midpoint = start + step * (n_steps - 1) + step / 2
        n_bins = (last_midpoint.year - y0) * 12 + (last_midpoint.month - m0) + 1
    elif isinstance(n_months, bool) or not isinstance(n_months, int) or n_months < 1:
        raise ValueError(f"n_months must be a positive integer; got {n_months!r}.")
    else:
        n_bins = n_months

    return _build_binned_mean(
        coupler,
        _gregorian_month_rule(
            start_days, start_seconds, sequential=True, y0=y0, m0=m0, n_bins=n_bins
        ),
        n_bins,
        carry,
    )


def windowed_mean(
    coupler: Any,
    window: str | float | Sequence[str | float],
    *,
    n_windows: int | None = None,
    total_time: str | float | None = None,
    carry: Any = None,
) -> BinnedMean:
    """Build the in-scan accumulator of a run's means over successive windows.

    The reduction a sub-seasonal forecast is scored on: the mean over each
    pentad, or each week, of a run, rather than over each calendar month::

        pentads = windowed_mean(coupler, "5 days", n_windows=73)   # a year
        weeks = windowed_mean(coupler, "7 days", total_time="1 year")

    The windows need not all be the same length. Give a **sequence** of
    lengths and it is a pattern the windows cycle through, repeating for as
    long as the accumulator is -- a forecast scored on daily leads for its
    first week and pentads thereafter, a run alternating a spin-up window with
    a sampling one::

        leads = windowed_mean(coupler, [1, 1, 1, 1, 1, 1, 1, 5, 5],
                              total_time="30 days")

    **Every window, whatever its length, is measured from the run's start
    date**, with no phase and no reference to the calendar. A pattern of
    calendar month lengths is therefore calendar months only for a run
    starting at 00:00 on 1 January; for a run starting on 1 July,
    :func:`month_lengths` (which is always January-first) would bin its first
    31 days together, then 28, and so on. Use
    ``monthly_mean(coupler, total_time=...)`` for one bin per calendar month
    of a run -- it rotates the month table to the month the run starts in and
    phases it to the start date, which is a thing the calendar knows and a
    window does not. No ``offset=`` knob is offered here for the same reason
    ``inclusive=`` is not: a window's meaning is "so many days into the run", and
    a bin that means something else belongs to the builder that knows what.

    **Which window a record counts in.** A record is binned against the
    **end** of the interval it covers, measured in elapsed run-time from the
    start date -- window ``w`` is the records whose intervals end in
    ``(start of w, end of w]``, so with daily coupling the first 5-day window
    is the records covering day 0 to day 5, which is what a forecast means by
    "the first pentad". This is **not** the same instant
    :class:`~jem.base.component.TimeAxis` labels that record with any more --
    jax-gcm v3 (PR 878) moved every written record's label to its interval's
    **midpoint**, not its end (see that class's docstring) -- and the two are
    deliberately kept independent here: a window's meaning is "so many days
    into the run", a fact about elapsed time that has nothing to do with what
    instant a plotting library happens to stamp on the record, so this
    function keeps binning against the interval end regardless of what the
    output label says. (:func:`monthly_mean`, by contrast, *is* required to
    agree with its own written output's ``groupby("time.month")``, which is
    why *its* bin rule moved to the midpoint along with the label -- see that
    function's docstring for why the two builders now differ here on purpose.)
    In terms of the counter of records of length ``r`` that is ``record //
    (window/r)`` for equal windows, since a record's interval-end position is
    ``(k+1)·r`` for record ``k``.

    A shared boundary places a record the same way in both builders whenever
    the boundary itself is grid-aligned (a whole number of coupling steps from
    the start, which both builders require of their own boundaries anyway):
    the record ending exactly on the boundary has its midpoint half a record
    *before* it, and the next record's midpoint sits half a record *after* --
    so "ends at or before the boundary" (this function's rule) and "has its
    midpoint before the boundary" (:func:`monthly_mean`'s) agree on every
    record, on both sides. A month-long window pattern
    (``windowed_mean(coupler, month_lengths(coupler), ...)``) from a run
    starting at 00:00 on 1 January therefore now lands every record in the
    same bin index a same-length :func:`monthly_mean` would -- which was
    *not* true before this migration, when both builders bound by the
    interval's end but closed a shared boundary in opposite directions (this
    function keeps records up to and including a boundary in the window
    before it; :func:`monthly_mean` used to put the record exactly on a
    boundary in the month after it). No ``inclusive=`` knob is offered to
    split the difference: the convention is not a preference but what makes
    each builder agree with the thing it is meant to agree with, and a run
    whose bins closed one way while its output was grouped the other would
    silently disagree with itself.

    **A run longer than the accumulator wraps**, exactly as
    :func:`monthly_mean` wraps at a year (or at ``n_months``): window ``w``
    then also collects
    windows ``w + n_windows``, ``w + 2·n_windows`` and so on, giving the
    composite of every *w*-th window of the run. That is the price of an
    accumulator whose size is fixed at trace time and does not grow with the
    run -- the whole point of reducing inside the scan. Size the accumulator
    to the run (pass ``total_time``, or ``n_windows`` counted for the run) if
    each window is meant to stand on its own; with a *pattern* of window
    lengths the wrap is at the sum of all ``n_windows`` lengths rather than at
    the end of one cycle of the pattern.

    Sub-steps and nested couplers are accumulated exactly as
    :func:`monthly_mean` describes: a component recording ``n`` times per
    coupled step keeps that axis (``(n_windows, n, ...)``) and each of its
    records is binned by its own label, so a window boundary falls between two
    sub-steps of a coupled step wherever the labels say it does.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for.
    window : str or float or sequence of str or float
        Length of one window, as a ``jem.base.component.parse_duration_days`` string
        (``"5 days"``, ``"2 days"``, ``"12 hours"``) or a number of days,
        parsed on the coupler's calendar; or a **sequence** of such lengths,
        which is a pattern the windows cycle through (``["10 days",
        "20 days"]``). Note that ``"1 month"`` is a *calendar-averaged* month
        -- 365/12 days, not a whole number of daily steps -- and that no
        window is a calendar month, whatever its length: calendar months come
        from :func:`monthly_mean`, which knows where in the calendar the run
        starts. Every length must be a whole number of coupling steps: a
        window that ended part-way through a step would have to attribute that
        step to one side or the other, and there is no defensible choice.
    n_windows : int, optional
        How many windows the accumulator holds -- the length of the leading
        axis of everything ``finalize`` returns, and, when it exceeds the
        length of a pattern, how far the pattern is repeated. For a single
        window length, exactly one of this and ``total_time`` must be given;
        for a pattern, giving neither means one cycle of it
        (``n_windows = len(window)``).
    total_time : str or float, optional
        The length of the run, in the same forms as a single ``window``, from
        which ``n_windows`` is counted: enough windows to cover the run, the
        last one short if the run does not end on a window boundary (its mean
        is then over the steps that did fall in it, because every bin is
        divided by its own count).
    carry : CoupledCarry, optional
        A carry to take the diagnostics' shapes from; see
        :func:`_build_binned_mean`.

    Returns
    -------
    BinnedMean
        Whose ``finalize`` returns ``(n_windows, ...)`` leaves, the first
        window of the run first.

    Raises
    ------
    ValueError
        If ``window`` is an empty sequence, if both ``n_windows`` and
        ``total_time`` are given (or neither, for a single window length), if
        ``n_windows`` is not a positive integer, if any duration is not
        positive, or if any window length is not a whole number of coupling
        steps.

    """
    dt_seconds = _exact_seconds(coupler.dt_seconds, "the coupling timestep")
    if dt_seconds <= 0:
        raise ValueError(
            f"The coupling timestep is {dt_seconds} s; a windowed mean needs a "
            "positive one to count steps per window."
        )

    # A string is iterable, and one is a single window rather than a pattern
    # of one-character ones; everything else iterable is a pattern.
    pattern = not isinstance(window, str) and isinstance(window, Iterable)
    entries: list[Any] = list(window) if pattern else [window]  # type: ignore[arg-type]
    if not entries:
        raise ValueError(
            "window=[] has no windows for the accumulator to cycle through; "
            "give at least one length."
        )
    lengths_seconds = []
    for position, entry in enumerate(entries):
        what = f"window[{position}]" if pattern else "window"
        length_seconds = _duration_to_seconds(entry, coupler.calendar, what)
        if length_seconds % dt_seconds:
            raise ValueError(
                f"{what}={entry!r} is {length_seconds} s, which is not a whole "
                f"number of coupling steps of {dt_seconds} s. A window that "
                "ended part-way through a coupled step could only be filled by "
                "splitting that step between two windows, which the reduction "
                "does not do."
            )
        lengths_seconds.append(length_seconds)

    if n_windows is not None and total_time is not None:
        raise ValueError(
            "Give exactly one of n_windows and total_time: n_windows sets the "
            "accumulator's size directly, total_time counts it from the run "
            f"(got n_windows={n_windows!r}, total_time={total_time!r})."
        )
    if total_time is not None:
        total_seconds = _duration_to_seconds(total_time, coupler.calendar, "total_time")
        # Round *up*: a run that is not a whole number of windows ends inside
        # one, and that window has to exist to hold it. It is divided by its
        # own count like every other, so a short final window is the mean of
        # what fell in it rather than a mean diluted by missing steps.
        bins, covered = 0, 0
        while covered < total_seconds:
            covered += lengths_seconds[bins % len(lengths_seconds)]
            bins += 1
    elif n_windows is None:
        if not pattern:
            raise ValueError(
                "Give exactly one of n_windows and total_time: n_windows sets "
                "the accumulator's size directly, total_time counts it from "
                "the run (got n_windows=None, total_time=None). Only a "
                "sequence of window lengths may be given neither, and then it "
                "is used once through."
            )
        # One cycle of the pattern is the only size a pattern implies on its
        # own, and it is the useful one: a year of calendar months, a
        # fortnight of alternating windows.
        bins = len(lengths_seconds)
    elif isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 1:
        # `bool` is an `int`, and `n_windows=True` would silently build a
        # one-window accumulator -- i.e. a mean of the whole run.
        raise ValueError(f"n_windows must be a positive integer; got {n_windows!r}.")
    else:
        bins = n_windows

    # The window boundaries: the cumulative sum of the `bins` lengths, the
    # pattern cycling for as long as the accumulator is. The last boundary is
    # the period the whole accumulator repeats with, so a run longer than it
    # wraps -- and a pattern sized to the run (the calendar-month case) does
    # not wrap at all, which is the point of being able to size it.
    boundaries = np.cumsum(
        [lengths_seconds[index % len(lengths_seconds)] for index in range(bins)],
        dtype=np.int64,
    )
    # Offset 0: the windows are measured from the run's own start date, not
    # from anything in the calendar. Closed at the end: a window is an
    # interval, and the record whose OWN interval ends exactly on a boundary
    # (in elapsed run-time -- see this function's own docstring for why this
    # is independent of the record's written label) is the last of the
    # window it closes.
    window_index = _variable_window_rule(boundaries, 0, "right")

    return _build_binned_mean(coupler, window_index, bins, carry)
