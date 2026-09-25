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
``tests/unit/test_calendar.py`` cross-checks all three against
``jcm.date`` and against ``pandas`` over 400+ years, century leap-year rules
included, so a divergence between the two copies is a test failure rather
than a silent drift.

:func:`gregorian_instant` is the one piece of arithmetic new to this module:
an int32-safe "reduce-before-multiply" decomposition (see its own docstring
for exactly which one, and :func:`max_safe_record` for the exact range it is
exact over) that turns a record counter that can run into the billions over a
long or fine-grained run into the (days, seconds)-since-epoch of one instant
within that record, without ever forming a product that could overflow
``int32``. Three callers use it -- :func:`jem.accumulate._gregorian_month_rule`
and :func:`jem.accumulate._midpoint_month_rule` at the record's **midpoint**
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

# The largest magnitude a JAX int32 value can hold. `gregorian_instant` and
# `max_safe_record` size their block decomposition against this, not against
# `jnp.iinfo(jnp.int32).max`, so that the bound is visible as a plain number
# in both docstrings without importing NumPy just for it.
_INT32_MAX = 2**31 - 1

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
    finer coupling). The fix, in the spirit of jax-gcm's own v3 clock
    (``jcm.date``) but a genuinely different decomposition from the one this
    function shipped with through 2026-09 (see **History** below): reduce the
    **traced** record counter modulo a small, **static** ``block_size`` --
    chosen, from ``record_seconds`` alone, to be the largest block whose own
    total length in seconds still fits an ``int32`` -- *before* multiplying,
    and let the (per-block) day/second decomposition of that block's own
    length, plus a day count that only grows with elapsed *days* rather than
    with the record count, carry the rest. Concretely: ``block_size =
    (2**31 - 1 - |offset_seconds| - |start_seconds|) // record_seconds``
    (at least 1), so ``block_size * record_seconds`` -- an exact, unbounded
    Python integer, since ``record_seconds`` is static -- never has to be
    multiplied by anything traced; ``record = block_index * block_size +
    within_block`` puts ``within_block * record_seconds`` (bounded by
    ``block_size * record_seconds``, int32-safe by construction) on one side
    and ``block_index`` -- which grows only as fast as ``record /
    block_size``, i.e. only as fast as *elapsed blocks*, not as fast as
    ``record`` itself -- on the other, multiplied only by the block's own
    (necessarily small, ``< SECONDS_PER_DAY``) leftover-seconds and
    (necessarily small relative to ``2**31``) day count.

    **Representable range.** This is exact up to :func:`max_safe_record`
    (see that function for the exact, tested bound and its derivation) --
    which reaches the *full* (or, for a ``record_seconds`` that is itself an
    exact multiple of a day, within about a day's worth of) ``int32`` record
    range for any ``record_seconds`` that divides or is divided by a day
    exactly, since then the block above has no leftover seconds at all and
    the only limit left is the day count itself (representable to about 5.87
    million years). For a ``record_seconds`` that shares little structure
    with a day -- a calendar month, say -- the safe range is smaller, but
    still enormously larger than the ``D <= 24855`` (about 68 years' worth of
    *records*, not of run length) this function's previous decomposition
    silently broke beyond: a "1 month" (2629746 s) Gregorian coupling step is
    now exact for 82415 records -- nearly 6,900 years of monthly output --
    rather than breaking after 817 of them (2026-09 migration review, item 2
    -- see :func:`max_safe_record`'s own docstring for the derivation, and
    ``tests/unit/test_calendar.py``'s property test for the cross-check
    against Python's exact arithmetic that caught the previous version being
    wrong).

    **History.** The version of this function shipped before the above fix
    reduced ``record`` modulo the *minimal* period that lines up with whole
    days (``period_records = SECONDS_PER_DAY // gcd(record_seconds,
    SECONDS_PER_DAY)``), which is a genuinely different quantity from
    ``block_size`` above and can be far larger: for the "1 month" step,
    ``gcd(2629746, 86400) == 54``, giving a period of 1600 records whose
    *own* total length (2145872736 s) already exceeds ``2**31`` -- so a
    record as small as 817 already overflowed. That decomposition's
    docstring claimed it was int32-safe "for any coupling timestep up to
    tens of years", which was never true (the bound is on
    ``record_seconds / gcd(record_seconds, SECONDS_PER_DAY)``, not on
    ``record_seconds`` itself) and is corrected here rather than repeated.

    Parameters
    ----------
    record : jax.Array
        int32 (or castable) record counter, traced.
    record_seconds : int
        Length of one record, in seconds. Static (a Python int, not traced).
        Must be positive.
    start_days, start_seconds : int
        Days and seconds since the Unix epoch of record 0's start. Static.
    offset_seconds : int, optional
        Seconds into record ``record`` at which to evaluate the instant.
        Static; both callers pass ``0`` or ``record_seconds // 2``, but
        nothing here requires ``offset_seconds <= record_seconds`` --
        :func:`jem.accumulate._midpoint_month_rule` also adds a pattern
        phase that is typically much larger than one record.

    Returns
    -------
    days, seconds : jax.Array
        int32 arrays: whole days since the Unix epoch, and the seconds within
        that day, of the requested instant, exact for any ``record`` up to
        ``max_safe_record(record_seconds, offset_seconds=...,
        start_seconds=...)`` (:func:`max_safe_record`).

    Raises
    ------
    ValueError
        If ``record_seconds`` is not positive, or if ``offset_seconds`` and
        ``start_seconds`` alone (before ``record`` even enters) already leave
        no int32-safe room for a single record -- see
        :func:`_gregorian_instant_block`.

    """
    block_size, block_days, block_extra_seconds = _gregorian_instant_block(
        record_seconds, offset_seconds=offset_seconds, start_seconds=start_seconds
    )
    record = jnp.asarray(record, dtype=jnp.int32)
    block_index, within_block = jnp.divmod(record, block_size)
    # `within_block * record_seconds` is bounded by `block_size *
    # record_seconds`, int32-safe by `_gregorian_instant_block`'s own
    # construction of `block_size`; `block_index * block_extra_seconds` is
    # what eventually overflows for a large enough `record` (bounded in
    # `max_safe_record`, not here) -- `block_extra_seconds < SECONDS_PER_DAY`
    # keeps it small for as long as `block_index` itself stays small.
    seconds_within_block = (
        within_block * record_seconds
        + block_index * block_extra_seconds
        + offset_seconds + start_seconds
    )
    extra_days, seconds = jnp.divmod(seconds_within_block, SECONDS_PER_DAY)
    # `block_index * block_days`: a plain int32 multiply of two quantities
    # that both grow only with elapsed *days* (not with the record count
    # directly), which is what stays representable for a run of any
    # realistic length -- see `max_safe_record`.
    days = start_days + block_index * block_days + extra_days
    return days, seconds


def _gregorian_instant_block(
    record_seconds: int, *, offset_seconds: int = 0, start_seconds: int = 0
) -> tuple[int, int, int]:
    """Return ``(block_size, block_days, block_extra_seconds)`` -- shared by
    :func:`gregorian_instant` and :func:`max_safe_record`, so the two can
    never disagree about what block a given ``record_seconds`` decomposes
    into.

    ``block_size`` is the largest number of records of ``record_seconds``
    whose *own* total length still leaves an exact, unbounded Python integer
    (``block_size * record_seconds``) safely under ``2**31 - 1`` once
    ``offset_seconds`` and ``start_seconds`` -- the other two additive terms
    :func:`gregorian_instant` ever multiplies nothing by, but does add --
    are accounted for, **rounded down to the nearest multiple of
    ``SECONDS_PER_DAY // gcd(record_seconds, SECONDS_PER_DAY)``** (the
    smallest number of records that lines up with a whole number of days --
    the same quantity this module's superseded decomposition used directly
    as its one, possibly-too-large, block; see :func:`gregorian_instant`'s
    **History** note) whenever that fits at least once. That rounding is
    what makes ``block_extra_seconds`` exactly ``0`` -- and so
    :func:`max_safe_record` unbounded but for the day-count limit itself --
    for *any* ``record_seconds`` that shares a day-aligning factor with
    ``SECONDS_PER_DAY`` small enough for at least one such minimal period to
    fit the budget (every sub-daily divisor of a day, and every whole-day
    multiple, in practice): rounding to the nearest such multiple, rather
    than using the plain floor of ``budget // record_seconds``, does not
    shrink ``block_size`` by more than one minimal period's worth of records
    -- negligible next to the budget -- but turns a would-be leftover
    fraction of a day into exactly none.  When even one minimal period does
    not fit the budget (an extreme ``record_seconds``, comparable to
    ``2**31`` seconds itself), this falls back to the plain, unaligned
    ``block_size``, which is still int32-safe by construction, just no
    longer exactly day-aligned.  ``block_days``/``block_extra_seconds`` are
    the resulting block's own exact whole-day and leftover-second lengths
    (both computed in Python, which has no overflow, so this never
    approximates).

    Parameters
    ----------
    record_seconds : int
        Length of one record, in seconds. Must be positive.
    offset_seconds, start_seconds : int, optional
        The same arguments :func:`gregorian_instant` takes; only their
        magnitude matters here, since both are added (never multiplied by
        anything traced) at every call.

    Returns
    -------
    block_size, block_days, block_extra_seconds : int

    Raises
    ------
    ValueError
        If ``record_seconds`` is not positive, or if ``offset_seconds`` and
        ``start_seconds`` alone already leave no room for even one record
        (a pathological combination no caller in this codebase constructs,
        but a clear error here is cheaper than a wrong one downstream).

    """
    record_seconds = int(record_seconds)
    if record_seconds <= 0:
        raise ValueError(
            f"record_seconds must be a positive whole number of seconds; "
            f"got {record_seconds!r}."
        )
    budget = _INT32_MAX - abs(int(offset_seconds)) - abs(int(start_seconds))
    if budget < record_seconds:
        raise ValueError(
            f"offset_seconds={offset_seconds!r} and start_seconds="
            f"{start_seconds!r} alone leave no int32-safe room for even one "
            f"{record_seconds} s record; this instant cannot be computed "
            "exactly."
        )
    unaligned_block_size = max(1, budget // record_seconds)
    # The smallest number of records that lines up with a whole number of
    # days -- exactly `period_records` in this module's superseded
    # decomposition (`gregorian_instant`'s **History** note). Rounding
    # `unaligned_block_size` down to a multiple of it costs at most one such
    # period's worth of records (negligible next to the budget) but makes the
    # block's own length an EXACT multiple of a day whenever at least one
    # period fits, which is what lets `max_safe_record` be unbounded but for
    # the day-count limit itself for the overwhelming majority of coupling
    # steps in practice (anything that shares a reasonably small
    # day-aligning factor with `SECONDS_PER_DAY`).
    minimal_period = SECONDS_PER_DAY // math.gcd(record_seconds, SECONDS_PER_DAY)
    aligned_periods = unaligned_block_size // minimal_period
    block_size = (
        aligned_periods * minimal_period if aligned_periods >= 1
        else unaligned_block_size
    )
    block_seconds = block_size * record_seconds  # exact Python int, <= budget
    block_days, block_extra_seconds = divmod(block_seconds, SECONDS_PER_DAY)
    return block_size, block_days, block_extra_seconds


def max_safe_record(
    record_seconds: int, *, offset_seconds: int = 0, start_seconds: int = 0
) -> int:
    """Return the largest ``record`` for which :func:`gregorian_instant` is exact.

    A caller that knows its own maximum record count ahead of time -- the
    only place in this codebase that does is
    :func:`jem.accumulate.monthly_mean`'s sequential form, which is handed
    ``n_months`` or ``total_time`` up front -- should check it against this
    bound **at construction**, before anything is traced, and raise rather
    than silently accumulate into bins :func:`gregorian_instant` can no
    longer place correctly. A caller with no such bound (``CouplingTime``'s
    ``year_fraction``, called once per step of a run whose length is not
    fixed in advance) cannot check this per call -- ``record`` is traced --
    and instead relies on the bound below being enormous for any coupling
    step of a realistic length.

    **The bound, derived exactly.** Write ``budget = 2**31 - 1 -
    |offset_seconds| - |start_seconds|``, ``block_size = budget //
    record_seconds`` and ``slack = budget - block_size * record_seconds``
    (the remainder that floor division leaves on the table) -- the same
    three quantities :func:`gregorian_instant` computes via
    :func:`_gregorian_instant_block`. Within one block,
    ``within_block * record_seconds <= (block_size - 1) * record_seconds =
    block_size * record_seconds - record_seconds``, so the one term that
    grows without bound as ``record`` does -- ``block_index *
    block_extra_seconds`` -- has, after accounting for ``slack``, exactly
    ``slack + record_seconds`` of int32 headroom left to spend before the
    sum could exceed ``2**31 - 1``. So the largest safe ``block_index`` is
    ``(slack + record_seconds) // block_extra_seconds`` (unbounded by this
    term when ``block_extra_seconds == 0``, i.e. when ``record_seconds``
    divides -- or is divided by -- a day exactly, so the block has no
    leftover seconds at all), and the largest safe ``record`` is that many
    whole blocks plus one block's worth of ``within_block``.

    A second, independent limit is the day count itself: ``block_index *
    block_days`` must also stay under ``2**31 - 1`` (this is the *inherent*
    limit of an ``int32`` days-since-epoch representation, about 5.87
    million years -- no algorithm can move it), so the safe ``block_index``
    is also capped by ``(2**31 - 1) // block_days`` whenever ``block_days``
    is positive. The bound returned is the smaller of the two.

    This is a **guaranteed-safe lower bound**, proven exact by the
    derivation above and checked against Python's own (arbitrary-precision)
    integer arithmetic for every sampled ``record`` up to it in
    ``tests/unit/test_calendar.py`` -- not necessarily the largest record
    :func:`gregorian_instant` happens to still get right (the true failure
    point can be a little further out, since the bound above pessimistically
    assumes ``within_block`` is simultaneously at its own maximum), but
    every record up to it is exact.

    For a ``record_seconds`` that divides, or is divided by, a day exactly
    (the overwhelmingly common case: sub-daily, daily, or multi-day coupling)
    ``block_extra_seconds == 0`` and this returns ``2**31 - 1`` -- the full
    ``int32`` record range -- or, for a ``record_seconds`` that is itself an
    exact number of days (so the "block" is many records long to begin with),
    a number within about a day's worth of records of it; either way, no
    representable-range caveat worth naming. For the "1 month" (2629746 s)
    Gregorian step this migration review found broken at record 817
    (:func:`gregorian_instant`'s **History** note), it returns 82415 --
    nearly 6,900 years of monthly records, not 817 of them.

    Parameters
    ----------
    record_seconds : int
        Length of one record, in seconds. Must be positive.
    offset_seconds, start_seconds : int, optional
        The same arguments :func:`gregorian_instant` takes.

    Returns
    -------
    int
        The largest ``record`` guaranteed exact.

    """
    block_size, block_days, block_extra_seconds = _gregorian_instant_block(
        record_seconds, offset_seconds=offset_seconds, start_seconds=start_seconds
    )
    budget = _INT32_MAX - abs(int(offset_seconds)) - abs(int(start_seconds))
    slack = budget - block_size * int(record_seconds)

    max_block_index = _INT32_MAX  # unbounded by the (would-be-zero) term below
    if block_extra_seconds > 0:
        max_block_index = (slack + int(record_seconds)) // block_extra_seconds
    if block_days > 0:
        # `gregorian_instant`'s final `days = ... + block_index * block_days +
        # extra_days` needs the WHOLE sum under `2**31 - 1`, not just the
        # product: `extra_days` (the leftover from `within_block`'s own
        # contribution, independent of `block_index`) can itself be almost a
        # full `block_days` -- `within_block` ranges over nearly a whole
        # block, and a block's own length is `block_days` days by
        # construction -- so it is reserved as headroom here rather than
        # (wrongly) treated as negligible.
        within_block_days = budget // SECONDS_PER_DAY
        max_block_index = min(
            max_block_index, (_INT32_MAX - within_block_days) // block_days
        )
    # `record` itself is also an int32 value, so the bound can never exceed
    # what that type holds regardless of what the block arithmetic allows.
    return min(_INT32_MAX, block_size * max_block_index + (block_size - 1))
