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
an int32-safe **limb (schoolbook) multiply-then-divide** (see its own
docstring for the design, and :func:`max_safe_record` for the one limit that
survives it -- the *inherent* range of an int32 day count, not an artifact of
the algorithm) that turns a record counter that can run up to ``2**31 - 1``
into the (days, seconds)-since-epoch of one instant within that record,
without ever forming an intermediate that could overflow ``int32``. Three
callers use it -- :func:`jem.accumulate._gregorian_month_rule` and
:func:`jem.accumulate._midpoint_month_rule` at the record's **midpoint** (the
bin a record counts in, matching the midpoint labelling convention
:class:`jem.base.component.TimeAxis` writes -- see that module's decision
record) and :class:`~jem.base.component.CouplingTime` at the record's
**start** (``year_fraction`` is defined at the start of a step, matching its
pre-existing convention on the ``365_day`` calendar).

**2026-09 review, round 2 (finding B1).** The version of this function that
shipped after round 1's fix (see git history / the CHANGELOG) still had a
*bound*, not a universal exactness proof: it multiplied by reducing the
record counter modulo a single static "block" chosen from ``record_seconds``
alone, which was int32-safe only up to a computed ``max_safe_record`` that
could be as small as about 68 simulated years (e.g. a 73453 s coupling step)
-- smaller than several real coupled-run lengths -- and nothing outside
:func:`jem.accumulate._midpoint_month_rule` ever checked it, so a run past it
silently wrapped to a wrong instant with no error at all (confirmed: a daily
``year_fraction`` at step 58471 of a 73453 s coupling from 2000-01-01 came
back ``0.9969`` against an exact ``0.0988``, and a 400000-step Gregorian
``monthly_mean`` at the same step length mis-binned three records). The limb
decomposition below removes the bound rather than raising it: it is exact for
*every* ``record`` up to ``2**31 - 1`` and *every* whole-second
``record_seconds``, up to the one limit no algorithm can move -- an ``int32``
day count's own representable range (:func:`max_safe_record`, now exactly
that limit, including the ``start_days`` offset it did not reserve before).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

SECONDS_PER_DAY = 86_400

# The largest magnitude a JAX int32 value can hold. `max_safe_record` sizes
# the one genuine limit -- an int32 day count's own range -- against this,
# not against `jnp.iinfo(jnp.int32).max`, so the bound is visible as a plain
# number in its own docstring without importing NumPy just for it.
_INT32_MAX = 2**31 - 1

# `gregorian_instant`'s limb width and count: `record` is `int32`, so it fits
# in 31 bits, and 3 limbs of `_LIMB_BITS` bits cover any width up to `3 *
# _LIMB_BITS` -- 42 for 14, comfortably over 31. `_LIMB_BITS` itself is
# chosen so that `SECONDS_PER_DAY * _LIMB_BASE` (the largest intermediate the
# per-limb combine step forms -- see `gregorian_instant`'s docstring) stays
# well under `2**31`: `86400 * 2**14 == 1415577600 < 2**31 - 1`, whereas
# `86400 * 2**15` would not.
_LIMB_BITS = 14
_LIMB_BASE = 1 << _LIMB_BITS
_N_LIMBS = 3

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


def _digit_tables(record_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-digit ``(days, seconds)`` lookup tables ``gregorian_instant`` reduces through.

    Entry ``d`` of each table is ``divmod(d * record_seconds, SECONDS_PER_DAY)``
    for ``d`` in ``[0, _LIMB_BASE)`` -- computed here, in plain Python/NumPy
    ``int64`` (unbounded for any ``record_seconds`` this codebase builds, and
    never traced), so it is exact regardless of how large ``record_seconds``
    is; only the *stored* dtype is ``int32``, and only because every value
    that ends up in it is checked to fit one first.

    This is the one place a per-digit product (``d * record_seconds``) is
    ever formed: :func:`gregorian_instant` looks a digit's contribution up
    here instead of multiplying it live, which is what lets the traced
    computation stay in ``int32`` however large ``record_seconds`` is.

    Parameters
    ----------
    record_seconds : int
        Length of one record, in seconds. Must be positive.

    Returns
    -------
    days, seconds : numpy.ndarray
        ``int32`` arrays of length ``_LIMB_BASE``.

    Raises
    ------
    ValueError
        If ``record_seconds`` is not positive, or is so large that even one
        digit's own contribution (``(_LIMB_BASE - 1) * record_seconds``, in
        days) would not fit ``int32`` -- true only for a ``record_seconds``
        far beyond anything this codebase constructs (order ``10**13`` s),
        included only so a pathological input fails here with a clear
        message rather than downstream with a wrapped one.

    """
    record_seconds = int(record_seconds)
    if record_seconds <= 0:
        raise ValueError(
            "record_seconds must be a positive whole number of seconds; "
            f"got {record_seconds!r}."
        )
    digits = np.arange(_LIMB_BASE, dtype=np.int64)
    products = digits * np.int64(record_seconds)  # exact int64; see docstring
    days, seconds = np.divmod(products, SECONDS_PER_DAY)
    if int(days.max()) > _INT32_MAX:
        raise ValueError(
            f"record_seconds={record_seconds!r} is too large: even a single "
            "digit's own contribution to gregorian_instant's limb reduction "
            "would not fit an int32 day count."
        )
    return days.astype(np.int32), seconds.astype(np.int32)


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
    offsets). ``record`` and every other argument are assumed non-negative
    except ``start_days``, which may be negative (a date before the epoch);
    nothing here supports a negative ``record``, ``record_seconds``,
    ``offset_seconds`` or ``start_seconds``, which no caller in this codebase
    ever constructs.

    **The naive way -- multiply the record counter by ``record_seconds`` and
    add the offsets -- costs one product that grows with the run**: JAX
    indices are ``int32`` by default, and a whole-run count of seconds passes
    ``2**31`` after 68 years of simulated time *regardless of the coupling
    step* (not "a few decades of daily coupling" -- a finer step reaches the
    same elapsed-seconds total in the same elapsed time, just over more
    records; the count that overflows is a count of *seconds*, which cares
    about elapsed time, not step size).

    **The fix is a limb (schoolbook) multiply-then-divide**, exact for any
    ``record`` an ``int32`` can hold and any whole-second ``record_seconds``,
    with only ``int32`` arithmetic throughout. Write ``record`` in base
    ``_LIMB_BASE`` (``2**14``, chosen so that ``SECONDS_PER_DAY *
    _LIMB_BASE`` -- the largest intermediate this ever forms -- stays under
    ``2**31``): ``record = d[2]*_LIMB_BASE**2 + d[1]*_LIMB_BASE + d[0]``,
    3 limbs comfortably covering any ``int32`` (``3 * 14 == 42 >= 31``).
    Processing the limbs from *most* significant to *least* (Horner's rule,
    run on the ``(days, seconds)`` pair rather than on ``record`` itself),
    maintain the exact ``(days, seconds)`` of ``acc * record_seconds`` for
    the *partial* record ``acc`` built from the limbs seen so far
    (``_digit_tables`` gives this directly for a single limb: ``acc = d``);
    folding in the next limb ``d`` replaces ``acc`` with ``acc * _LIMB_BASE +
    d``, so its seconds and days update as

    .. code-block:: text

        combined     = seconds * _LIMB_BASE + table_seconds[d]
        extra, seconds = divmod(combined, SECONDS_PER_DAY)
        days         = days * _LIMB_BASE + table_days[d] + extra

    -- exactly (`combined` reconstructs `(acc*_LIMB_BASE+d)*record_seconds`'s
    own seconds-of-day component before re-reducing it, and the days update
    absorbs the carry; see the proof below, and
    ``tests/unit/test_calendar.py``'s property tests -- densely sampled
    near ``2**31 - 1`` and near every tested ``record_seconds``'s own
    bound, not merely a handful of spot values -- for the empirical
    confirmation of it).

    **Why the loop's own arithmetic stays int32-safe.** ``combined`` is
    bounded by ``SECONDS_PER_DAY * _LIMB_BASE + SECONDS_PER_DAY`` regardless
    of ``record`` or ``record_seconds`` (``seconds < SECONDS_PER_DAY`` and
    ``table_seconds[d] < SECONDS_PER_DAY`` are both invariants), which is
    what ``_LIMB_BASE`` was chosen to keep under ``2**31``. The loop's own
    ``days`` accumulator is bounded by the same induction: at every step,
    ``acc`` (this step's partial record, built from the limbs folded in so
    far) satisfies ``acc <= record`` and ``acc * _LIMB_BASE <= acc'`` for the
    *next* partial record ``acc'``, so ``days * _LIMB_BASE <= floor(acc *
    _LIMB_BASE * record_seconds / SECONDS_PER_DAY) <= floor(acc' *
    record_seconds / SECONDS_PER_DAY)`` -- i.e. the multiply this function
    performs on ``days`` at each step is bounded by the *next* step's own
    (bounded, by the same induction) result, and so transitively by
    ``floor(record * record_seconds / SECONDS_PER_DAY)``, the loop's *own*
    final value, computed with no ``start_days`` in it at all (that is added
    only once, at the very end, line 350 below).

    That final loop value is what :func:`max_safe_record` bounds by
    ``_INT32_MAX - start_days`` (its own ``day_budget``) -- **not** by
    ``_INT32_MAX`` itself -- exactly so the eventual ``start_days + days``
    stays in range. For ``start_days >= 0`` those two bounds coincide and the
    loop's own ``days`` therefore never needs to represent more than
    ``_INT32_MAX`` either. For a sufficiently negative ``start_days``,
    though, ``_INT32_MAX - start_days`` **exceeds** ``_INT32_MAX``, so the
    loop's own ``days`` -- still exact arithmetic *mod* ``2**32``, just no
    longer within int32's own positive range -- can itself need more than an
    int32's worth of magnitude before the final ``+ start_days`` brings the
    total back down (2026-09 review, round 3, finding 8; confirmed:
    ``record_seconds=172800, start_days=-10**9`` reaches a loop ``days`` of
    ``3147483646``, past ``2**31 - 1``, at :func:`max_safe_record`'s own
    bound for that input). This is harmless, not merely "usually fine":
    ``jnp.int32`` addition and subtraction are two's-complement, i.e. exact
    modulo ``2**32``, so ``(loop_days mod 2**32) + start_days`` and
    ``(loop_days + start_days) mod 2**32`` are the *same* value -- and since
    :func:`max_safe_record` guarantees the true, infinite-precision
    ``start_days + loop_days`` fits in ``[-2**31, 2**31 - 1]`` whenever
    ``record`` is within its bound, that unique representable value is
    exactly what the final addition below computes, regardless of whether
    the loop's own running total needed to "overflow" (wrap through
    ``2**32``) to get there. So this is still a proof, not an empirical
    bound: there is no ``record``/``record_seconds``/``start_days``
    combination for which this function is inexact below
    :func:`max_safe_record`'s own limit (the inherent range of an int32 *day
    count*, about 5.87 million years for a realistic coupling step), which no
    algorithm can move -- only the claim that every *named local* stays
    within int32's own dynamic range along the way is specific to
    ``start_days >= 0``; the arithmetic is exact modulo ``2**32``
    unconditionally.

    **History.** Two earlier decompositions shipped and were superseded:
    reducing ``record`` modulo ``SECONDS_PER_DAY // gcd(record_seconds,
    SECONDS_PER_DAY)`` (int32-safe only up to ``D <= 24855``, i.e. as little
    as 817 records for a "1 month" 2629746 s step -- 2026-09 review, round 1,
    finding 2); then reducing modulo a single static "block" sized from
    ``record_seconds`` alone (int32-safe up to a computed
    ``max_safe_record``, but that bound could itself be as small as about 68
    simulated years -- e.g. a 73453 s coupling step -- and nothing but
    :func:`jem.accumulate._midpoint_month_rule` ever checked it, so a longer
    run silently wrapped with no error; 2026-09 review, round 2, finding B1).
    Both are corrected here rather than repeated, and neither is a
    description of what this function does any more.

    Parameters
    ----------
    record : jax.Array
        int32 (or castable) record counter, traced. Non-negative.
    record_seconds : int
        Length of one record, in seconds. Static (a Python int, not traced).
        Must be positive.
    start_days, start_seconds : int
        Days and seconds since the Unix epoch of record 0's start. Static.
        ``start_days`` may be negative; ``start_seconds`` must not be.
    offset_seconds : int, optional
        Seconds into record ``record`` at which to evaluate the instant.
        Static; both callers pass ``0`` or ``record_seconds // 2``, but
        nothing here requires ``offset_seconds <= record_seconds`` --
        :func:`jem.accumulate._midpoint_month_rule` also adds a pattern
        phase that is typically much larger than one record. Non-negative.

    Returns
    -------
    days, seconds : jax.Array
        int32 arrays: whole days since the Unix epoch, and the seconds within
        that day, of the requested instant, exact for any ``record`` up to
        ``max_safe_record(record_seconds, offset_seconds=...,
        start_seconds=..., start_days=...)`` (:func:`max_safe_record`).

    Raises
    ------
    ValueError
        If ``record_seconds`` is not positive -- see :func:`_digit_tables`.

    """
    offset_seconds = int(offset_seconds)
    start_seconds = int(start_seconds)
    table_days, table_seconds = _digit_tables(record_seconds)
    table_days_j = jnp.asarray(table_days)
    table_seconds_j = jnp.asarray(table_seconds)

    # Base-`_LIMB_BASE` digits of `record`, least significant first, then
    # reversed so Horner's rule below folds in the MOST significant limb
    # first -- `digits[0]` after the reversal is `record`'s top limb.
    remaining = jnp.asarray(record, dtype=jnp.int32)
    digits = []
    for _ in range(_N_LIMBS - 1):
        remaining, digit = jnp.divmod(remaining, _LIMB_BASE)
        digits.append(digit)
    digits.append(remaining)  # the top limb: whatever is left after `_N_LIMBS - 1` shifts
    digits.reverse()

    days = table_days_j[digits[0]]
    seconds = table_seconds_j[digits[0]]
    for digit in digits[1:]:
        combined = seconds * _LIMB_BASE + table_seconds_j[digit]
        extra_days, seconds = jnp.divmod(combined, SECONDS_PER_DAY)
        days = days * _LIMB_BASE + table_days_j[digit] + extra_days

    final_seconds = seconds + offset_seconds + start_seconds
    extra_days, seconds = jnp.divmod(final_seconds, SECONDS_PER_DAY)
    days = start_days + days + extra_days
    return days, seconds


def max_safe_record(
    record_seconds: int,
    *,
    offset_seconds: int = 0,
    start_seconds: int = 0,
    start_days: int = 0,
) -> int:
    """Return the largest ``record`` for which ``gregorian_instant``'s day count fits int32.

    :func:`gregorian_instant` is now exact for *every* ``record`` up to this
    bound (a proof, not an empirical one -- see that function's own
    docstring), so this is not a workaround for an imprecise algorithm: it is
    the **inherent** range of an ``int32`` days-since-epoch value, about 5.87
    million years, which no algorithm can extend. A caller that knows its own
    maximum record count ahead of time -- the only place in this codebase
    that does is :func:`jem.accumulate.monthly_mean`'s sequential form,
    handed ``n_months``/``total_time`` up front, and
    :func:`jem.driver.run_chunked`'s own up-front duration validation --
    should check it against this bound **at construction**, before anything
    is traced, and refuse rather than silently integrate or bin past it. A
    caller with no such bound (``CouplingTime.year_fraction``, called once
    per step of a run whose length is not fixed in advance) cannot check
    this per call -- ``record`` is traced -- so it is protected differently:
    ``jem.driver.run_chunked``'s own up-front check
    (``_check_step_counters_fit_int32``) refuses, before anything is
    compiled, any run whose coupled step counter could ever reach a record
    past this bound. That is a guarantee, not a hope that the bound happens
    to be large -- though for any realistic coupling step it also is (the
    inherent int32 range above, about 5.87 million simulated years).

    **The bound, derived exactly.** ``gregorian_instant`` returns ``days =
    start_days + floor((record * record_seconds + offset_seconds +
    start_seconds) / SECONDS_PER_DAY)``, non-decreasing in ``record`` since
    every other term is non-negative; the largest ``record`` keeping ``days
    <= 2**31 - 1`` is (writing ``K = 2**31 - 1 - start_days`` for the day
    budget ``start_days`` leaves, and using ``floor(x / d) <= K <=> x <= d*K
    + (d - 1)`` for the exact seconds budget that corresponds to)::

        record <= (SECONDS_PER_DAY * K + SECONDS_PER_DAY - 1
                   - offset_seconds - start_seconds) / record_seconds

    floored, and clamped to ``[0, 2**31 - 1]`` (the ``record`` dtype's own
    range). Unlike the bound this replaces (2026-09 review, round 2, finding
    N1), ``start_days`` is charged against the same ``2**31 - 1`` day budget
    every other term is, rather than left unreserved -- the earlier bound
    could itself be wrong by exactly the amount a nonzero ``start_days``
    left unaccounted for (confirmed: ``record_seconds=86400,
    offset_seconds=43200, start_days=10957, start_seconds=86399`` -- 2000-01-01
    -- computed a bound of 2147473170, one more than 2147473169, the actual
    edge, and ``gregorian_instant`` at that (wrong) bound had already wrapped
    to a negative day count).

    Parameters
    ----------
    record_seconds : int
        Length of one record, in seconds. Must be positive.
    offset_seconds, start_seconds : int, optional
        The same arguments :func:`gregorian_instant` takes. Non-negative.
    start_days : int, optional
        The same argument :func:`gregorian_instant` takes. May be negative
        (a start date before the epoch), which only *enlarges* the budget.

    Returns
    -------
    int
        The largest ``record`` for which every intermediate
        :func:`gregorian_instant` forms, and its result, is exact.

    Raises
    ------
    ValueError
        If ``record_seconds`` is not positive, or if ``start_days``,
        ``offset_seconds`` and ``start_seconds`` together already exceed
        what an ``int32`` day count can hold before a single record is added
        (so there is no non-negative ``record``, not even ``0``, this is
        safe for) -- whether that is because ``start_days`` alone is already
        past int32's range, or because it is in range but ``offset_seconds``
        / ``start_seconds`` alone are large enough to push even record 0's
        day count past it.

    """
    record_seconds = int(record_seconds)
    if record_seconds <= 0:
        raise ValueError(
            "record_seconds must be a positive whole number of seconds; "
            f"got {record_seconds!r}."
        )
    offset_seconds = int(offset_seconds)
    start_seconds = int(start_seconds)
    start_days = int(start_days)
    day_budget = _INT32_MAX - start_days
    seconds_budget = (
        SECONDS_PER_DAY * day_budget + (SECONDS_PER_DAY - 1)
        - offset_seconds - start_seconds
    )
    # One check covers both ways this can happen: `start_days` alone past
    # int32 (`day_budget < 0`, which drives `seconds_budget` very negative on
    # its own), or `start_days` in range but `offset_seconds`/`start_seconds`
    # alone big enough to push even record 0's day count past int32 (2026-09
    # review, round 3, finding 3 -- the previous version only checked the
    # first case, and `max(0, ...)` clamped the second's negative result up
    # to a lying `0`, claiming record 0 was safe when it was not:
    # `record_seconds=1, offset_seconds=400*86400, start_days=2**31-6` lands
    # record 0 alone 400 days past int32's own range). Either way, a negative
    # `seconds_budget` means there is no non-negative `record` -- not even
    # 0 -- this is safe for, which is exactly what the docstring promises to
    # raise on rather than silently answer.
    if seconds_budget < 0:
        raise ValueError(
            f"start_days={start_days!r}, offset_seconds={offset_seconds!r} "
            f"and start_seconds={start_seconds!r} together already exceed "
            "what an int32 day count can hold before a single record is "
            "added; there is no safe record, not even 0."
        )
    return min(_INT32_MAX, seconds_budget // record_seconds)
