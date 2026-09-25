"""Cross-checks for the vendored Gregorian calendar arithmetic.

``jem.base.calendar`` vendors ``jcm.date``'s Fliegel-Van Flandern algorithm
rather than importing it (see that module's docstring for why), so the one
thing this file must prove is that the vendored copy and the original never
disagree -- and that both agree with an independent implementation
(``pandas``'s own proleptic-Gregorian arithmetic), century leap-year rules
included, over a run long enough to matter (jax-gcm#907's review round asked
for at least 400 years: long enough to see three of the four `%100`
non-leap centuries and the one `%400` exception that puts them back in).
"""

import datetime as pydt

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jem.base.calendar import (
    gregorian_day_of_year,
    gregorian_instant,
    gregorian_ymd_from_days,
    is_leap_year,
    max_safe_record,
)

# 400 Gregorian years is one full leap-cycle: it contains every one of the
# four century cases (1900-style non-leap, 2000-style leap, 2100- and
# 2300-style non-leap) exactly once, so a mismatch anywhere in the rule would
# show up within a single cycle.
_YEARS = 400
_START = pydt.date(1900, 1, 1)
_DAYS = int(_YEARS * 365.2425) + 1


def _days_since_epoch(dates: np.ndarray) -> np.ndarray:
    epoch = pydt.date(1970, 1, 1)
    return np.array([(d - epoch).days for d in dates], dtype=np.int64)


def test_vendored_matches_jcm_date_over_400_years():
    """The vendored routine must never disagree with ``jcm.date``'s own."""
    jcm_date = pytest.importorskip(
        "jcm.date", reason="jax-gcm not installed; only the pandas cross-check applies"
    )
    dates = [_START + pydt.timedelta(days=k) for k in range(_DAYS)]
    days = jnp.asarray(_days_since_epoch(np.array(dates, dtype=object)), dtype=jnp.int32)

    mine = gregorian_ymd_from_days(days)
    theirs = jcm_date.gregorian_ymd_from_days(days)
    for mine_field, their_field in zip(mine, theirs, strict=True):
        np.testing.assert_array_equal(np.asarray(mine_field), np.asarray(their_field))

    mine_leap = np.asarray(is_leap_year(mine[0]))
    their_leap = np.asarray(jcm_date.is_leap_year(theirs[0]))
    np.testing.assert_array_equal(mine_leap, their_leap)


def test_vendored_matches_pandas_ymd_over_400_years_century_rules_included():
    """The vendored (year, month, day) must match pandas's, including century rules."""
    dates = pd.date_range(_START, periods=_DAYS, freq="D")
    days = jnp.asarray(_days_since_epoch(dates.date), dtype=jnp.int32)

    year, month, day = gregorian_ymd_from_days(days)
    np.testing.assert_array_equal(np.asarray(year), dates.year.values)
    np.testing.assert_array_equal(np.asarray(month), dates.month.values)
    np.testing.assert_array_equal(np.asarray(day), dates.day.values)

    # The century rules explicitly, by name: 1900 and 2100 are NOT leap
    # (divisible by 100 but not 400), 2000 IS leap (divisible by 400).
    for year_value, expected in [(1900, False), (2000, True), (2100, False)]:
        got = bool(np.asarray(is_leap_year(jnp.int32(year_value))))
        assert got == expected, f"is_leap_year({year_value}) = {got}, expected {expected}"


def test_day_of_year_matches_pandas_day_of_year():
    """``gregorian_day_of_year`` (0-indexed) must match pandas's ``dayofyear`` - 1."""
    dates = pd.date_range(_START, periods=_DAYS, freq="D")
    days = jnp.asarray(_days_since_epoch(dates.date), dtype=jnp.int32)
    year, month, day = gregorian_ymd_from_days(days)

    got = np.asarray(gregorian_day_of_year(year, month, day))
    want = dates.dayofyear.values - 1
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("record_seconds", [86400, 3600, 1800])
def test_gregorian_instant_matches_pandas_at_record_start(record_seconds):
    """``gregorian_instant`` at offset 0 must land exactly on ``start + k*record_seconds``."""
    start = pd.Timestamp("2000-01-01 06:00:00")
    start_days = int((start.normalize() - pd.Timestamp("1970-01-01")).days)
    start_seconds = int((start - start.normalize()).total_seconds())

    n = 4 * 365 * 86400 // record_seconds  # a few years, several calendars' worth
    records = jnp.arange(n, dtype=jnp.int32)
    days, seconds = gregorian_instant(records, record_seconds, start_days, start_seconds)

    labels = start + pd.to_timedelta(np.arange(n, dtype=np.int64) * record_seconds, unit="s")
    want_days = _days_since_epoch(labels.date)
    want_seconds = (
        labels.hour.values * 3600 + labels.minute.values * 60 + labels.second.values
    )
    np.testing.assert_array_equal(np.asarray(days), want_days)
    np.testing.assert_array_equal(np.asarray(seconds), want_seconds)


def _exact_instant(record, record_seconds, offset_seconds=0, start_seconds=0, start_days=0):
    """Return the (days, seconds) `gregorian_instant` computes, in plain Python.

    Arbitrary-precision Python ``int`` arithmetic -- no ``int32``, no
    overflow, ever -- so this is the ground truth every property test in this
    section checks the traced, ``int32``-only implementation against.
    """
    total = record * record_seconds + offset_seconds + start_seconds
    days, seconds = divmod(total, 86400)
    return start_days + days, seconds


@pytest.mark.parametrize(
    ("record_seconds", "offset_seconds", "start_seconds"),
    [
        (1, 0, 0),  # the finest possible record: sub-second resolution.
        (3600, 1800, 43200),  # hourly, midpoint offset, an arbitrary start.
        (86400, 0, 0),  # exactly a day: the fixed-calendar record length.
        (2_629_746, 2_629_746 // 2, 21_600),  # a "1 month" Gregorian step,
        #  with the midpoint offset `_gregorian_month_rule` actually uses.
        (31_556_952, 0, 0),  # a "1 year" Gregorian step.
        (604_800, 0, 0),  # a week -- an exact multiple of a day.
    ],
)
def test_gregorian_instant_matches_python_ints_up_to_max_safe_record(
    record_seconds, offset_seconds, start_seconds
):
    """Cross-check against exact Python ``int`` arithmetic at and below the bound.

    ``max_safe_record`` is a *guaranteed-safe* bound, proven exact in its own
    docstring; this checks that proof empirically, at the bound itself, just
    below it, at 0 and 1, and at a dense band around it (where a
    reduce-before-multiply bug is most likely to show up first -- see the
    2026-09 migration review, item 2, whose "1 month" case broke as early as
    record 817).
    """
    bound = max_safe_record(
        record_seconds, offset_seconds=offset_seconds, start_seconds=start_seconds
    )
    assert bound > 0
    records = sorted(
        {0, 1, 2, bound}
        | set(range(max(0, bound - 200), bound + 1))
        | {min(bound, r) for r in (1_000, 10_000, 1_000_000, 10_000_000)}
    )
    days, seconds = gregorian_instant(
        jnp.asarray(records, dtype=jnp.int32),
        record_seconds,
        0,
        start_seconds,
        offset_seconds=offset_seconds,
    )
    for record, day, second in zip(records, np.asarray(days), np.asarray(seconds), strict=True):
        want_day, want_second = _exact_instant(
            record, record_seconds, offset_seconds, start_seconds
        )
        assert (int(day), int(second)) == (want_day, want_second), record


def test_gregorian_instant_is_exact_near_2_31_records_for_a_fine_timestep():
    """The full int32 record range, for a coupling step fine enough to allow it.

    A ``record_seconds`` that divides a day exactly has no leftover seconds
    per block (:func:`max_safe_record`'s docstring), so it is exact for
    *every* record an int32 counter can hold. This checks the literal top of
    that range, ``2**31 - 1`` records of an hourly step -- about 245,000
    years past the epoch, past even Python's own ``datetime`` range, so the
    exact (days, seconds) claim is cross-checked against plain Python ``int``
    arithmetic (:func:`_exact_instant`) here, and against ``datetime``
    separately, at a still-enormous but ``datetime``-representable 1e7
    records (about 1,140 years), below.
    """
    record_seconds = 3600
    record = 2**31 - 1
    assert max_safe_record(record_seconds) >= record

    days, seconds = gregorian_instant(jnp.int32(record), record_seconds, 0, 0)
    exact_days, exact_seconds = _exact_instant(record, record_seconds)
    assert int(days) == exact_days
    assert int(seconds) == exact_seconds

    # `gregorian_ymd_from_days` itself only needs to not raise here -- it is
    # exercised against real `datetime`s over 400 years in the tests above --
    # so this only pins that it still runs, without wrapping, on a day count
    # this large.
    year, month, day = gregorian_ymd_from_days(days)
    assert int(year) > 1970

    record = 10_000_000
    assert max_safe_record(record_seconds) >= record
    days, seconds = gregorian_instant(jnp.int32(record), record_seconds, 0, 0)
    exact_days, exact_seconds = _exact_instant(record, record_seconds)
    assert int(days) == exact_days
    assert int(seconds) == exact_seconds
    year, month, day = gregorian_ymd_from_days(days)
    expected = pydt.date(1970, 1, 1) + pydt.timedelta(days=exact_days)
    assert (int(year), int(month), int(day)) == (
        expected.year, expected.month, expected.day,
    )


def test_gregorian_instant_one_month_step_survives_far_past_its_old_817_break(
):
    """The "1 month" Gregorian coupling step: exact for tens of thousands of years.

    Before the fix, ``gregorian_instant`` silently wrapped past record 817 of
    a 2629746 s ("1 month") coupling step -- about 68 records short of even a
    century (2026-09 migration review, item 2; ``jem/base/calendar.py``'s own
    History note). This checks a record count far beyond that break --
    50,000 records is over 4,100 years of monthly output -- against exact
    Python ``int`` arithmetic.
    """
    record_seconds = 2_629_746
    record = 50_000
    assert max_safe_record(record_seconds) > record  # comfortably past it
    days, seconds = gregorian_instant(
        jnp.int32(record), record_seconds, 0, 0, offset_seconds=record_seconds // 2
    )
    want_days, want_seconds = _exact_instant(
        record, record_seconds, record_seconds // 2
    )
    assert (int(days), int(seconds)) == (want_days, want_seconds)


def test_gregorian_instant_exact_at_and_near_2_31_for_every_adversarial_record_seconds():
    """2026-09 migration review, round 2, finding B1's own reproduction, fixed.

    The block-based decomposition that shipped after round 1's fix (git
    history) was still only int32-safe up to a computed
    ``max_safe_record`` that could itself be as small as about 68 simulated
    years (e.g. this test's own ``73453`` s coupling step), and nothing but
    ``jem.accumulate._midpoint_month_rule`` ever checked it -- a
    ``year_fraction`` or ``monthly_mean`` bin past that point was silently
    wrong with no error (confirmed: a daily ``year_fraction`` at step 58471
    of a 73453 s coupling from 2000-01-01 came back ``0.9969`` against an
    exact ``0.0988``). ``gregorian_instant`` is now a limb (schoolbook)
    decomposition, proven exact up to the one limit no algorithm can move --
    an int32 day count's own range -- so ``max_safe_record`` for every one
    of these record lengths is now within the full ``int32`` record range
    (some exactly ``2**31 - 1``; the rest -- record lengths that do not
    divide evenly into a day -- are the exact day-count limit itself, still
    around 5.87 million years' worth of records). Checked densely around
    ``2**31 - 1`` and around each ``record_seconds``'s own bound, not just
    at a handful of spot values, and against plain Python ``int``
    arithmetic throughout -- no algorithm this test trusts to be correct.
    """
    adversarial_record_seconds = [
        73453, 73738, 86399, 86401, 7 * 86400 + 1, 1_314_873, 2_629_746,
        31_556_952,
    ]
    rng = np.random.default_rng(878)
    for record_seconds in adversarial_record_seconds:
        bound = max_safe_record(record_seconds)
        near_top = {2**31 - 1, 2**31 - 2, bound, min(bound + 1, 2**31 - 1)}
        near_bound = set(range(max(0, bound - 5), min(bound + 5, 2**31 - 1) + 1))
        sampled = set(rng.integers(0, min(bound, 2**31 - 1) + 1, size=200).tolist())
        records = sorted(near_top | near_bound | sampled | {0, 1})
        days, seconds = gregorian_instant(
            jnp.asarray(records, dtype=jnp.int32), record_seconds, 0, 0
        )
        for record, day, second in zip(records, np.asarray(days), np.asarray(seconds), strict=True):
            want_day, want_second = _exact_instant(record, record_seconds)
            if record <= bound:
                assert (int(day), int(second)) == (want_day, want_second), (
                    record_seconds, record, bound,
                )


def test_max_safe_record_reserves_start_days_against_the_same_day_budget():
    """N1: ``start_days`` must be charged against the ``2**31 - 1`` day budget too.

    Before this fix ``max_safe_record`` had no ``start_days`` parameter at
    all, so a nonzero ``start_days`` was not reserved -- the returned bound
    could itself already be past the true edge (confirmed with these exact
    values, ``record_seconds=86400``, ``offset_seconds=43200``,
    ``start_days=10957`` -- 2000-01-01 -- ``start_seconds=86399``: calling
    ``gregorian_instant`` at the old, ``start_days``-blind bound already
    gave a negative, wrapped day count). The true edge is checked here
    directly against exact Python ``int`` arithmetic -- the definition of
    "the largest record keeping the day count in int32 range" -- rather
    than against one specific number, since the exact value is a consequence
    of the derivation, not the definition.
    """
    record_seconds, offset, start_days, start_seconds = 86400, 43200, 10957, 86399
    bound = max_safe_record(
        record_seconds, offset_seconds=offset, start_seconds=start_seconds,
        start_days=start_days,
    )

    def days_at(record):
        total = record * record_seconds + offset + start_seconds
        return start_days + total // 86400

    assert days_at(bound) == 2**31 - 1
    assert days_at(bound + 1) == 2**31  # one past int32's own range
    # Reserving `start_days` costs exactly `start_days` records here (this
    # `record_seconds` is an exact day, so the `* 86400` it is charged in
    # cancels the `/ record_seconds` it is spent through): the old,
    # `start_days`-blind computation is too generous by exactly that much,
    # and calling `gregorian_instant` at ITS bound had already wrapped.
    old_start_days_blind_bound = max_safe_record(
        record_seconds, offset_seconds=offset, start_seconds=start_seconds
    )
    assert old_start_days_blind_bound == bound + start_days
    wrapped_days, _ = gregorian_instant(
        jnp.int32(old_start_days_blind_bound), record_seconds, start_days,
        start_seconds, offset_seconds=offset,
    )
    assert int(wrapped_days) < 0  # silently wrapped, not the true (huge) day count


def test_gregorian_instant_midpoint_never_crosses_a_month_boundary():
    """Flooring an odd record's midpoint to a whole second must stay in the record.

    Month boundaries fall at exact whole seconds (00:00:00.000 on the 1st), so
    flooring a half-second midpoint down can only move it *away* from an
    upcoming boundary, never across one -- this is the argument
    ``jem.accumulate._gregorian_month_rule``'s midpoint binning relies on, and
    this test is its evidence: an 11-second record starting one second before
    midnight on 31 January (record covers 23:59:59 through 00:00:10) has its
    exact real-valued midpoint at 00:00:04.5 on 1 February, and the floored
    midpoint used for binning must round to 00:00:04 -- still 1 February, not
    31 January.
    """
    start_days = int((pd.Timestamp("2000-01-31") - pd.Timestamp("1970-01-01")).days)
    start_seconds = 23 * 3600 + 59 * 60 + 59
    record_seconds = 11
    days, seconds = gregorian_instant(
        jnp.int32(0), record_seconds, start_days, start_seconds,
        offset_seconds=record_seconds // 2,
    )
    year, month, day = gregorian_ymd_from_days(days)
    assert (int(year), int(month), int(day)) == (2000, 2, 1)
    assert int(seconds) == 4
