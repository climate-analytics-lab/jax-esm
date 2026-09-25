"""Cross-checks for the vendored Gregorian calendar arithmetic.

``jem.base.calendar`` vendors ``jcm.date``'s Fliegel-Van Flandern algorithm
rather than importing it (see that module's docstring for why), so the one
thing this file must prove is that the vendored copy and the original never
disagree -- and that both agree with an independent implementation
(``pandas``'s own proleptic-Gregorian arithmetic), century leap-year rules
included, over a run long enough to matter (jax-esm#907's review round asked
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
