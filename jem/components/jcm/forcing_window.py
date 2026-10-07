"""Restrict an atmosphere's time-varying forcing to the dates a run covers.

jax-gcm keeps a boundary condition read from a file as a
:class:`jcm.forcing.TimeSeries` -- every record of it, on the device -- and
selects the current one on each internal step. JEM holds that forcing in the
atmosphere's carry (exchangers write parts of it), and a carry is what a
differentiated trajectory stores at every checkpoint. At high resolution
that is decisive: jax-gcm's packaged T255 daily climatology is five
``(365, 768, 384)`` fields, 4.3 GB in double precision, carried -- and, under
``jax.checkpoint``, stored -- at every coupling step of a run that needs
perhaps a fortnight of it.

:func:`restrict_forcing_to_window` returns the same forcing holding only the
records a window needs. It changes no value the model sees: every step of a
run inside the window selects exactly the record it would have selected from
the full series (``tests/unit/test_forcing_window.py`` checks this against
:meth:`jcm.forcing.ForcingData.select`).
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax_datetime as jdt
import numpy as np
from jcm.forcing import (
    BY_DATE,
    BY_DATE_INTERP,
    WRAP_YEAR,
    ForcingData,
    TimeSeries,
    make_time_series,
)

logger = logging.getLogger(__name__)


def _to_datetime64(value: Any) -> np.datetime64:
    if isinstance(value, jdt.Datetime):
        return np.datetime64(value.to_datetime64(), "s")
    return np.datetime64(value, "s")


def _climatology_record(dates: np.ndarray, n_records: int) -> np.ndarray:
    """Index of the record a ``WRAP_YEAR`` table selects on each midnight.

    The same rule as jax-gcm's selection: twelve records are the calendar
    months; 365/366 are nominal-date daily records, a 365-record table
    holding Feb 28 through Feb 29.
    """
    months = dates.astype("datetime64[M]").astype(np.int64) % 12 + 1
    if n_records == 12:
        return months - 1
    days = (dates.astype("datetime64[D]") - dates.astype("datetime64[M]")).astype(np.int64) + 1
    nominal = np.arange("2001-01-01", "2002-01-01", dtype="datetime64[D]")
    if n_records == 366:
        nominal = np.arange("2000-01-01", "2001-01-01", dtype="datetime64[D]")
    table_months = nominal.astype("datetime64[M]").astype(np.int64) % 12 + 1
    table_days = (nominal - nominal.astype("datetime64[M]")).astype(np.int64) + 1
    keys = table_months * 32 + table_days
    return np.asarray(np.searchsorted(keys, months * 32 + days, side="right") - 1, dtype=np.int64)


def _restrict(series: TimeSeries, start: np.datetime64, end: np.datetime64) -> TimeSeries:
    mode = int(np.asarray(series.align_mode))
    n_records = series.values.shape[0]
    # `persist` is a 0-d int array (or None, which means strict);
    # `make_time_series` takes the policy as a Python int code.
    persist = "strict" if series.persist is None else int(np.asarray(series.persist))
    if mode == WRAP_YEAR:
        if n_records not in (12, 365, 366):
            # Selected by fractions of the year rather than calendar dates
            # (MACv2-SP's weekly cycle): a record boundary need not fall on
            # midnight, so midnight samples would not reproduce it. Such
            # tables are small; keep them whole.
            return series
        first = start.astype("datetime64[D]")
        last = end.astype("datetime64[D]")
        dates = np.arange(first, last + np.timedelta64(1, "D"), dtype="datetime64[D]")
        records = _climatology_record(dates, n_records)
        # A climatology record holds from its nominal date's midnight to the
        # next one, which is exactly what a by-date series sampled at every
        # midnight in the window selects.
        return make_time_series(
            series.values[records], dates.astype("datetime64[s]"), align_mode=BY_DATE)
    if mode in (BY_DATE, BY_DATE_INTERP):
        times = np.asarray(series.times.to_datetime64()).astype("datetime64[s]")
        # The last record at or before the start (what a step-held or
        # interpolated lookup at the start reads) through the first record at
        # or after the end.
        lo = max(int(np.searchsorted(times, start, side="right")) - 1, 0)
        hi = min(int(np.searchsorted(times, end, side="left")), n_records - 1)
        return make_time_series(
            series.values[lo:hi + 1], times[lo:hi + 1], align_mode=mode, persist=persist)
    return series


def restrict_forcing_to_window(forcing: ForcingData, start: Any, end: Any) -> ForcingData:
    """Return ``forcing`` with every time series cut down to ``[start, end]``.

    Parameters
    ----------
    forcing : jcm.forcing.ForcingData
        As built by ``jcm.runners.build_forcing`` (or any jax-gcm reader).
    start, end : jax_datetime.Datetime, numpy.datetime64 or str
        The interval a run will integrate over; ``end`` is the end of its
        last step.

    Returns
    -------
    jcm.forcing.ForcingData
        A climatology (``WRAP_YEAR``) series of monthly or daily records
        becomes a by-date series of one record per midnight from ``start``'s
        day to ``end``'s; a by-date series keeps the records that bracket the
        window. Anything else, and every field that is not a time series, is
        returned unchanged. Every step inside ``[start, end]`` selects the
        same values from the result as from ``forcing``.

    """
    start64, end64 = _to_datetime64(start), _to_datetime64(end)
    if end64 < start64:
        raise ValueError(f"window end {end64} is before its start {start64}")
    before = sum(leaf.nbytes for leaf in jax.tree.leaves(forcing) if hasattr(leaf, "nbytes"))
    restricted = jax.tree.map(
        lambda leaf: _restrict(leaf, start64, end64) if isinstance(leaf, TimeSeries) else leaf,
        forcing,
        is_leaf=lambda leaf: isinstance(leaf, TimeSeries),
    )
    after = sum(leaf.nbytes for leaf in jax.tree.leaves(restricted) if hasattr(leaf, "nbytes"))
    logger.info(
        "Forcing restricted to %s..%s: %.1f MB -> %.1f MB.",
        start64, end64, before / 1e6, after / 1e6,
    )
    return restricted
