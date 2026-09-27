"""Turning a run length -- a fixed duration or a calendar end date -- into seconds.

``run_chunked``, :func:`jem.accumulate.monthly_mean` and
:func:`jem.accumulate.windowed_mean` each take how far to run in one of two
mutually exclusive forms: ``total_time``, a fixed duration
(``jcm.date.parse_duration_seconds``: ``"90 days"``, ``"6 hours"``, or a
number of days), or ``end_time``, an absolute ISO date/datetime
(``jcm.date.to_datetime``). A fixed duration cannot spell a calendar target --
"10 years" is not a fixed number of seconds, since a year is 365 or 366 days
depending on which one, and ``parse_duration_seconds`` refuses it for exactly
that reason -- so ``end_time`` is how a run is given one instead: "run to
2011-01-01" rather than a day count worked out by hand.

:func:`resolve_duration_seconds` is the one place both forms are turned into
whole seconds, so the three callers above agree on what "the same run length"
means rather than each computing it from ``end_time`` its own way.
"""

from __future__ import annotations

from typing import Any


def resolve_duration_seconds(
    coupler: Any,
    total_time: str | float | None,
    end_time: str | None,
    *,
    required: bool = True,
    what: str = "total_time",
) -> int | None:
    """Return ``total_time``/``end_time`` as whole seconds from ``coupler.start_date``.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        Supplies ``start_date``, which ``end_time`` is measured from.
    total_time : str or float, optional
        A fixed duration, in ``jcm.date.parse_duration_seconds`` forms.
    end_time : str, optional
        An ISO date/datetime (``jcm.date.to_datetime``) after
        ``coupler.start_date``. The length returned is
        ``end_time - coupler.start_date``, in whole seconds -- computed on
        the host from ``jax_datetime`` arithmetic, never as float days, so a
        run whose length is exactly some number of seconds is never off by a
        rounding error that a multi-year run would otherwise accumulate.
    required : bool
        If True (the default, what ``run_chunked`` needs), giving neither
        argument is a ``ValueError``. If False, giving neither returns
        ``None`` instead -- for a caller (``monthly_mean``, ``windowed_mean``)
        whose accumulator can also be sized another way (``n_months``,
        ``n_windows``, or a bare window pattern), so "no run length was
        given" is not by itself an error there.
    what : str
        The name ``total_time`` is known by in the message, so a caller whose
        argument is spelled differently still gets an accurate one.

    Returns
    -------
    int or None
        The duration in whole seconds, or ``None`` when neither was given and
        ``required`` is False.

    Raises
    ------
    ValueError
        If both are given, if neither is given and ``required`` is True, or
        if ``end_time`` is not strictly after ``coupler.start_date``.

    """
    if total_time is not None and end_time is not None:
        raise ValueError(
            f"Give at most one of {what} and end_time: both say how long the "
            f"run is (got {what}={total_time!r}, end_time={end_time!r})."
        )
    if total_time is None and end_time is None:
        if not required:
            return None
        raise ValueError(
            f"Give exactly one of {what} and end_time: a run's length is "
            f"either a fixed duration ({what}, e.g. \"90 days\") or a "
            "calendar end date (end_time, e.g. \"2011-01-01\") -- a fixed "
            "duration cannot spell a calendar target (\"10 years\" is not a "
            "fixed number of seconds), which is what end_time is for."
        )
    if end_time is not None:
        # Deferred so that importing this module does not pull in jax-gcm (and
        # with it the whole atmosphere); `jem.driver` and `jem.accumulate`
        # import `jcm.date` the same way and for the same reason.
        from jcm.date import to_datetime

        delta = to_datetime(end_time, name="end_time") - to_datetime(
            coupler.start_date
        )
        seconds = int(delta.days) * 86400 + int(delta.seconds)
        if seconds <= 0:
            raise ValueError(
                f"end_time={end_time!r} is not after the coupler's "
                f"start_date {coupler.start_date.to_pydatetime().isoformat()!r}: "
                "a run must have a positive duration."
            )
        return seconds

    from jcm.date import parse_duration_seconds

    return int(parse_duration_seconds(total_time))
