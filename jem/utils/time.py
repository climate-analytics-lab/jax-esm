"""The output time coordinate, written the way JCM writes it.

A coupled run produces one dataset per component. They are only useful
together if ``xr.merge`` aligns them, and that requires the ``time``
coordinate to carry not just the same *instants* but the same *representation*
of them, down to the last bit. This module is the one place that
representation is written down, so a slab dataset and a JCM dataset cannot
drift apart.

What JCM does (``jcm/predictions.py``, ``ModelPredictions._trajectory_dataset``)::

    times = start_date.delta.days + sim_time_days + save_interval * (arange(n) + 1)
    ds['time'] = (times * (timedelta64(1, 'D') / timedelta64(1, 'ns'))
                 ).astype('datetime64[ns]')

Three properties of that are load-bearing and are reproduced here:

1. **datetime64[ns], absolute.** Not "hours since <start>", which is what the
   slab models used to write and which merges with nothing.
2. **Via float64 days since the 1970 epoch.** ``days * 86400e9`` overflows the
   exactly-representable range of float64 (a 2001 date is ~9.8e17 ns, whose
   ulp is 128 ns), so the resulting instants are not exact to the nanosecond.
   They are, however, *identically* inexact for both models as long as both go
   through the same arithmetic -- which is the property that makes the merge
   work, and the reason this helper does not compute the exact integer
   nanosecond count instead.
3. **Labelled at the END of the interval the record covers.** JCM's frame index
   runs ``arange(outer_steps) + 1``: a saved frame holds the state *after* the
   interval, so it is stamped with the time it actually holds. A component's
   step-``k`` diagnostics are likewise the state after the step, so record
   ``k`` from a :class:`~jem.base.component.TimeAxis` is stamped
   ``start_date + (k + 1) * dt``.

Known difference, deliberately not copied: JCM drops the sub-day part of its
start date (it uses ``start_date.delta.days`` and ignores ``.seconds``), so a
run starting at 06:00 is labelled from midnight. This helper includes it. For
a start date at midnight -- every configuration JEM ships -- the two agree
exactly, since adding an exact 0.0 changes no bits.
"""

import numpy as np

from jem.base.component import TimeAxis

#: Nanoseconds in a day, as a float64 -- the exact factor JCM multiplies by.
NANOSECONDS_PER_DAY = np.timedelta64(1, "D") / np.timedelta64(1, "ns")

SECONDS_PER_DAY = 86400.0

#: Attributes JCM's CF pass puts on a datetime64 ``time`` axis
#: (``jcm.cf_metadata``). ``units`` is deliberately absent: xarray owns it via
#: the datetime encoding, and setting it here collides on write.
TIME_ATTRS: dict[str, str] = {
    "standard_name": "time",
    "axis": "T",
    "long_name": "time",
}


def time_coordinate(time: TimeAxis) -> tuple[np.ndarray, dict[str, str]]:
    """Return the ``time`` coordinate values and attributes for an output file.

    Parameters
    ----------
    time : jem.base.component.TimeAxis
        The coupler's description of the records being written: the run's
        start date, the coupling timestep and the coupled-step index of each
        record.

    Returns
    -------
    values : numpy.ndarray
        ``datetime64[ns]``, one per record, stamped at the end of the coupling
        step the record came from (see the module docstring).
    attrs : dict of str
        CF attributes for the coordinate.

    """
    start = time.start_date.delta
    start_days = float(np.asarray(start.days)) + float(
        np.asarray(start.seconds)
    ) / SECONDS_PER_DAY
    step_days = float(np.asarray(time.dt.days)) + float(
        np.asarray(time.dt.seconds)
    ) / SECONDS_PER_DAY

    steps = np.asarray(time.steps, dtype=np.float64)
    days = start_days + step_days * (steps + 1.0)
    values = (days * NANOSECONDS_PER_DAY).astype("datetime64[ns]")
    return values, dict(TIME_ATTRS)
