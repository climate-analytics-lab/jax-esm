"""Reductions of a run's diagnostics computed *inside* the coupled scan.

``Coupler.generate_trajectory_function(iterations, accumulate=(init, update))``
runs ``update`` on every step's diagnostics inside the ``lax.scan`` body and
returns only the accumulator, never the stacked per-step output. This module
packages the reductions a long run almost always wants -- a mean over each
bin of a fixed set of bins -- as such a pair:

- :func:`monthly_mean`, the twelve calendar months;
- :func:`windowed_mean`, ``n_windows`` windows of a fixed length, which is
  what a sub-seasonal forecast is scored on (pentads, weeks).

Both are :func:`_binned_mean` with a different step-to-bin rule, so there is
one running-sum-and-count implementation and one :meth:`BinnedMean.finalize`.

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

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jem.base.component import CouplingTime, Diagnostics

#: The accumulator a :class:`BinnedMean` carries: an ``(n_bins, ...)`` running
#: sum shaped like one step's diagnostics, and the ``(n_bins,)`` count of
#: steps that landed in each bin.
BinnedAccumulator = tuple[Any, jnp.ndarray]

#: Number of calendar months :func:`monthly_mean` bins into. The leading axis
#: of everything it makes :meth:`BinnedMean.finalize` return is January first.
MONTHS_PER_YEAR = 12

_SECONDS_PER_DAY = 86400

#: Month lengths, in days, of the fixed-length calendars a static day-of-year
#: to month table can be built for. ``jcm.date`` defines only ``365_day``
#: (and ``gregorian``, whose leap years make no such table possible), so
#: ``360_day`` is here because the table is the same kind of object, not
#: because a ``Coupler`` can be built with it today.
_MONTH_LENGTHS: dict[int, tuple[int, ...]] = {
    365: (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    360: (30,) * MONTHS_PER_YEAR,
}


def _month_of_day_table(days_per_year: float) -> jnp.ndarray:
    """Return the 0-based month of each 0-based day of a fixed-length year.

    This table is what makes month binning cheap and static inside a scan: a
    month is a lookup on the day of year, not calendar arithmetic, and no
    branch depends on which month it is -- so one compiled trajectory covers
    a whole year whatever its steps straddle.

    It comes back as a JAX array because the only thing that ever reads it is
    a traced index inside the scan body; building it with numpy and handing
    back a host array would leave every caller to convert it.
    """
    length = int(days_per_year)
    if length != days_per_year or length not in _MONTH_LENGTHS:
        raise NotImplementedError(
            f"Monthly means need a calendar whose year is a fixed whole number "
            f"of days with fixed month lengths, so that the day-of-year to "
            f"month table is a constant; this run's year is {days_per_year} "
            f"days. Supported: {sorted(_MONTH_LENGTHS)} days. A Gregorian "
            "calendar's leap years change the table from year to year, which "
            "a static table cannot express -- bin such a run on the host, by "
            "the datetime64 labels of `Coupler.to_xarray`."
        )
    return jnp.asarray(
        np.repeat(np.arange(MONTHS_PER_YEAR), _MONTH_LENGTHS[length]),
        dtype=jnp.int32,
    )


def _whole_seconds(duration: str | float, calendar: str, what: str) -> int:
    """Return ``duration`` as a whole positive number of seconds.

    The duration is parsed on the *coupler's* calendar, so ``"1 year"`` is as
    long as the model's year rather than as long as a Gregorian one, and is
    then rounded to whole seconds: everything downstream of here is integer
    arithmetic on seconds, because a float32 count of seconds since the start
    of a run stops being exact within a few decades of simulated time.
    """
    # Imported here rather than at module scope so that importing this module
    # does not pull in jax-gcm (and with it dinosaur and the whole
    # atmosphere); `jem.driver` imports it the same way and for the same
    # reason.
    from jcm.date import parse_duration_days

    days = float(parse_duration_days(duration, calendar))
    seconds = int(round(days * _SECONDS_PER_DAY))
    if seconds <= 0:
        raise ValueError(
            f"{what}={duration!r} is {seconds} s, which is not a positive "
            "duration."
        )
    return seconds


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
        coupled step's diagnostics into its bin.

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
            leading axis of length ``n_bins`` -- the twelve calendar months,
            January first, for :func:`monthly_mean`; the ``n_windows`` windows
            from the start of the run for :func:`windowed_mean`. A bin **no
            step fell in is NaN**, not zero: a run of two months has ten empty
            monthly bins, and zero would be a value that plotted and averaged
            as if it were data.

        """
        sums, counts = accumulator
        counts = jnp.asarray(counts)
        # The bin count is the accumulator's own leading axis rather than
        # anything captured when the pair was built, so `finalize` is correct
        # for whatever `init` made and stays a plain function of its argument.
        n_bins = counts.shape[0]
        empty = counts == 0
        # The division is taken against a count of 1 where there is no data
        # and the result thrown away by the `where`. Dividing by zero and
        # masking the NaN afterwards would give the same value but a NaN
        # gradient, which propagates back through the whole trajectory.
        safe_counts = jnp.where(empty, 1, counts)

        def mean(total: jnp.ndarray) -> jnp.ndarray:
            total = jnp.asarray(total)
            per_bin = (n_bins,) + (1,) * (total.ndim - 1)
            return jnp.where(
                empty.reshape(per_bin),
                jnp.nan,
                total / safe_counts.reshape(per_bin).astype(total.dtype),
            )

        return jax.tree_util.tree_map(mean, sums)


def _binned_mean(
    coupler: Any,
    bin_of_step: Callable[[jnp.ndarray], jnp.ndarray],
    n_bins: int,
    carry: Any = None,
) -> BinnedMean:
    """Return the ``(init, update)`` pair that means each step into its bin.

    This is the whole of the reduction; :func:`monthly_mean` and
    :func:`windowed_mean` differ only in the ``bin_of_step`` they hand it, so
    a change to how sums are kept, to the dtype they are kept in, or to what
    an empty bin means, happens once.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for. Only the structure, shapes
        and dtypes of one step's diagnostics are taken from it.
    bin_of_step : callable
        ``step -> int32 bin index in [0, n_bins)``, evaluated on the traced
        coupled step counter inside the scan. It must be total: an index
        outside the range would be clipped by ``.at[].add`` and silently
        counted in the nearest bin, so every caller reduces modulo
        ``n_bins``.
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
        return sums, jnp.zeros(n_bins, dtype=jnp.int32)

    def update(
        accumulator: BinnedAccumulator,
        diagnostics: dict[str, Diagnostics],
        time: CouplingTime,
    ) -> BinnedAccumulator:
        sums, counts = accumulator
        index = bin_of_step(time.step)
        new_sums = jax.tree_util.tree_map(
            lambda total, value: total.at[index].add(value), sums, diagnostics
        )
        return new_sums, counts.at[index].add(1)

    return BinnedMean(init=init, update=update)


def monthly_mean(coupler: Any, carry: Any = None) -> BinnedMean:
    """Build the in-scan accumulator of a coupled run's monthly means.

    Everything the reduction needs it takes from ``coupler``: the diagnostics
    a step produces (their structure, shapes and dtypes), the coupling
    timestep, the start date's offset into the year and the calendar's year
    length. The caller supplies nothing but the coupler.

    **Which month a step counts in.** A coupled step covers
    ``[start + k·dt, start + (k+1)·dt)`` and JEM labels the output record it
    produces with the **end** of that interval (``TimeAxis.datetimes``, JCM's
    convention). The bin follows the label: a step is counted in the month its
    *label* falls in, so
    ``monthly.finalize(accumulator)`` is exactly
    ``coupler.to_xarray(diagnostics).groupby("time.month").mean()`` of the same
    run, leaf for leaf. The one visible consequence is at a boundary: the
    daily step covering 31 January is labelled 1 February and counted in
    February. Binning by the start of the interval instead would be equally
    defensible, but then the accumulated mean and the written output would
    disagree about the same run, which is worse than either convention.

    **A run longer than a year wraps**, because the bin is the calendar month
    and not the month since the run started: a three-year run's January bin
    holds all three Januaries, which is a climatology and is what the fixed
    ``(12, ...)`` accumulator is for. :func:`windowed_mean` wraps the same
    way, at ``n_windows`` windows instead of at a year.

    **Sub-steps.** A component the workflow runs ``n > 1`` times per coupled
    step returns diagnostics with a leading sub-step axis of length ``n``, and
    that axis is accumulated as it is: the mean for that component comes back
    as ``(12, n, ...)``, the monthly mean of each sub-step slot -- a
    monthly-mean diurnal cycle for a component sub-cycling through the day.
    Average over axis 1 for the plain monthly mean. All ``n`` sub-steps are
    binned by the coupled step that contains them, which is the same month for
    every one of them because a coupled step never straddles a month boundary
    under this labelling. A nested :class:`~jem.base.coupler.Coupler` is
    treated the same way: its diagnostics arrive as a mapping of *its*
    components (with their own leading axis of inner steps) and are
    accumulated raw, in the structure a step produced them in, rather than
    flattened the way :meth:`~jem.base.coupler.Coupler.to_xarray` flattens
    them.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for.
    carry : CoupledCarry, optional
        A carry to take the diagnostics' shapes from; see
        :func:`_binned_mean`.

    Returns
    -------
    BinnedMean
        Whose ``finalize`` returns ``(12, ...)`` leaves, January first.

    Raises
    ------
    NotImplementedError
        If the run's calendar has no fixed day-of-year to month table --
        ``gregorian``, whose leap years change it from year to year.
    ValueError
        If the coupling timestep does not divide the year exactly. The month
        of a step is then not a function of ``step`` reduced modulo a whole
        number of steps per year, and the seconds-since-start arithmetic that
        would be needed instead overflows int32 within a human lifetime of
        simulated time.

    """
    month_of_day = _month_of_day_table(coupler.days_per_year)

    # Whole seconds throughout: `jdt.Timedelta` is integer-backed and the year
    # offset is a difference of two dates, so this arithmetic is exact, which
    # float32 seconds-since-start would not be after a few decades of a run.
    seconds_per_year = int(round(_SECONDS_PER_DAY * coupler.days_per_year))
    dt_seconds = int(round(coupler.dt_seconds))
    year_offset_seconds = int(round(coupler.year_offset_seconds))
    if dt_seconds <= 0 or seconds_per_year % dt_seconds:
        raise ValueError(
            f"A monthly mean needs the coupling timestep ({dt_seconds} s) to "
            f"divide the year ({seconds_per_year} s) exactly, so that the "
            "month of a step can be found from the step counter reduced modulo "
            "a whole number of steps per year."
        )
    steps_per_year = seconds_per_year // dt_seconds

    def month_index(step: jnp.ndarray) -> jnp.ndarray:
        """Return the 0-based month the step starting at ``step`` is counted in."""
        # `step + 1` is the end of the step, i.e. the instant its output
        # record is labelled with. Reducing modulo the steps in a year first
        # keeps the product inside int32 for a run of any length, and is what
        # makes a multi-year run's bins a climatology.
        step_in_year = jnp.mod(jnp.asarray(step, dtype=jnp.int32) + 1, steps_per_year)
        seconds_into_year = jnp.mod(
            year_offset_seconds + step_in_year * dt_seconds, seconds_per_year
        )
        return month_of_day[seconds_into_year // _SECONDS_PER_DAY]

    return _binned_mean(coupler, month_index, MONTHS_PER_YEAR, carry)


def windowed_mean(
    coupler: Any,
    window: str | float,
    *,
    n_windows: int | None = None,
    total_time: str | float | None = None,
    carry: Any = None,
) -> BinnedMean:
    """Build the in-scan accumulator of a run's means over fixed-length windows.

    The reduction a sub-seasonal forecast is scored on: the mean over each
    pentad, or each week, of a run, rather than over each calendar month::

        pentads = windowed_mean(coupler, "5 days", n_windows=73)   # a year
        weeks = windowed_mean(coupler, "7 days", total_time="1 year")

    **Which window a step counts in.** As in :func:`monthly_mean`, a step is
    binned by the label of the output record it produces -- the **end** of the
    interval it covers -- not by where the interval begins. Window ``w`` is
    therefore the steps whose labels fall in ``(w·window, (w+1)·window]``
    measured from the run's start date, so with daily coupling the first
    5-day window is the records labelled day 1 to day 5, which is what a
    forecast means by "the first pentad". In terms of the step counter that
    is ``step // steps_per_window``, since step ``k`` is labelled at
    ``(k+1)·dt``.

    That is the same rule :func:`monthly_mean` follows -- bin by the label --
    applied to bins the *run* defines instead of bins the calendar defines,
    and the boundaries close the other way as a result: a label falling
    exactly on a boundary ends the window before it (JEM labels every record
    at the end of its interval, and a window is one such interval), while the
    same label starts the calendar month after it (which is what
    ``groupby("time.month")`` does, and what a monthly mean has to agree
    with). It is one step of difference in each case and both are documented
    where they are; what neither does is bin by the *start* of the step.

    **A run longer than ``n_windows · window`` wraps**, exactly as
    :func:`monthly_mean` wraps at a year: window ``w`` then also collects
    windows ``w + n_windows``, ``w + 2·n_windows`` and so on, giving the
    composite of every *w*-th window of the run. That is the price of an
    accumulator whose size is fixed at trace time and does not grow with the
    run -- the whole point of reducing inside the scan. Size the accumulator
    to the run (pass ``total_time``, or ``n_windows`` counted for the run) if
    each window is meant to stand on its own.

    Sub-steps and nested couplers are accumulated exactly as
    :func:`monthly_mean` describes.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for.
    window : str or float
        Length of one window, as a ``jcm.date.parse_duration_days`` string
        (``"5 days"``, ``"1 month"``) or a number of days, parsed on the
        coupler's calendar. It must be a whole number of coupling steps: a
        window that ended part-way through a step would have to attribute
        that step to one side or the other, and there is no defensible
        choice.
    n_windows : int, optional
        How many windows the accumulator holds -- the length of the leading
        axis of everything ``finalize`` returns. Exactly one of this and
        ``total_time`` must be given.
    total_time : str or float, optional
        The length of the run, in the same forms as ``window``, from which
        ``n_windows`` is counted: enough windows to cover the run, the last
        one short if the run does not divide into whole windows (its mean is
        then over the steps that did fall in it, because every bin is divided
        by its own count).
    carry : CoupledCarry, optional
        A carry to take the diagnostics' shapes from; see
        :func:`_binned_mean`.

    Returns
    -------
    BinnedMean
        Whose ``finalize`` returns ``(n_windows, ...)`` leaves, the first
        window of the run first.

    Raises
    ------
    ValueError
        If neither or both of ``n_windows`` and ``total_time`` are given, if
        ``n_windows`` is not a positive integer, if either duration is not
        positive, or if ``window`` is not a whole number of coupling steps.

    """
    dt_seconds = int(round(coupler.dt_seconds))
    if dt_seconds <= 0:
        raise ValueError(
            f"The coupling timestep is {dt_seconds} s; a windowed mean needs a "
            "positive one to count steps per window."
        )
    window_seconds = _whole_seconds(window, coupler.calendar, "window")
    if window_seconds % dt_seconds:
        raise ValueError(
            f"window={window!r} is {window_seconds} s, which is not a whole "
            f"number of coupling steps of {dt_seconds} s. A window that ended "
            "part-way through a coupled step could only be filled by splitting "
            "that step between two windows, which the reduction does not do."
        )
    steps_per_window = window_seconds // dt_seconds

    if (n_windows is None) == (total_time is None):
        raise ValueError(
            "Give exactly one of n_windows and total_time: n_windows sets the "
            "accumulator's size directly, total_time counts it from the run "
            f"(got n_windows={n_windows!r}, total_time={total_time!r})."
        )
    if total_time is not None:
        total_seconds = _whole_seconds(total_time, coupler.calendar, "total_time")
        # Round *up*: a run that is not a whole number of windows ends inside
        # one, and that window has to exist to hold it. It is divided by its
        # own count like every other, so a short final window is the mean of
        # what fell in it rather than a mean diluted by missing steps.
        bins = -(-total_seconds // window_seconds)
    elif isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 1:
        # `bool` is an `int`, and `n_windows=True` would silently build a
        # one-window accumulator -- i.e. a mean of the whole run.
        raise ValueError(f"n_windows must be a positive integer; got {n_windows!r}.")
    else:
        bins = n_windows

    def window_index(step: jnp.ndarray) -> jnp.ndarray:
        """Return the 0-based window the step starting at ``step`` counts in."""
        # `step` rather than `step + 1` because the label of step k is at
        # (k+1)·dt and a window is closed at its end: the step labelled
        # exactly `window` is the last of window 0, not the first of window 1.
        # The modulo is what wraps a run longer than the accumulator and, with
        # it, keeps the index inside the accumulator whatever the run's
        # length.
        return jnp.mod(jnp.asarray(step, dtype=jnp.int32) // steps_per_window, bins)

    return _binned_mean(coupler, window_index, bins, carry)
