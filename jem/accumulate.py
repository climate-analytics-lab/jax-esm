"""Reductions of a run's diagnostics computed *inside* the coupled scan.

``Coupler.generate_trajectory_function(iterations, accumulate=(init, update))``
runs ``update`` on every step's diagnostics inside the ``lax.scan`` body and
returns only the accumulator, never the stacked per-step output. This module
packages the reduction a long run almost always wants -- a monthly mean --
as such a pair.

Why it is in the scan at all. A twelve-month run reduced on the host must
hold every step's diagnostics until the chunk ends, which for an atmosphere
is the largest array in the run by a wide margin; and a run chunked *by
month* must compile a 28-, a 30- and a 31-day trajectory, because
``iterations`` is static. Accumulating inside the scan does neither: one
compiled trajectory of whatever length suits the machine, and a fixed-size
``(12, ...)`` accumulator that a chunked run threads from call to call.

Both are ordinary JAX: the accumulator is a pytree in the scan carry, so
``jax.grad`` of a monthly mean with respect to a component parameter flows
through the reduction exactly as it flows through the trajectory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jem.base.component import CouplingTime, Diagnostics

#: The accumulator a :class:`MonthlyMean` carries: a ``(12, ...)`` running
#: sum shaped like one step's diagnostics, and the ``(12,)`` count of steps
#: that landed in each month.
MonthlyAccumulator = tuple[Any, jnp.ndarray]

#: Number of calendar months the accumulator bins into. The leading axis of
#: everything :meth:`MonthlyMean.finalize` returns is January first.
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


def _month_of_day_table(days_per_year: float) -> np.ndarray:
    """Return the 0-based month of each 0-based day of a fixed-length year.

    This table is what makes month binning cheap and static inside a scan: a
    month is a lookup on the day of year, not calendar arithmetic, and no
    branch depends on which month it is -- so one compiled trajectory covers
    a whole year whatever its steps straddle.
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
    return np.repeat(np.arange(MONTHS_PER_YEAR), _MONTH_LENGTHS[length]).astype(
        np.int32
    )


class MonthlyMean(NamedTuple):
    """The ``(init, update)`` pair for a monthly mean, and its ``finalize``.

    It *is* the pair ``generate_trajectory_function(accumulate=...)`` takes --
    a two-field :class:`typing.NamedTuple`, so it both unpacks as
    ``init, update`` and carries the :meth:`finalize` that turns the
    accumulator into means::

        monthly = jem.accumulate.monthly_mean(coupler)
        trajectory = coupler.generate_trajectory_function(365, accumulate=monthly)
        carry, accumulator = trajectory(coupler.initialize())
        means = monthly.finalize(accumulator)

    Build one with :func:`monthly_mean`, which is what knows the coupler's
    clock.

    Attributes
    ----------
    init : callable
        ``() -> MonthlyAccumulator``; the zeroed sums and counts.
    update : callable
        ``(accumulator, diagnostics, time) -> MonthlyAccumulator``; adds one
        coupled step's diagnostics into its month's bin.

    """

    init: Callable[[], MonthlyAccumulator]
    update: Callable[
        [MonthlyAccumulator, dict[str, Diagnostics], CouplingTime],
        MonthlyAccumulator,
    ]

    def finalize(self, accumulator: MonthlyAccumulator) -> Any:
        """Return the monthly means: the running sums divided by their counts.

        Parameters
        ----------
        accumulator : MonthlyAccumulator
            What a trajectory built with this pair returned.

        Returns
        -------
        pytree
            The structure of one coupled step's diagnostics, every leaf with
            a leading axis of length 12 -- January first, whatever month the
            run started in. A month **no step fell in is NaN**, not zero: a
            run of two months has ten empty bins, and zero would be a value
            that plotted and averaged as if it were data.

        """
        sums, counts = accumulator
        counts = jnp.asarray(counts)
        empty = counts == 0
        # The division is taken against a count of 1 where there is no data
        # and the result thrown away by the `where`. Dividing by zero and
        # masking the NaN afterwards would give the same value but a NaN
        # gradient, which propagates back through the whole trajectory.
        safe_counts = jnp.where(empty, 1, counts)

        def mean(total: jnp.ndarray) -> jnp.ndarray:
            total = jnp.asarray(total)
            per_month = (MONTHS_PER_YEAR,) + (1,) * (total.ndim - 1)
            return jnp.where(
                empty.reshape(per_month),
                jnp.nan,
                total / safe_counts.reshape(per_month).astype(total.dtype),
            )

        return jax.tree_util.tree_map(mean, sums)


def monthly_mean(coupler: Any, carry: Any = None) -> MonthlyMean:
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
        A carry to take the diagnostics' shapes from. The shapes are obtained
        with ``jax.eval_shape`` of one coupled step, so the step is never run
        and nothing is computed; this argument only saves a driver that
        already has a carry the cost of ``coupler.initialize()``.

    Returns
    -------
    MonthlyMean

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
    month_of_day_array = jnp.asarray(month_of_day)

    def month_index(step: jnp.ndarray) -> jnp.ndarray:
        """Return the 0-based month the step starting at ``step`` is counted in."""
        # `step + 1` is the end of the step, i.e. the instant its output
        # record is labelled with. Reducing modulo the steps in a year first
        # keeps the product inside int32 for a run of any length.
        step_in_year = jnp.mod(jnp.asarray(step, dtype=jnp.int32) + 1, steps_per_year)
        seconds_into_year = jnp.mod(
            year_offset_seconds + step_in_year * dt_seconds, seconds_per_year
        )
        return month_of_day_array[seconds_into_year // _SECONDS_PER_DAY]

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

    def init() -> MonthlyAccumulator:
        sums = jax.tree_util.tree_map(
            lambda leaf: jnp.zeros(
                (MONTHS_PER_YEAR, *leaf.shape), accumulator_dtype(leaf.dtype)
            ),
            diagnostics_shapes,
        )
        return sums, jnp.zeros(MONTHS_PER_YEAR, dtype=jnp.int32)

    def update(
        accumulator: MonthlyAccumulator,
        diagnostics: dict[str, Diagnostics],
        time: CouplingTime,
    ) -> MonthlyAccumulator:
        sums, counts = accumulator
        month = month_index(time.step)
        new_sums = jax.tree_util.tree_map(
            lambda total, value: total.at[month].add(value), sums, diagnostics
        )
        return new_sums, counts.at[month].add(1)

    return MonthlyMean(init=init, update=update)
