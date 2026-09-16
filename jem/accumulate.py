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
#: sum shaped like one step's diagnostics, and the count of records that
#: landed in each bin. The counts are a single ``(n_bins,)`` array when every
#: component of the model produces one record per coupled step, and one array
#: per component -- shaped ``(n_bins, *the component's sub-step axes)`` --
#: when any of them produces more, because components that record at
#: different rates then fill different bins as a coupled step is folded in
#: (see :func:`_record_axes`).
BinnedAccumulator = tuple[Any, Any]

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


def _record_axes(coupler: Any) -> dict[str, Any]:
    """Return the sub-step axes each component's diagnostics carry, by name.

    A component the workflow runs ``n > 1`` times per coupled step has its
    ``n`` diagnostics stacked on a leading axis of length ``n``, and a nested
    :class:`~jem.base.coupler.Coupler` that runs ``r`` of its own coupled
    steps per outer step stacks its inner components' diagnostics on a leading
    axis of length ``r`` -- so a step's diagnostics for such a component are
    not one record but several, each covering its own sub-interval and
    labelled with the end of it.

    This is the map of that: ``{name: axes}``, where ``axes`` is the tuple of
    leading axis lengths one coupled step puts in front of the diagnostic's
    own shape (``()`` for a component that records once per coupled step),
    or -- for a nested coupler -- a ``{inner name: axes}`` mapping of its own
    components, since those may record at different rates from each other.
    The axes are in the order the leading axes appear, which is the order the
    records were produced in, so flattening them row-major gives the records
    of one coupled step in time order. That is the whole reason this map
    exists: :func:`_binned_mean` needs each record's own place in the coupled
    step to bin it by its own label rather than by the step's.

    The nested coupler is recognised by duck-typing rather than by an
    ``isinstance`` check, because importing :mod:`jem.base.coupler` here would
    pull in jax-gcm (and with it the whole atmosphere) just to import this
    module; :func:`_whole_seconds` defers its import for the same reason.
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
    if remainder or seconds < 1:
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


def _binned_means(sums: Any, counts: Any) -> Any:
    """Return the running sums divided by the counts of their own bins.

    Recurses on ``counts`` rather than on ``sums``: the counts mirror the
    *components* of the model (one array per component when they record at
    different rates, one shared array when they do not), while a component's
    sums are an opaque pytree of its diagnostics, which may itself be a dict.
    """
    if isinstance(counts, dict):
        return {name: _binned_means(sums[name], node) for name, node in counts.items()}
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
        per record (see :func:`_binned_mean`).

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
            from the start of the run for :func:`windowed_mean`. A component
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
        return _binned_means(sums, counts)


def _binned_mean(
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
    ``n``, and each is labelled with the end of its own sub-interval, so they
    need not all fall in the same bin. The 24 hourly records of the daily step
    that covers 31 January are labelled 01:00 on the 31st through 00:00 on 1
    February -- 23 of them in January and one in February. Every record is
    therefore binned by its own label, from the coupler's own sub-step clock
    (:meth:`~jem.base.coupler.Coupler.coupling_time_at_substep`), which is the
    same instant :meth:`~jem.base.coupler.Coupler.to_xarray` labels that
    record with. Binning them all by the coupled step instead would put a
    whole day of hourly records in the month the day *ended* in, and the
    accumulated mean would disagree with a ``groupby`` of the written output
    at every month boundary.

    The sub-step axis is **kept**, not folded: a component recording ``n``
    times per coupled step accumulates into ``(n_bins, n, ...)``, bin ``b``
    slot ``j`` holding the records of call ``j`` that fell in bin ``b`` -- a
    monthly-mean diurnal cycle for a component sub-cycling through the day.
    Folding it would be a mean of means, and it is one ``sum`` on the host
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

    dt_seconds = int(round(coupler.dt_seconds))
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
                # coupled step `s` is sub-step `s * records + k`, and the
                # record it produces is labelled at the end of that sub-step.
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
    that axis is kept: the mean for that component comes back as
    ``(12, n, ...)``, the monthly mean of each sub-step slot -- a
    monthly-mean diurnal cycle for a component sub-cycling through the day.
    Each of the ``n`` records is binned by **its own** label, not by the
    coupled step's, so the 23 hourly records of 31 January count in January
    and the one labelled 1 February 00:00 counts in February, exactly as
    ``groupby("time.month")`` of the written output does. Fold the axis with
    the counts to get the plain monthly mean::

        sums, counts = accumulator                      # counts["atm"]: (12, n)
        means = monthly.finalize(accumulator)["atm"]    # (12, n, ...)
        per_month = jnp.nansum(means * counts[..., None], 1) / counts.sum(1)

    -- a straight mean over axis 1 is only the same number when every slot
    holds the same number of records, which is what fails at the month
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

    def month_index(record: jnp.ndarray, record_seconds: int) -> jnp.ndarray:
        """Return the 0-based month a record of ``record_seconds`` counts in.

        ``record`` counts records of that length from the start of the run:
        coupled steps for a component that records once per coupled step, and
        sub-steps for one that records more often. The rule is the same
        either way -- the month the record's own label falls in.
        """
        # A record's label is the END of the interval it covers, which is
        # `record + 1` records after the start of the run. Reducing modulo the
        # records in a year first keeps the product inside int32 for a run of
        # any length, and is what makes a multi-year run's bins a climatology.
        # The year is a whole number of records because it is a whole number
        # of coupled steps and a coupled step is a whole number of records.
        records_per_year = steps_per_year * (dt_seconds // record_seconds)
        in_year = jnp.mod(jnp.asarray(record, dtype=jnp.int32) + 1, records_per_year)
        seconds_into_year = jnp.mod(
            year_offset_seconds + in_year * record_seconds, seconds_per_year
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

    **Which window a record counts in.** As in :func:`monthly_mean`, a record
    is binned by its own label -- the **end** of the interval it covers -- not
    by where that interval begins. Window ``w`` is therefore the records whose
    labels fall in ``(w·window, (w+1)·window]`` measured from the run's start
    date, so with daily coupling the first 5-day window is the records
    labelled day 1 to day 5, which is what a forecast means by "the first
    pentad". In terms of the counter of records of length ``r`` that is
    ``record // (window/r)``, since record ``k`` is labelled at ``(k+1)·r``.

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
    :func:`monthly_mean` describes: a component recording ``n`` times per
    coupled step keeps that axis (``(n_windows, n, ...)``) and each of its
    records is binned by its own label, so a window boundary falls between two
    sub-steps of a coupled step wherever the labels say it does.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for.
    window : str or float
        Length of one window, as a ``jcm.date.parse_duration_days`` string
        (``"5 days"``, ``"2 days"``, ``"12 hours"``) or a number of days,
        parsed on the coupler's calendar. Note that ``"1 month"`` is a
        *calendar-averaged* month -- 365/12 days, not a whole number of daily
        steps -- so monthly bins come from :func:`monthly_mean`, which knows
        the real month lengths, and not from here. It must be a whole number of coupling steps: a
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

    def window_index(record: jnp.ndarray, record_seconds: int) -> jnp.ndarray:
        """Return the 0-based window a record of ``record_seconds`` counts in.

        ``record`` counts records of that length from the start of the run,
        so this is the coupled step for a component that records once per
        coupled step and the sub-step for one that records more often; a
        window holds a whole number of either, since the window is a whole
        number of coupled steps and a coupled step of records.
        """
        # `record` rather than `record + 1` because the label of record k is
        # at (k+1)·record_seconds and a window is closed at its end: the
        # record labelled exactly `window` is the last of window 0, not the
        # first of window 1. The modulo is what wraps a run longer than the
        # accumulator and, with it, keeps the index inside the accumulator
        # whatever the run's length.
        records_per_window = window_seconds // record_seconds
        return jnp.mod(
            jnp.asarray(record, dtype=jnp.int32) // records_per_window, bins
        )

    return _binned_mean(coupler, window_index, bins, carry)
