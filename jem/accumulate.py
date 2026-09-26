"""Reductions of a run's diagnostics computed *inside* the coupled scan.

``Coupler.generate_trajectory_function(iterations, accumulate=(init, update))``
runs ``update`` on every step's diagnostics inside the ``lax.scan`` body and
returns only the accumulator, never the stacked per-step output. This module
packages the reductions a long run almost always wants -- a mean over each
bin of a fixed set of bins -- as such a pair:

- :func:`monthly_mean`, the Gregorian calendar months -- twelve bins that
  composite a multi-year run into a climatology, or (``total_time=`` /
  ``n_months=``) one bin per month the run passes through. Each record is
  binned by ``jcm.date.gregorian_ymd_from_days`` of its own interval midpoint
  -- the real calendar, so a February is 28 or 29 days exactly as the run's
  actual dates say and no fixed-length month table or start-of-year phase is
  needed;
- :func:`windowed_mean`, ``n_windows`` windows measured from the run's own
  start date -- of one fixed length, which is what a sub-seasonal forecast is
  scored on (pentads, weeks), or of a repeating *pattern* of lengths. Bins are
  plain record counts from the start of the run, via :func:`_variable_window_rule`.

Both are :func:`_build_binned_mean` with a different step-to-bin rule, so
there is one running-sum-and-count implementation and one
:meth:`BinnedMean.finalize`.

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
worked calibration example in ``docs/source/design/running.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, NamedTuple

import math

import jax
import jax.numpy as jnp
import jax_datetime as jdt
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

#: Number of calendar months in a year, and the number of bins
#: :func:`monthly_mean` accumulates into unless it is asked for one bin per
#: month of the run (``total_time=`` / ``n_months=``). In the twelve-bin form
#: the leading axis of everything :meth:`BinnedMean.finalize` returns is
#: January first; in the sequential form it starts with the month the run
#: starts in.
MONTHS_PER_YEAR = 12

_SECONDS_PER_DAY = 86400


def _variable_window_rule(
    boundaries_seconds: np.ndarray,
    offset_seconds: int,
    inclusive: Literal["left", "right"],
) -> Callable[[jnp.ndarray, int, Any], jnp.ndarray]:
    """Return the ``bin_of_record`` rule :func:`windowed_mean` bins with.

    Windows are laid end to end, cycling for as long as the run lasts, and a
    record belongs to the window its own label falls in -- entirely by
    record count and coupling timestep, with no reference to any calendar.

    Parameters
    ----------
    boundaries_seconds : numpy.ndarray
        The **ends** of the windows, in whole seconds from the start of the
        run: the cumulative sum of the window lengths, strictly increasing,
        one entry per window. The last entry is where the windows end and the
        period they repeat with, which is what makes a run longer than the
        accumulator wrap.
    offset_seconds : int
        Always 0 for :func:`windowed_mean`, which measures every window from
        the run's own start with no phase; kept as a parameter because the
        arithmetic below is exact for any offset, not because a caller needs
        one today.
    inclusive : {"left", "right"}
        Which end of a window is included in it, in the sense of
        :func:`pandas.date_range`'s ``inclusive``. A record labelled exactly
        on a boundary between two windows belongs to the window that includes
        that end.

        Take 5-day windows, counting record ``k`` as ending on day ``k + 1``
        (this rule's own ordinal convention, independent of how a record's
        output timestamp is labelled). ``inclusive="left"`` makes the first
        window the records in ``[day 0, day 5)`` and the second
        ``[day 5, day 10)``, so day 5 opens the second window.
        ``inclusive="right"`` -- what :func:`windowed_mean` uses, so that the
        first 5-day window reads as "the first pentad" -- makes the first
        window ``(day 0, day 5]`` and the second ``(day 5, day 10]``, so the
        first 5-day window is records 1 to 5 and day 5 closes it rather than
        opening the next one.

    Returns
    -------
    callable
        ``(record, record_seconds, record_time) -> int32 bin index``, total
        by construction (see :func:`_build_binned_mean`). ``record_time`` is
        accepted and ignored -- this rule bins by record count, not by date.

    Notes
    -----
    **Why the boundaries are converted to record counts.** Record ``k`` of
    length ``r`` is labelled at ``(k + 1)·r`` seconds from the start of the
    run, so the obvious rule -- multiply the record counter by ``r`` and look
    the result up in a table of seconds -- costs one multiplication whose
    product grows with the run. JAX indices are int32 by default, and a
    product of seconds passes 2^31 after 68 years of simulated time, at which
    point the bins would silently wrap to nonsense. Reducing the record
    counter modulo the records in one period *before* multiplying bounds the
    product by the period rather than by the run, which the previous
    fixed-window arithmetic did too; converting the boundaries to record
    counts on the host, in int64, removes the multiplication from the traced
    code altogether and so bounds nothing by int32 but the record counter
    itself. The conversion is exact because every boundary is a whole number
    of coupling steps and a record is a whole division of one.

    """
    if inclusive not in ("left", "right"):
        # `Literal` is a promise to the type checker, not a runtime check, so a
        # misspelling ("rigth") would otherwise be read as "left" by the
        # comparison below and silently shift every bin boundary by a record.
        raise ValueError(
            f'inclusive must be "left" or "right"; got {inclusive!r}.'
        )
    boundaries = np.asarray(boundaries_seconds, dtype=np.int64)
    period = int(boundaries[-1])
    # The two conventions differ by one second of this rule's own ordinal
    # count, which treats record k as ending at day k + 1 regardless of how
    # its output timestamp is labelled. `bin_of_record` counts how many
    # boundaries lie at or before `label + shift_seconds`. With 5-day bins and
    # daily records (record k counted as day k + 1):
    #
    #   inclusive="right": shift -1 s. A label of day 5 is looked up a second
    #     before the first boundary, so no boundary precedes it -> bin 0; it
    #     is the LAST record of (day 0, day 5]. Day 6 -> bin 1.
    #   inclusive="left": no shift. A label of day 5 is looked up at the
    #     boundary itself, which now counts -> bin 1; it is the FIRST record
    #     of [day 5, day 10).
    shift_seconds = -1 if inclusive == "right" else 0

    def bin_of_record(
        record: jnp.ndarray, record_seconds: int, record_time: Any = None
    ) -> jnp.ndarray:
        """Return the 0-based bin a record of ``record_seconds`` counts in.

        ``record_time`` is accepted and ignored, for the common call site in
        :func:`_build_binned_mean`; this rule bins by record count alone.

        ``record`` counts records of that length from the start of the run.
        With daily records, 5-day bins and ``inclusive="right"``: record 0 is
        labelled day 1 and lands in bin 0; record 4 (day 5) is the last of
        bin 0; record 5 (day 6) is the first of bin 1; and in a 73-bin
        accumulator record 365 -- labelled day 366, one year on -- wraps to
        bin 0 again. Under ``inclusive="left"`` record 4 (day 5) is instead
        the first of bin 1.
        """
        records_per_period, remainder = divmod(period, record_seconds)
        # An invariant of the callers, not a user error: every builder checks
        # that the bins' period is a whole number of coupling steps, and the
        # coupler refuses a workflow whose sub-timestep is not a whole
        # division of one (`Coupler._element_timestep`, re-checked in
        # `_record_seconds`), so this division is exact. It is asserted rather
        # than assumed because a silent rounding here would drift every bin
        # boundary by a fraction of a record.
        assert remainder == 0, (
            f"a period of {period} s is not a whole number of "
            f"{record_seconds} s records"
        )
        del record_time
        # Where the boundaries sit on this component's record grid. The run's
        # offset into the pattern need not be a whole number of records, so it
        # is split into whole records (`shifted`, folded into the counter) and
        # a remainder (`phase`, folded into the boundaries); the ceiling is
        # then the first record whose label reaches the boundary.
        shifted, phase = divmod(offset_seconds + shift_seconds, record_seconds)
        # The last entry lands exactly on `records_per_period`: the period is
        # a whole number of records and `phase` is less than one, so the
        # ceiling cannot overshoot it, and every wrapped record has a bin.
        in_records = _ceil_div(boundaries - phase, record_seconds)
        boundary_records = jnp.asarray(in_records, dtype=jnp.int32)
        # `record + 1` because this rule counts record k as ending at day
        # k + 1 (its own ordinal convention). The modulo is what wraps a run longer than the pattern
        # and, with it, keeps the index inside the accumulator whatever the
        # run's length; it is also what keeps the arithmetic in int32.
        label = jnp.mod(
            jnp.asarray(record, dtype=jnp.int32) + (1 + shifted), records_per_period
        )
        return jnp.searchsorted(boundary_records, label, side="right").astype(
            jnp.int32
        )

    return bin_of_record


def _ceil_div(numerator: Any, denominator: int) -> Any:
    """Return ``ceil(numerator / denominator)`` in integer arithmetic.

    Python's ``//`` rounds towards negative infinity, so negating the
    numerator, floor-dividing, and negating the result rounds *up* instead:
    ``ceil(7 / 2) == -((-7) // 2) == 4``. It stays exact for the int64 arrays
    and Python ints this module works in, where ``math.ceil(a / b)`` would go
    through a float and lose precision past 2**53.
    """
    return -((-numerator) // denominator)


def _duration_to_seconds(duration: str | float, what: str) -> int:
    """Return ``duration`` as a whole positive number of seconds.

    ``duration`` must be a fixed length ("5 days", "12 hours"; jax_datetime
    has no calendar-dependent duration to parse a 'year' or 'month' against)
    and a whole number of seconds: everything downstream of here is integer
    arithmetic on seconds, because a float32 count of seconds since the start
    of a run stops being exact within a few decades of simulated time, and a
    fractional second is refused rather than rounded away.
    """
    # Imported here rather than at module scope so that importing this module
    # does not pull in jax-gcm (and with it dinosaur and the whole
    # atmosphere); `jem.driver` imports it the same way and for the same
    # reason.
    from jcm.date import parse_duration_days

    days = float(parse_duration_days(duration))
    seconds = _exact_seconds(days * _SECONDS_PER_DAY, f"{what}={duration!r}")
    if seconds <= 0:
        raise ValueError(
            f"{what}={duration!r} is {seconds} s, which is not a positive "
            "duration."
        )
    return seconds


def _exact_seconds(value: float, what: str) -> int:
    """Return ``value`` as an ``int``, refusing a fractional second.

    Everything in this module is integer arithmetic on seconds, and the
    coupler only ever builds whole-second clocks (``Coupler._element_timestep``
    refuses a sub-timestep that is not), so a value that is not a whole
    number of seconds is a broken invariant or a duration the bins cannot
    represent. Either way it is refused rather than rounded: rounding would
    silently move every bin boundary.

    The comparison allows a float-rounding margin, because a duration given
    in days reaches here through a float multiplication (``11 / 86400`` days
    comes back as ``10.999999999999998`` s). A microsecond, plus a relative
    part for durations of centuries, is far below anything a clock or a
    duration can express and far above that rounding, so a genuine fraction
    of a second is still refused.
    """
    seconds = round(value)
    if not math.isclose(value, seconds, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            f"{what} is {value!r} s, which is not a whole number of seconds; "
            "the bins are laid out in whole seconds."
        )
    return int(seconds)


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
    exists: :func:`_build_binned_mean` needs each record's own place in the coupled
    step to bin it by its own label rather than by the step's.

    The nested coupler is recognised by duck-typing rather than by an
    ``isinstance`` check, because importing :mod:`jem.base.coupler` here would
    pull in jax-gcm (and with it the whole atmosphere) just to import this
    module; :func:`_duration_to_seconds` defers its import for the same reason.
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
    if remainder != 0 or seconds < 1:
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


def _divide_sums_by_counts(sums: Any, counts: Any) -> Any:
    """Return the running sums divided by the counts of their own bins.

    Recurses on ``counts`` rather than on ``sums``: the counts mirror the
    *components* of the model (one array per component when they record at
    different rates, one shared array when they do not), while a component's
    sums are an opaque pytree of its diagnostics, which may itself be a dict.
    """
    if isinstance(counts, dict):
        return {name: _divide_sums_by_counts(sums[name], node) for name, node in counts.items()}
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


def fold_records(means: Any, counts: Any) -> Any:
    """Return one component's binned means with its sub-step axes folded away.

    :meth:`BinnedMean.finalize` **keeps** the axes a component that records
    more than once per coupled step accumulates into -- bin ``b``, slot ``j``
    is the mean of the records of call ``j`` that fell in ``b`` -- because
    that is a monthly-mean diurnal cycle, which folding destroys and which
    cannot be recovered afterwards. This is the fold, for when the plain mean
    over every record of a bin is what was wanted::

        monthly = monthly_mean(coupler)
        ...
        sums, counts = accumulator
        means = monthly.finalize(accumulator)
        per_month = fold_records(means["atm"], counts["atm"])

    It is a *weighted* mean, by each slot's own count, which is what makes it
    equal a ``groupby`` of the written output exactly (on the bins' own
    terms): a straight mean over the slots is the same number only when every
    slot holds the same number of records, and what breaks that is exactly
    the bin boundary this binning exists to get right (23 of a day's hourly
    records in January, one in February).

    Parameters
    ----------
    means : pytree
        One component's entry in what :meth:`BinnedMean.finalize` returned:
        leaves shaped ``(n_bins, *sub-step axes, ...)``, empty bins NaN.
    counts : array
        That component's entry in the accumulator's counts, shaped
        ``(n_bins, *sub-step axes)``. A component that records once per
        coupled step has no sub-step axes, and folding it then returns its
        means unchanged -- exactly, not to within a float32 rounding -- so a
        caller need not know which kind of component it has.

    Returns
    -------
    pytree
        ``means`` with the sub-step axes summed away: leaves shaped
        ``(n_bins, ...)``, a bin no record fell in still NaN.

    """
    counts = jnp.asarray(counts)
    if counts.ndim == 1:
        # Nothing to fold: this component records once per coupled step, so a
        # bin already holds one mean over all of its records. Returning the
        # means untouched rather than multiplying by the count and dividing by
        # it again keeps this exactly the identity -- that round trip moves
        # the last bit of a float32 mean -- and keeps NaN for an empty bin.
        return means
    record_axes = tuple(range(1, counts.ndim))
    per_bin = jnp.sum(counts, axis=record_axes)

    def fold(leaf: jnp.ndarray) -> jnp.ndarray:
        leaf = jnp.asarray(leaf)
        # The counts are `(n_bins, *sub-step axes)` and a leaf is those axes
        # followed by the diagnostic's own shape, so the trailing axes are
        # what has to be broadcast over.
        weights = counts.reshape(counts.shape + (1,) * (leaf.ndim - counts.ndim))
        # An empty slot's mean is NaN, so its weight is applied by selection
        # rather than by multiplication: 0 * NaN is NaN, and one empty slot
        # would otherwise make the whole bin NaN (and its gradient NaN too).
        total = jnp.sum(jnp.where(weights == 0, 0.0, leaf) * weights, axis=record_axes)
        divisor = per_bin.reshape(per_bin.shape + (1,) * (total.ndim - per_bin.ndim))
        return jnp.where(
            divisor == 0,
            jnp.nan,
            total / jnp.where(divisor == 0, 1, divisor).astype(total.dtype),
        )

    return jax.tree_util.tree_map(fold, means)


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
        per record (see :func:`_build_binned_mean`).

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
            leading axis of length ``n_bins`` -- for :func:`monthly_mean`,
            either the twelve calendar months, January first, or the
            ``n_months`` months the run passes through, its first month first;
            for :func:`windowed_mean`, the ``n_windows`` windows from the
            start of the run. A component
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
        return _divide_sums_by_counts(sums, counts)


def _build_binned_mean(
    coupler: Any,
    bin_of_record: Callable[[jnp.ndarray, int, Any], jnp.ndarray],
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
    ``n``, and each is labelled at its own sub-interval's midpoint, so they
    need not all fall in the same bin. Every record is therefore binned by its
    own label, from the coupler's own sub-step clock
    (:meth:`~jem.base.coupler.Coupler.coupling_time_at_substep`), which is the
    same instant :meth:`~jem.base.coupler.Coupler.to_xarray` labels that
    record with -- so binning agrees with a ``groupby`` of the written output
    exactly, with no residual month-boundary disagreement to document.

    The sub-step axis is **kept**, not folded: a component recording ``n``
    times per coupled step accumulates into ``(n_bins, n, ...)``, bin ``b``
    slot ``j`` holding the records of call ``j`` that fell in bin ``b`` -- a
    monthly-mean diurnal cycle for a component sub-cycling through the day.
    Folding it would be a mean of means, and it is one :func:`fold_records`
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
        ``(record, record_seconds, record_time) -> int32 bin index in
        [0, n_bins)``, evaluated inside the scan on the traced record counter
        and the record's own :class:`~jax_datetime.Datetime` -- the coupled
        step counter/coupling timestep/time for a component recording once
        per coupled step, the sub-step counter/sub-timestep/time for one
        recording more often. :func:`windowed_mean`'s rule
        (:func:`_variable_window_rule`) bins by ``record``/``record_seconds``
        alone and ignores ``record_time``; :func:`monthly_mean` does the
        reverse, binning by the Gregorian month of ``record_time`` alone. It
        must be total: an index outside the range would be clipped by
        ``.at[].add`` and silently counted in the nearest bin.
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

    dt_seconds = _exact_seconds(coupler.dt_seconds, "the coupling timestep")
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
            index = bin_of_record(time.step, dt_seconds, time.time)
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
                # record it produces is labelled at that sub-step's own clock.
                sub_times = [
                    coupler.coupling_time_at_substep(time.step, time.time, call, records)
                    for call in range(records)
                ]
                substeps = jnp.stack([sub.step for sub in sub_times])
                record_times = jax.tree_util.tree_map(
                    lambda *leaves: jnp.stack(leaves), *(sub.time for sub in sub_times)
                )
                bins_by_rate[records] = bin_of_record(
                    substeps, record_seconds[records], record_times
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
                index = bin_of_record(time.step, dt_seconds, time.time)
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


def _record_month(
    record_time: jdt.Datetime, record_seconds: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the Gregorian ``(year, month)`` of a record's own midpoint.

    A record is labelled at the midpoint of the interval it covers
    (``TimeAxis.datetimes``, JCM's convention), so binning by the same
    instant is what makes ``monthly.finalize(accumulator)`` agree with
    ``coupler.to_xarray(diagnostics).groupby("time.month")`` exactly, leaf for
    leaf -- there is one calendar (jax_datetime's proleptic Gregorian), so
    there is no model-calendar-vs-label reconciliation left to do.
    ``record_seconds // 2`` is the same floor division jcm's own output
    labels use (``jcm.predictions.output_time_labels``); only the midpoint's
    *day* matters here, so the within-day remainder is never needed.
    """
    from jcm.date import gregorian_ymd_from_days

    midpoint = record_time + jdt.to_timedelta(record_seconds // 2, "second")
    year, month, _ = gregorian_ymd_from_days(midpoint.delta.days)
    return year, month


def _months_covering_gregorian(
    start_date: jdt.Datetime, dt_seconds: int, total_seconds: int
) -> int:
    """Return how many distinct calendar months a run of ``total_seconds`` touches.

    Host-side and exact, from the real Gregorian calendar: unlike a
    fixed-length month table, ``total_seconds`` need not be a whole number of
    months (or of years) for this to make sense. The count runs from the
    start date's own month to the last record's midpoint's, inclusive.
    """
    from jcm.date import gregorian_ymd_from_days

    if total_seconds % dt_seconds:
        raise ValueError(
            f"total_time is {total_seconds} s, which is not a whole number "
            f"of coupling steps of {dt_seconds} s."
        )
    n_records = total_seconds // dt_seconds
    last_midpoint_seconds = (n_records - 1) * dt_seconds + dt_seconds // 2
    last_midpoint = start_date + jdt.to_timedelta(
        int(last_midpoint_seconds), "second"
    )
    start_year, start_month, _ = gregorian_ymd_from_days(
        np.asarray(start_date.delta.days)
    )
    end_year, end_month, _ = gregorian_ymd_from_days(
        np.asarray(last_midpoint.delta.days)
    )
    return (
        int(end_year - start_year) * MONTHS_PER_YEAR
        + int(end_month - start_month)
        + 1
    )


def monthly_mean(
    coupler: Any,
    carry: Any = None,
    *,
    n_months: int | None = None,
    total_time: str | float | None = None,
) -> BinnedMean:
    """Build the in-scan accumulator of a coupled run's monthly means.

    Everything the reduction needs it takes from ``coupler``: the diagnostics
    a step produces (their structure, shapes and dtypes), the coupling
    timestep, and the start date. Every bin is a real Gregorian calendar
    month -- each record is binned by ``jcm.date.gregorian_ymd_from_days`` of
    its own interval midpoint -- so February holds 28 or 29 records exactly
    as the run's actual dates say, with no fixed-length month table and no
    start-of-year phase to compute. The caller supplies nothing but the
    coupler -- and, if the run's months are each to have a bin of their own,
    how many::

        monthly_mean(coupler)                          # (12, ...): a climatology
        monthly_mean(coupler, total_time="3650 days")  # (~121, ...): every month
        monthly_mean(coupler, n_months=121)            # sized directly instead

    The two forms bin by the same rule; they differ in what the accumulator
    *is*. Twelve bins are the calendar months, so a ten-year run composites
    its ten Januaries into bin 0 -- a climatology. ``n_months`` bins are the
    months the run passes through, in order, so the same run gives January of
    year 1 in bin 0 and December of year 10 in bin 119 -- the monthly time
    series of the run.

    **Which month a record counts in.** A record is labelled at its own
    interval's midpoint (``TimeAxis.datetimes``), and the bin is that
    instant's real Gregorian month, so ``monthly.finalize(accumulator)`` is
    ``coupler.to_xarray(diagnostics).groupby("time.month").mean()`` of the
    same run, leaf for leaf, exactly -- there being one calendar, a bin and
    the label it corresponds to can never part company.

    **The twelve-bin form wraps at the year**: a three-year run's January bin
    holds all three Januaries.

    **The sequential form wraps at ``n_months``**: a run that outlasts the
    accumulator is folded back modulo it, so size it with ``total_time``
    (which counts the calendar months the run actually touches) when every
    month is meant to stand on its own rather than composite with a later
    one ``n_months`` months on.

    **Sub-steps.** A component the workflow runs ``n > 1`` times per coupled
    step keeps that axis: its mean comes back as ``(12, n, ...)``, because
    each of its ``n`` records is binned by **its own** midpoint rather than
    the coupled step's -- a monthly-mean diurnal cycle for a component
    sub-cycling through the day. :func:`fold_records` folds that axis away,
    weighted by each slot's own count, when the plain monthly mean is what
    was wanted::

        sums, counts = accumulator                     # counts["atm"]: (12, n)
        means = monthly.finalize(accumulator)          # means["atm"]: (12, n, ...)
        per_month = fold_records(means["atm"], counts["atm"])   # (12, ...)

    A nested :class:`~jem.base.coupler.Coupler` is treated the same way; see
    :func:`_build_binned_mean`.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for.
    carry : CoupledCarry, optional
        A carry to take the diagnostics' shapes from; see
        :func:`_build_binned_mean`.
    n_months : int, optional
        Build the **sequential** form with this many bins: the months the run
        passes through, in order, starting with the one the run starts in. At
        most one of this and ``total_time``; neither gives the twelve-bin
        climatology.
    total_time : str or float, optional
        Build the sequential form sized to cover a run of this fixed length
        (a ``jcm.date.parse_duration_days`` string, or a number of days) --
        need not be a whole number of months or of years; the bin count is
        simply the number of distinct calendar months the run's records
        touch.

    Returns
    -------
    BinnedMean
        Whose ``finalize`` returns ``(12, ...)`` leaves, January first, or
        ``(n_months, ...)`` leaves, the run's first month first.

    Raises
    ------
    ValueError
        If both ``n_months`` and ``total_time`` are given, if ``n_months`` is
        not a positive integer, or if ``total_time`` is not positive or not a
        whole number of coupling steps.

    """
    dt_seconds = _exact_seconds(coupler.dt_seconds, "the coupling timestep")
    if n_months is not None and total_time is not None:
        raise ValueError(
            "Give at most one of n_months and total_time: n_months sets the "
            "accumulator's size directly and total_time counts it from the "
            "run, while giving neither is the twelve-month climatology (got "
            f"n_months={n_months!r}, total_time={total_time!r})."
        )

    def bin_of_month(
        elapsed: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], n_bins: int
    ) -> Callable[[jnp.ndarray, int, jdt.Datetime], jnp.ndarray]:
        def bin_of_record(record: jnp.ndarray, record_seconds: int,
                          record_time: jdt.Datetime) -> jnp.ndarray:
            del record
            year, month = _record_month(record_time, record_seconds)
            return jnp.mod(elapsed(year, month), n_bins).astype(jnp.int32)

        return bin_of_record

    if n_months is None and total_time is None:
        # Bin 0 is January, whatever year: a multi-year run composites its
        # Januaries into it, which is what the fixed (12, ...) accumulator is
        # for.
        return _build_binned_mean(
            coupler,
            bin_of_month(lambda year, month: month - 1, MONTHS_PER_YEAR),
            MONTHS_PER_YEAR,
            carry,
        )

    # The sequential form: bin 0 is the month the run starts in.
    from jcm.date import gregorian_ymd_from_days

    start_year_arr, start_month_arr, _ = gregorian_ymd_from_days(
        np.asarray(coupler.start_date.delta.days)
    )
    start_year, start_month = int(start_year_arr), int(start_month_arr)

    if total_time is not None:
        total_seconds = _duration_to_seconds(total_time, "total_time")
        bins = _months_covering_gregorian(
            coupler.start_date, dt_seconds, total_seconds
        )
    elif isinstance(n_months, bool) or not isinstance(n_months, int) or n_months < 1:
        # `bool` is an `int`, and `n_months=True` would silently build a
        # one-month accumulator.
        raise ValueError(f"n_months must be a positive integer; got {n_months!r}.")
    else:
        bins = n_months

    def elapsed_months(year: jnp.ndarray, month: jnp.ndarray) -> jnp.ndarray:
        return (year - start_year) * MONTHS_PER_YEAR + (month - start_month)

    return _build_binned_mean(coupler, bin_of_month(elapsed_months, bins), bins, carry)


def windowed_mean(
    coupler: Any,
    window: str | float | Sequence[str | float],
    *,
    n_windows: int | None = None,
    total_time: str | float | None = None,
    carry: Any = None,
) -> BinnedMean:
    """Build the in-scan accumulator of a run's means over successive windows.

    The reduction a sub-seasonal forecast is scored on: the mean over each
    pentad, or each week, of a run, rather than over each calendar month::

        pentads = windowed_mean(coupler, "5 days", n_windows=73)   # a year
        weeks = windowed_mean(coupler, "7 days", total_time="365 days")

    The windows need not all be the same length. Give a **sequence** of
    lengths and it is a pattern the windows cycle through, repeating for as
    long as the accumulator is -- a forecast scored on daily leads for its
    first week and pentads thereafter, a run alternating a spin-up window with
    a sampling one::

        leads = windowed_mean(coupler, [1, 1, 1, 1, 1, 1, 1, 5, 5],
                              total_time="30 days")

    **Every window, whatever its length, is measured from the run's start
    date**, with no phase and no reference to the calendar -- windows are
    plain fixed-length record counts, unlike :func:`monthly_mean`'s real
    Gregorian months. A run starting on 1 July therefore has its first
    30-day window span 1-30 July regardless of how long July itself is; use
    :func:`monthly_mean` for bins that are calendar months. No ``offset=``
    knob is offered here for the same reason ``inclusive=`` is not: a
    window's meaning is "so many records into the run", and a bin that means
    something else belongs to the builder that knows what.

    **Which window a record counts in.** Purely by record count, closed at
    the *end* of each window (:func:`_variable_window_rule`): record ``k``
    (of length ``r``) is the ``(k+1)``-th record from the start of the run,
    and window ``w`` is the records whose ordinal position falls in
    ``(start of w, end of w]``, so with daily coupling the first 5-day window
    is records 1 to 5, which is what a forecast means by "the first pentad".
    This is a plain count of records, not a date -- unlike
    :func:`monthly_mean`, which bins by each record's own real calendar
    month.

    **A run longer than the accumulator wraps**, exactly as
    :func:`monthly_mean` wraps at a year (or at ``n_months``): window ``w``
    then also collects
    windows ``w + n_windows``, ``w + 2·n_windows`` and so on, giving the
    composite of every *w*-th window of the run. That is the price of an
    accumulator whose size is fixed at trace time and does not grow with the
    run -- the whole point of reducing inside the scan. Size the accumulator
    to the run (pass ``total_time``, or ``n_windows`` counted for the run) if
    each window is meant to stand on its own; with a *pattern* of window
    lengths the wrap is at the sum of all ``n_windows`` lengths rather than at
    the end of one cycle of the pattern.

    Sub-steps and nested couplers are accumulated exactly as
    :func:`monthly_mean` describes: a component recording ``n`` times per
    coupled step keeps that axis (``(n_windows, n, ...)``) and each of its
    records is binned by its own label, so a window boundary falls between two
    sub-steps of a coupled step wherever the labels say it does.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the accumulator is for.
    window : str or float or sequence of str or float
        Length of one window, as a ``jcm.date.parse_duration_days`` string
        (``"5 days"``, ``"2 days"``, ``"12 hours"``) or a number of days --
        a **fixed** duration; ``jcm.date`` has no calendar-dependent unit
        ("1 month", "1 year") to parse, since those are not a fixed number of
        seconds. No window is a calendar month, whatever its length: calendar
        months come from :func:`monthly_mean`, which bins by the real
        calendar instead of by a fixed length. Or a **sequence** of such
        lengths, which is a pattern the windows cycle through (``["10 days",
        "20 days"]``). Every length must be a whole number of coupling steps:
        a window that ended part-way through a step would have to attribute
        that step to one side or the other, and there is no defensible
        choice.
    n_windows : int, optional
        How many windows the accumulator holds -- the length of the leading
        axis of everything ``finalize`` returns, and, when it exceeds the
        length of a pattern, how far the pattern is repeated. For a single
        window length, exactly one of this and ``total_time`` must be given;
        for a pattern, giving neither means one cycle of it
        (``n_windows = len(window)``).
    total_time : str or float, optional
        The length of the run, in the same forms as a single ``window``, from
        which ``n_windows`` is counted: enough windows to cover the run, the
        last one short if the run does not end on a window boundary (its mean
        is then over the steps that did fall in it, because every bin is
        divided by its own count).
    carry : CoupledCarry, optional
        A carry to take the diagnostics' shapes from; see
        :func:`_build_binned_mean`.

    Returns
    -------
    BinnedMean
        Whose ``finalize`` returns ``(n_windows, ...)`` leaves, the first
        window of the run first.

    Raises
    ------
    ValueError
        If ``window`` is an empty sequence, if both ``n_windows`` and
        ``total_time`` are given (or neither, for a single window length), if
        ``n_windows`` is not a positive integer, if any duration is not
        positive, or if any window length is not a whole number of coupling
        steps.

    """
    dt_seconds = _exact_seconds(coupler.dt_seconds, "the coupling timestep")
    if dt_seconds <= 0:
        raise ValueError(
            f"The coupling timestep is {dt_seconds} s; a windowed mean needs a "
            "positive one to count steps per window."
        )

    # A string is iterable, and one is a single window rather than a pattern
    # of one-character ones; everything else iterable is a pattern.
    pattern = not isinstance(window, str) and isinstance(window, Iterable)
    entries: list[Any] = list(window) if pattern else [window]  # type: ignore[arg-type]
    if not entries:
        raise ValueError(
            "window=[] has no windows for the accumulator to cycle through; "
            "give at least one length."
        )
    lengths_seconds = []
    for position, entry in enumerate(entries):
        what = f"window[{position}]" if pattern else "window"
        length_seconds = _duration_to_seconds(entry, what)
        if length_seconds % dt_seconds:
            raise ValueError(
                f"{what}={entry!r} is {length_seconds} s, which is not a whole "
                f"number of coupling steps of {dt_seconds} s. A window that "
                "ended part-way through a coupled step could only be filled by "
                "splitting that step between two windows, which the reduction "
                "does not do."
            )
        lengths_seconds.append(length_seconds)

    if n_windows is not None and total_time is not None:
        raise ValueError(
            "Give exactly one of n_windows and total_time: n_windows sets the "
            "accumulator's size directly, total_time counts it from the run "
            f"(got n_windows={n_windows!r}, total_time={total_time!r})."
        )
    if total_time is not None:
        total_seconds = _duration_to_seconds(total_time, "total_time")
        # Round *up*: a run that is not a whole number of windows ends inside
        # one, and that window has to exist to hold it. It is divided by its
        # own count like every other, so a short final window is the mean of
        # what fell in it rather than a mean diluted by missing steps.
        bins, covered = 0, 0
        while covered < total_seconds:
            covered += lengths_seconds[bins % len(lengths_seconds)]
            bins += 1
    elif n_windows is None:
        if not pattern:
            raise ValueError(
                "Give exactly one of n_windows and total_time: n_windows sets "
                "the accumulator's size directly, total_time counts it from "
                "the run (got n_windows=None, total_time=None). Only a "
                "sequence of window lengths may be given neither, and then it "
                "is used once through."
            )
        # One cycle of the pattern is the only size a pattern implies on its
        # own, and it is the useful one: a fortnight of alternating windows.
        bins = len(lengths_seconds)
    elif isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 1:
        # `bool` is an `int`, and `n_windows=True` would silently build a
        # one-window accumulator -- i.e. a mean of the whole run.
        raise ValueError(f"n_windows must be a positive integer; got {n_windows!r}.")
    else:
        bins = n_windows

    # The window boundaries: the cumulative sum of the `bins` lengths, the
    # pattern cycling for as long as the accumulator is. The last boundary is
    # the period the whole accumulator repeats with, so a run longer than it
    # wraps -- and a pattern sized to the run (via `total_time`) does not
    # wrap at all, which is the point of being able to size it.
    boundaries = np.cumsum(
        [lengths_seconds[index % len(lengths_seconds)] for index in range(bins)],
        dtype=np.int64,
    )
    # Offset 0: the windows are measured from the run's own start, purely by
    # record count. Closed at the end: record k is the (k+1)-th record from
    # the start of the run, and a window is a run of consecutive records, so
    # the record landing exactly on a boundary is the last of the window it
    # closes.
    window_index = _variable_window_rule(boundaries, 0, "right")

    return _build_binned_mean(coupler, window_index, bins, carry)
