"""Turn a chunk of a coupled run into files on disk.

A run produces stacked diagnostics; :meth:`jem.base.coupler.Coupler.to_xarray`
turns those into one labelled :class:`xarray.Dataset` per component. What is
left -- and what every driver has had to re-invent -- is the last step:
thinning or averaging the records, and writing them out under names that say
which component they came from and where on the run's clock they start. That
is this module:

- :func:`chunk_datasets` labels a chunk's diagnostics -- ``Coupler.to_xarray``
  with the chunk's first step -- and reduces nothing,
- :func:`postprocess` reduces one dataset (subsample, chunk mean) and
  :func:`postprocess_datasets` does that to a whole mapping of them,
- :func:`write_chunk` writes a mapping of datasets to
  ``<output_dir>/<component>-<first step>.nc``, naming each file with
  :func:`output_file_name`,
- :func:`output_file_step` is that naming rule read backwards -- the coupled
  step a file in the directory holds, or None if the run did not write it --
  so that anything which has to *recognise* this module's files (the driver's
  check of what a resume would leave behind past a restart point) cannot
  drift from what wrote them,
- :func:`datasets_for_chunk` is the labelling and the reduction in one call,
  for a caller that wants only the reduced form.

The labelling and the reduction are separate calls because a chunk has two
consumers that need different things from it. What is *written* is the reduced
form -- that is the point of ``output_averages`` and ``subsample``. What a
health gate inspects must be the **unreduced** chunk: both reductions are
lossy in exactly the direction a gate cares about, since a time mean skips
NaNs and dilutes a finite extreme, and a stride can drop the very last record
-- so a state that went bad at the end of a chunk would be reported healthy.
:func:`jem.driver.run_chunked` therefore calls :func:`chunk_datasets` once and
feeds the gate that, reducing only the copy it writes.

The **coupled step a chunk starts at** is also what places the chunk in the
run's ``subsample`` stride (see below), and what labels a file, rather than a
chunk index. A chunk index counts chunks of one particular length, so the same
simulated time has a different index under a different chunk length -- and a
run resumed with a different chunk (a perfectly legitimate choice: the chunk
belongs to the run, not to the checkpoint) would then write over a file the
earlier run already wrote, with different contents. The coupled step is the
run's clock: it is unique whatever the chunking, it is the same number the
checkpoint holds and the one the records are labelled from, and zero-padding
it keeps a directory listing in run order.

Nothing here holds state, opens a run or decides when a chunk ends; the run
loop does that and calls these.

What ``subsample`` means here
-----------------------------
``subsample=n`` keeps every ``n``-th **coupled step of the run** -- the step
``s`` when ``s % n == 0``, counting from the start of the run -- and with it
every record that step produced. Two consequences are the point of counting
run-global steps rather than the records in front of us. The retained cadence
is regular across chunk boundaries and across a resume: an uninterrupted run,
the same run in chunks of any length, and a run continued from a checkpoint
all write exactly the same records, so ``chunk`` stays free to be chosen for
memory and restart granularity alone. And a component the workflow runs ``n``
times per coupled step -- or a nested coupler's inner steps -- keeps all of a
kept step's records and none of a dropped step's, rather than being thinned at
a different rate from everyone else; the stride is in coupled steps, not in
records. :func:`postprocess` therefore takes the chunk's ``first_step`` and
its number of coupled ``steps``, and reads each component's records per step
off its own record count.

A chunk can then contain no step on the stride at all -- a ``subsample``
longer than the chunk gives that, and so does the short final batch a resume
under a different chunk length ends with. Such a chunk reduces to no records
and :func:`write_chunk` writes **no file** for it: an empty netCDF file would
carry nothing and its zero-length dimension would make
``xr.open_mfdataset(files)`` fail on the whole directory.

What ``output_averages`` means here
-----------------------------------
JCM's ``run.output_averages`` switches each saved record from an
instantaneous snapshot at the end of its save interval to the **mean over
that interval**, labelled at the interval's **midpoint** and carrying an
exact ``time_bounds`` variable naming the interval itself (jax-gcm v3, PR 878,
``jcm.predictions.ModelPredictions.to_xarray``; ``jcm/config/run/default.yaml``
is where a bare jcm run turns this on). The coupler's records are already one
per coupling step, so the same idea one level up: the coupler's output
interval is the **chunk**, and ``output_averages=True`` replaces a chunk's
records with their time mean -- one record, labelled at the *chunk's own*
midpoint, carrying the CF ``cell_methods = "time: mean"`` that says so.
As in JCM's own interval means, an integer or boolean time series (a step
counter, a convection type, a flag) is categorical and has no meaningful
mean, so it is left out of the chunk mean and named in the dataset's
``omitted_interval_mean_variables`` attribute, alongside any names already
listed there. JCM's own output never reaches this point with one:
``JCMComponent`` always steps JCM with averaging on, so JCM drops them
itself and its list is carried through. The rule is for any other
component that records one; with ``output_averages`` off, such a component
keeps it.

For a dataset that itself carries ``time_bounds`` (JCM's, read off its
``time`` coordinate's CF ``bounds`` attribute rather than assumed by name),
that chunk midpoint and the chunk's own ``time_bounds`` are computed exactly,
from the first and last (pre-``subsample``) record's own bounds, and
``time_bounds`` is carried through as the bound it is rather than averaged
like a sampled quantity (:func:`postprocess`): treating `time_bounds` as an
ordinary variable to average would be nonsense for a bound, and labelling
the mean with the chunk's *last* record's own midpoint would give neither
the chunk's true midpoint nor its end. A dataset with no such bounds (every
non-JCM component today) has no bound to read, but its records are still
labelled at their own interval's midpoint by the same shared ``TimeAxis``
JCM's are, so the chunk's own midpoint is still computed exactly -- as the
average of the first and last (pre-``subsample``) record's own midpoint
labels, which equals the chunk's true midpoint for any equal-length,
contiguous run of records regardless of the per-record interval length (see
:func:`postprocess`'s body for the derivation). Every component's averaged
record is therefore labelled identically for the same chunk, which is what
lets ``xr.merge(..., join="exact")`` of two components' averaged output
succeed at all: labelling a chunk's non-``time_bounds`` mean with its *last*
record's own label instead would tie the label's meaning to whichever
per-record convention happens to be live (an end-of-interval label before
jax-gcm PR 878, a midpoint label after it), so it would approximate neither
the chunk's end nor its midpoint in general, and two components' averaged
chunks could disagree with each other.

That keeps JCM's rule ("one record per output interval, the mean over it,
labelled at its midpoint") rather than inventing a second meaning for the same
word, and it is the reduction a long run actually needs: one record per chunk
rather than one per coupling step. The bins are the **chunks**, so a 30-day
chunk gives 30-day-window means and not calendar months -- on a 365-day
calendar those windows drift about five days a year against the months. A
calendar-month mean is :func:`jem.accumulate.monthly_mean`, which bins every
record against its own interval (its midpoint, to match the label -- see that
function's docstring) whatever the chunking. Note that in a coupled run the
atmosphere's per-step records are *already* step means -- the JCM wrapper
integrates each coupling step with ``output_averages=True`` -- so averaging a
chunk of them is the chunk mean exactly, with no double counting of the
*values*. Their ``cell_methods`` already says ``"time: mean"`` too, though,
and :func:`postprocess` must not double-count *that*: appending the same bare
text again would give the meaningless ``"time: mean time: mean"``, so it is
skipped when the text is already identical to the tag already there, and
otherwise (a subsampled chunk, see :func:`postprocess`'s own docstring)
appended with a distinguishing comment instead of a bare repeat (see
:func:`_with_cell_method`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from jem.base.coupler import Coupler

logger = logging.getLogger(__name__)

#: The dimension the reductions here act on. Every component's ``to_xarray``
#: labels its records with it (:class:`jem.base.component.TimeAxis`).
TIME_DIMENSION = "time"

#: CF cell method :func:`postprocess` stamps on an averaged variable.
TIME_MEAN_CELL_METHOD = f"{TIME_DIMENSION}: mean"

#: The dataset attribute naming the categorical (integer or boolean) time
#: series left out of an interval mean -- jax-gcm's own name for it, so a
#: chunk mean of JCM output and JCM's own interval-mean output agree.
OMITTED_INTERVAL_MEAN_ATTRIBUTE = "omitted_interval_mean_variables"

# Characters allowed in the component part of an output file name. A dataset
# name is a component name, which a user chooses, and a nested coupler's
# components arrive under their own names -- so nothing stops a name that is
# a path fragment or worse. Anything else becomes an underscore.
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")

# What an output file name looks like: a component name, a hyphen, and the
# zero-padded coupled step its chunk starts at. The name part is greedy
# because a component name may itself contain a hyphen, and the step is
# ``\d{8,}`` rather than exactly eight because the padding is a minimum -- a
# run long enough to pass a hundred million coupled steps writes nine.
# :func:`output_file_step` still confirms a match by rebuilding the name with
# :func:`output_file_name`, so the pattern only has to be no *narrower* than
# what :func:`write_chunk` writes.
_OUTPUT_FILE_PATTERN = re.compile(r"(?P<name>.+)-(?P<step>\d{8,})\.nc\Z")


def _safe_name(name: str) -> str:
    """Return ``name`` reduced to characters that are safe in a file name."""
    safe = _UNSAFE_NAME_CHARACTERS.sub("_", name).lstrip(".")
    if not safe:
        raise ValueError(
            f"Dataset name {name!r} has no characters that can appear in a file "
            "name; rename the component."
        )
    if safe != name:
        logger.debug("Output name %r written as %r.", name, safe)
    return safe


def _timed_variables(dataset: xr.Dataset) -> list[str]:
    """Return the data variables that have a time dimension.

    Excludes the CF bounds variable the ``time`` coordinate's own ``bounds``
    attribute names, if any (jax-gcm's ``time_bounds``): it is a *bound* on
    the interval a record covers, not a sampled quantity, so it must not be
    averaged like one -- see :func:`_chunk_time_bounds`, which recomputes it
    for the chunk as a whole instead.
    """
    excluded = _time_bounds_name(dataset)
    return [
        str(name)
        for name, variable in dataset.data_vars.items()
        if TIME_DIMENSION in variable.dims and str(name) != excluded
    ]


def _time_bounds_name(dataset: xr.Dataset) -> str | None:
    """Return the name of ``dataset``'s CF time-bounds variable, if it has one.

    Read from the ``time`` coordinate's own ``bounds`` attribute (the CF
    convention jax-gcm's ``ModelPredictions.to_xarray`` sets,
    ``time.attrs["bounds"] == "time_bounds"``) rather than a hardcoded name,
    so a component that names its bounds variable something else, or has
    none at all (every non-JCM component today), is handled correctly either
    way.
    """
    name = dataset[TIME_DIMENSION].attrs.get("bounds")
    if name is None or name not in dataset.data_vars:
        return None
    return str(name)


def _with_cell_method(attrs: Mapping[str, Any], method: str) -> dict[str, Any]:
    """Return ``attrs`` with ``method`` appended to its CF ``cell_methods``.

    Skipped when ``method`` is already the existing string verbatim, or its
    trailing entry: a JCM variable already carries ``cell_methods = "time:
    mean"`` by the time :func:`postprocess` sees it, because JCM integrates
    each coupling step with ``run.output_averages=True`` (see the module
    docstring) -- so a bare, unannotated call here would append the identical
    text a second time, giving ``"time: mean time: mean"``. Two indistinguishable
    entries say nothing a single one didn't already, so that is a duplicate to
    suppress, not a description of a genuine second reduction (see
    ``test_postprocess_does_not_duplicate_an_identical_cell_method_already_present``).

    A *distinguishing* ``method`` -- the parenthetical
    :func:`postprocess` builds for a subsampled chunk, see its own docstring
    -- is never identical to a bare one, so it is always appended even when a
    bare "time: mean" is already present: unlike the first case, the two
    entries genuinely say different things (JCM's own per-step average vs.
    this subsampled chunk average) and neither should be dropped.
    """
    updated = dict(attrs)
    existing = str(updated.get("cell_methods", "")).strip()
    if existing == method or existing.endswith(f" {method}"):
        return updated
    # CF cell methods are a space-separated list, applied in order.
    updated["cell_methods"] = f"{existing} {method}".strip()
    return updated


def check_subsample(subsample: Any) -> int:
    """Return ``subsample`` if it is a valid coupled-step stride, else raise.

    The rule lives here, beside the reduction that applies it, and is called
    both by :func:`postprocess` and -- before anything is compiled -- by
    :func:`jem.driver.run_chunked`, so a run configured with a nonsensical
    stride is refused up front rather than after a chunk has been integrated.

    Parameters
    ----------
    subsample : Any
        The value to check.

    Returns
    -------
    int
        ``subsample`` itself.

    Raises
    ------
    ValueError
        If it is not a positive integer. ``bool`` is rejected too: ``True``
        would pass ``isinstance(_, int)`` and silently mean "keep every
        step", which is not what anyone writing it meant.

    """
    if not isinstance(subsample, int) or isinstance(subsample, bool) or subsample < 1:
        raise ValueError(f"subsample must be a positive integer; got {subsample!r}.")
    return subsample


def _records_per_step(n_records: int, steps: int) -> int:
    """Return ``n_records // steps``, or raise if it does not divide evenly.

    Shared by :func:`_kept_records` (the stride needs to know which coupled
    step a record belongs to) and :func:`_assert_contiguous_chunk` (the
    chunk-mean label/``time_bounds`` derivation needs the records to be a
    whole number of equal-length coupled steps to begin with) -- one message
    for the one underlying condition, rather than two call sites drifting
    apart on what "not a whole multiple" is called.

    Raises
    ------
    ValueError
        If ``n_records`` is not a whole positive multiple of ``steps``.

    """
    records_per_step, remainder = divmod(n_records, steps)
    if remainder or records_per_step < 1:
        raise ValueError(
            f"A chunk of {steps} coupled step(s) cannot hold {n_records} "
            f"record(s): a component records the same whole number of times "
            f"every coupled step, so the coupled step each record belongs to "
            f"-- and with it a stride over coupled steps -- is undefined here."
        )
    return records_per_step


def _kept_records(
    n_records: int, *, first_step: int, steps: int, subsample: int
) -> slice | np.ndarray:
    """Return the time selection that keeps every ``subsample``-th coupled step.

    The stride counts **coupled steps of the run**, not records of this chunk:
    the step ``s`` is kept when ``s % subsample == 0``, counting from the start
    of the run, so step 0 is always kept. That is what makes the retained
    cadence a property of the run rather than of how it happened to be cut up:
    a chunk that begins part-way through a stride period continues the phase
    instead of restarting it, so an uninterrupted run, the same run in chunks
    of any length, and a run resumed from a checkpoint all keep exactly the
    same records. A stride reapplied from each chunk's first record instead
    would give an irregular cadence, and more output than was asked for:
    three-step chunks with ``subsample=2`` would keep global coupled steps
    0, 2, 3 and 5 rather than 0, 2 and 4.

    A component the workflow runs ``n > 1`` times per coupled step (or a
    nested coupler's inner steps) contributes ``n`` consecutive records per
    step, and they are kept or dropped **together**, because the stride is in
    coupled steps and not in records. ``n`` is read off the chunk as
    ``len(time) // steps`` rather than asked of the coupler, so this works for
    whatever produced the dataset.

    Parameters
    ----------
    n_records : int
        Records along the time dimension of the chunk being reduced.
    first_step : int
        The coupled step the chunk starts at -- the same number that labels
        its records and its file.
    steps : int
        Coupled steps in the chunk.
    subsample : int
        The stride, in coupled steps; already checked by
        :func:`check_subsample`.

    Returns
    -------
    slice or numpy.ndarray
        Something to hand to ``Dataset.isel``. A plain slice for the usual
        one-record-per-step chunk, and an index array when a step contributes
        several records, since those are runs of ``n`` and not a stride.

    Raises
    ------
    ValueError
        If ``steps`` is not positive, ``first_step`` is negative, or
        ``n_records`` is not a whole positive multiple of ``steps`` -- a
        component records the same number of times every coupled step, so
        anything else leaves the step a record belongs to, and with it the
        stride, undefined.

    """
    if steps < 1:
        raise ValueError(f"steps must be a positive integer; got {steps!r}.")
    if first_step < 0:
        raise ValueError(f"first_step must not be negative; got {first_step!r}.")
    records_per_step = _records_per_step(n_records, steps)
    # Where this chunk sits in the run's stride period: the first of its
    # coupled steps whose run-global index is a multiple of `subsample`.
    first_kept = -first_step % subsample
    if records_per_step == 1:
        return slice(first_kept, None, subsample)
    kept_steps = np.arange(first_kept, steps, subsample)
    return (
        kept_steps[:, np.newaxis] * records_per_step + np.arange(records_per_step)
    ).ravel()


def _dataset_is_empty(dataset: xr.Dataset) -> bool:
    """Return whether ``dataset`` has a time dimension holding no record.

    The one definition of "nothing to write", shared by :func:`write_chunk`,
    which skips such a dataset, and by :func:`postprocess`, which has nothing
    to average in one. A dataset with **no** time dimension is not empty in
    this sense: it is a grid or a mask, and it is written and passed through
    as it always is.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to examine.

    Returns
    -------
    bool

    """
    return dataset.sizes.get(TIME_DIMENSION, 1) == 0


def _assert_contiguous_chunk(
    dataset: xr.Dataset, *, bounds_name: str | None, steps: int | None
) -> None:
    """Raise unless ``dataset``'s records are the contiguous run ``postprocess`` assumes.

    ``postprocess``'s chunk-mean label -- and, for a dataset with
    ``time_bounds``, the chunk-wide bound it recomputes -- is derived from
    only the *first* and *last* record (see that function's own docstring's
    derivation), which is exact only when the records in between are their
    equal-length, gap-free continuation. Nothing upstream of
    :func:`postprocess` enforces that on its own: a caller handing it a
    record count that is not a whole number of ``steps``, or a set of
    records with an irregular gap, would otherwise get a label back anyway
    -- silently wrong rather than refused -- so this check refuses it here
    instead.

    Every chunk :func:`datasets_for_chunk`/``Coupler.to_xarray`` actually
    produce is contiguous by construction, so this should never fire on a
    real run; it exists to catch a chunk assembled some other way before it
    can mislabel silently, which is why it is checked here rather than left
    to the derivation itself to get subtly wrong.

    Parameters
    ----------
    dataset : xarray.Dataset
        The chunk as given to :func:`postprocess`, before ``subsample``
        removes any record -- the derivation runs on the chunk as given, so
        that is what must be contiguous.
    bounds_name : str, optional
        The CF ``time_bounds`` variable's name, from :func:`_time_bounds_name`,
        or ``None`` if this dataset has none.
    steps : int, optional
        Coupled steps in the chunk, as passed to :func:`postprocess`; ``None``
        skips the whole-multiple check (the same default ``postprocess``
        itself uses: one record per coupled step).

    Raises
    ------
    ValueError
        If ``steps`` is given and is not positive, if the record count is not
        a whole multiple of it, or if the records are not evenly spaced (no
        ``time_bounds``) / contiguously bounded (with ``time_bounds``).

    """
    n_records = int(dataset.sizes[TIME_DIMENSION])
    if steps is not None:
        if steps < 1:
            raise ValueError(f"steps must be a positive integer; got {steps!r}.")
        _records_per_step(n_records, steps)
    if n_records < 2:
        # A single record is trivially "contiguous" -- there is nothing
        # between a record and itself to have a gap.
        return
    if bounds_name is not None:
        bounds_dim = next(d for d in dataset[bounds_name].dims if d != TIME_DIMENSION)
        starts = (
            dataset[bounds_name]
            .isel({bounds_dim: 0})
            .values.astype("datetime64[ms]")
            .astype("int64")
        )
        ends = (
            dataset[bounds_name]
            .isel({bounds_dim: 1})
            .values.astype("datetime64[ms]")
            .astype("int64")
        )
        if not np.array_equal(ends[:-1], starts[1:]):
            raise ValueError(
                "Cannot average this chunk: its records' time_bounds are not "
                "contiguous (record k's interval must end exactly where "
                "record k+1's begins). postprocess's chunk-wide time_bounds "
                "and label are built from only the first record's own start "
                "and the last record's own end, which would silently claim "
                "coverage over a gap -- or double-count an overlap -- that "
                "these records do not actually have."
            )
    else:
        times = (
            dataset[TIME_DIMENSION].values.astype("datetime64[ms]").astype("int64")
        )
        gaps = np.diff(times)
        if not np.all(gaps == gaps[0]):
            raise ValueError(
                "Cannot average this chunk: its records are not evenly "
                f"spaced in time (gaps of {sorted(set(gaps.tolist()))} "
                "milliseconds between consecutive records). postprocess's "
                "chunk label -- the average of the first and last record's "
                "own midpoint -- is exact only for a contiguous run of "
                "equal-length intervals (see postprocess's own docstring's "
                "derivation)."
            )


def postprocess(
    dataset: xr.Dataset,
    *,
    output_averages: bool = False,
    subsample: int = 1,
    first_step: int = 0,
    steps: int | None = None,
) -> xr.Dataset:
    """Thin and/or average one component's records for a chunk.

    The two reductions compose in the order they are listed: ``subsample``
    chooses which records are kept, and ``output_averages`` then reduces
    whatever is left to its mean. That order is what the two words mean --
    the stride is decided from the run's clock, which averaging cannot move,
    and reversing them would make the stride select among one-record chunk
    means, a stride over *chunks* rather than over coupled steps.

    Asking for both is legal but unusual, and a run normally sets one or the
    other, because the mean is then over the **kept** records only. It is
    still the mean of one chunk, labelled at that chunk's own midpoint (or, on
    a dataset with no ``time_bounds`` to compute one from, the chunk's last
    record's own label -- see the module docstring), so the series of means
    stays one record per chunk, evenly spaced with the chunks. What it is not
    is evenly *weighted*: how many of a chunk's coupled steps the run-global
    stride keeps depends on where the chunk falls in the stride period, so
    successive means can average different numbers of records, and a chunk
    that keeps none yields no mean at all (and, through :func:`write_chunk`,
    no file).

    Variables without a time dimension (grid masks, layer thicknesses) are
    passed through untouched by both reductions. A ``time_bounds`` variable
    (identified by the ``time`` coordinate's own CF ``bounds`` attribute, not
    by name) is a third case: it has a time dimension but is a *bound*, not a
    sampled quantity, so it is neither averaged nor stamped with
    ``cell_methods`` -- it is recomputed for the chunk as a whole instead (see
    the module docstring).

    Parameters
    ----------
    dataset : xarray.Dataset
        One component's chunk of output, from ``Coupler.to_xarray``.
    output_averages : bool
        Replace the records with their time mean: one record, labelled at the
        chunk's own midpoint exactly -- from ``time_bounds`` when the dataset
        carries one, otherwise from the average of its first and last
        record's own midpoint labels (see the module docstring) -- whether or
        not ``subsample`` dropped a record from the mean itself, with
        ``cell_methods = "time: mean"`` appended to every variable that was
        averaged (skipped if that exact text is already the trailing entry --
        a JCM variable already carries it from JCM's own per-step average --
        see :func:`_with_cell_method`). When ``subsample > 1`` the appended
        text instead carries an explicit, nesting-free CF comment saying the
        mean was computed from only the kept records while the recorded time
        still spans the chunk's whole interval, so the two do not get
        conflated into one misleadingly plain "time: mean". The comment's
        wording is deliberately valid CF and does not assume a
        ``time_bounds`` exists. See the module docstring for why the chunk
        is the averaging interval.
    subsample : int
        Keep every ``subsample``-th **coupled step** of the run, counting
        from its start, with all of the records that step produced; ``1``
        (the default) keeps everything. See :func:`_kept_records` for why the
        stride is in coupled steps of the run rather than in records of this
        chunk.
    first_step : int
        The coupled step this chunk starts at, which is what places the
        chunk in the run's stride period. The default, 0, means "the start of
        the run", so a single dataset reduced on its own keeps its records
        0, ``subsample``, ``2 * subsample`` ...
    steps : int, optional
        Coupled steps in the chunk, from which the records each step
        contributed are counted (``len(time) // steps``). Defaults to the
        number of records, i.e. one record per coupled step.

    Returns
    -------
    xarray.Dataset
        A chunk none of whose coupled steps the stride keeps comes back with
        no records, rather than with a record the run's cadence does not call
        for; :func:`write_chunk` writes no file for such a chunk.

    Raises
    ------
    ValueError
        If ``subsample`` is not a positive integer, a reduction was asked for
        and the dataset has no time dimension to reduce, or (when the stride
        is applied) the record count is not a whole multiple of ``steps``.
        When ``output_averages`` is set, also if the record count is not a
        whole multiple of ``steps`` (checked here too, independently of
        ``subsample``) or the records are not the contiguous, equal-length
        run the chunk label/``time_bounds`` derivation assumes -- see
        :func:`_assert_contiguous_chunk`.

    """
    check_subsample(subsample)
    if subsample == 1 and not output_averages:
        return dataset
    if TIME_DIMENSION not in dataset.dims:
        raise ValueError(
            f"Cannot subsample or average a dataset with no {TIME_DIMENSION!r} "
            f"dimension (it has {sorted(map(str, dataset.dims))!r})."
        )

    # What the averaged record's label -- and, if this dataset carries one,
    # its `time_bounds` -- has to be computed from is the chunk's records AS
    # GIVEN, before the stride removes any: the stride chooses what goes INTO
    # the mean, not what interval the mean covers, and computing either from
    # the kept records instead would make the chunk-mean series unevenly
    # spaced (and, for `time_bounds`, wrong) whenever the stride's phase falls
    # differently in successive chunks.
    bounds_name = _time_bounds_name(dataset)
    if output_averages:
        # The label/`time_bounds` built below assume a contiguous run of
        # equal-length coupled steps (see `_assert_contiguous_chunk`'s own
        # docstring); only checked when there is actually a mean to label --
        # a `subsample`-only call never reads either.
        _assert_contiguous_chunk(dataset, bounds_name=bounds_name, steps=steps)
    if bounds_name is not None:
        # Exact: the CHUNK's own true interval is
        # [the first record's own interval start, the last record's own
        # interval end], read directly from their `time_bounds` rather than
        # approximated from the (midpoint-labelled) `time` coordinate.
        # Falling through to the no-`time_bounds` path below for a dataset
        # that DOES carry one would instead average `time_bounds` itself
        # like any other sampled variable (nonsense for a bound) and label
        # the mean with the LAST record's own midpoint (neither the chunk's
        # true midpoint nor its end) -- e.g. a real 3-day chunk starting
        # 2000-02-02 would come back labelled `02-04T12:00` (not the correct
        # midpoint, `02-03T12:00`, nor the end, `02-05`) with
        # `time_bounds=[02-03, 02-04]`, a false one-day interval for a 3-day
        # mean.
        bounds_dim = next(
            d for d in dataset[bounds_name].dims if d != TIME_DIMENSION
        )
        interval_start = dataset[bounds_name].isel(
            {TIME_DIMENSION: 0, bounds_dim: 0}
        )
        interval_end = dataset[bounds_name].isel(
            {TIME_DIMENSION: -1, bounds_dim: 1}
        )
        # Exact integer-millisecond midpoint, the same halving
        # `jem.base.component.TimeAxis.datetimes` uses for a single record's
        # midpoint, applied here to the whole chunk's interval instead.
        start_ms = interval_start.values.astype("datetime64[ms]").astype("int64")
        end_ms = interval_end.values.astype("datetime64[ms]").astype("int64")
        chunk_label = np.atleast_1d(
            np.asarray(start_ms + (end_ms - start_ms) // 2, dtype="datetime64[ms]")
        )
        chunk_bounds = np.array(
            [interval_start.values, interval_end.values]
        ).astype(dataset[bounds_name].dtype)
    else:
        # No `time_bounds`: every non-JCM component's dataset today. There is
        # no bound to read, but there is still an exact chunk midpoint to
        # compute, because every record -- on every component, `time_bounds`
        # or not -- is labelled at its OWN interval's midpoint by the one
        # shared `TimeAxis` (see that class's docstring): the records are
        # equal-length and contiguous, so the chunk's own interval is
        # `[first record's start, last record's end]` and its midpoint is,
        # by that symmetry, exactly the average of the first and last
        # record's own midpoint labels -- the per-record interval length
        # cancels (`first_start = first_label - dt/2`,
        # `last_end = last_label + dt/2`, so their mean is
        # `(first_label + last_label) / 2` whatever `dt` is). Taking the
        # chunk's last record's own label instead would be wrong under the
        # current (post-jax-gcm-PR-878) MIDPOINT convention: it approximates
        # neither the chunk's end nor (except for a one-record chunk) its
        # midpoint, and it can disagree between components after an average
        # -- a JCM dataset (which carries `time_bounds`, so takes the branch
        # above) and a slab-ocean dataset (this branch) would then label the
        # same chunk differently, breaking `xr.merge(..., join="exact")`
        # across them (see this function's docstring and
        # `test_postprocess_averages_jcm_and_slab_chunks_to_matching_labels`
        # in `tests/unit/test_coupled.py`).
        first_label = dataset[TIME_DIMENSION].isel({TIME_DIMENSION: 0}).values
        last_label = dataset[TIME_DIMENSION].isel({TIME_DIMENSION: -1}).values
        first_ms = first_label.astype("datetime64[ms]").astype("int64")
        last_ms = last_label.astype("datetime64[ms]").astype("int64")
        chunk_label = np.atleast_1d(
            np.asarray(first_ms + (last_ms - first_ms) // 2, dtype="datetime64[ms]")
        )
        chunk_bounds = None

    if subsample > 1:
        n_records = int(dataset.sizes[TIME_DIMENSION])
        dataset = dataset.isel(
            {
                TIME_DIMENSION: _kept_records(
                    n_records,
                    first_step=first_step,
                    steps=n_records if steps is None else steps,
                    subsample=subsample,
                )
            }
        )
    if not output_averages:
        return dataset
    if _dataset_is_empty(dataset):
        # The stride kept none of this chunk's coupled steps, so there is
        # nothing to average. An empty chunk is the honest answer -- inventing
        # a record here would put one in the output at a cadence the run did
        # not ask for -- and `write_chunk` writes no file for it.
        return dataset

    timed = _timed_variables(dataset)  # excludes `bounds_name`, if any
    # An integer or boolean time series is categorical (a step counter, a
    # convection type, a flag), and its mean is not a meaningful value of
    # it. jax-gcm's own interval-mean output drops such variables and names
    # them in `omitted_interval_mean_variables`; a chunk mean follows the
    # same convention, adding to any names the dataset already lists there.
    categorical = [name for name in timed if _is_categorical(dataset[name])]
    timed = [name for name in timed if name not in categorical]
    # `Dataset.mean` drops the dimension it reduces, so the label -- the
    # chunk's own true midpoint, computed exactly either way (see above) --
    # has to be put back by hand.
    if timed:
        averaged = (
            dataset[timed]
            .mean(dim=TIME_DIMENSION, keep_attrs=True)
            .expand_dims({TIME_DIMENSION: chunk_label})
        )
    else:
        # Every time series was categorical: `dataset[[]]` keeps no `time`
        # dimension to reduce, so the one-record time axis is built directly.
        averaged = xr.Dataset(coords={TIME_DIMENSION: chunk_label})
    averaged[TIME_DIMENSION].attrs = dict(dataset[TIME_DIMENSION].attrs)
    if subsample > 1:
        # The label and, for a `time_bounds`-carrying dataset, the bound
        # itself (computed earlier in this function) span the chunk's WHOLE
        # interval regardless of `subsample` -- computed from the chunk's
        # records as given, before the stride removed any -- but the mean
        # just below is only over the records the stride *kept*. A bare
        # "time: mean" would then read as "the mean of the whole interval
        # this record is labelled with", which is not quite what happened.
        # Say so explicitly with a CF comment rather than silently letting
        # the label overstate what fed the average; this also never
        # collides with an already-present bare "time: mean" (see
        # `_with_cell_method`), since it is a different, more specific
        # string.
        #
        # The comment must itself be valid CF: CF's own `(comment: ...)`
        # extra-info block does not nest, so it must contain no parenthesis
        # of its own -- e.g. "coupled step(s)" would not do, since its inner
        # "(s)" would close the block early, silently dropping everything
        # written after it from what any CF reader would parse as the
        # comment. It also must not refer to
        # "the label above" (an attribute string has no "above" to point at)
        # or assume every dataset has a `time_bounds` (a non-JCM component's
        # never does) -- so it names only what is true unconditionally: the
        # record's own label spans the whole chunk, whatever this dataset
        # does or does not also carry as an explicit bound.
        method = (
            f"{TIME_MEAN_CELL_METHOD} (comment: mean of one coupled step in "
            f"every {subsample}; the recorded time still spans the whole "
            "chunk, not just the steps that fed this mean)"
        )
    else:
        method = TIME_MEAN_CELL_METHOD
    for name in timed:
        averaged[name].attrs = _with_cell_method(dataset[name].attrs, method)
    for variable in dataset.data_vars:
        if (str(variable) not in timed and str(variable) not in categorical
                and str(variable) != bounds_name):
            averaged[variable] = dataset[variable]
    if bounds_name is not None:
        # `chunk_bounds` is set together with `bounds_name` above (both None
        # or both not), so this is always an `ndarray` here.
        assert chunk_bounds is not None
        # Set explicitly, at the new single-record shape -- copying the
        # chunk's own (still multi-record) `time_bounds` through unchanged,
        # as the loop above does for an ordinary time-invariant variable,
        # would leave it the wrong length for the one averaged record, and
        # averaging it like a sampled quantity is the bug this branch exists
        # to not repeat (see above). Not stamped with `cell_methods`: a bound
        # is not a sampled quantity that was averaged, so CF's "time: mean"
        # would misdescribe it.
        averaged[bounds_name] = (
            (TIME_DIMENSION, bounds_dim),
            chunk_bounds.reshape(1, 2),
        )
        averaged[bounds_name].attrs = dict(dataset[bounds_name].attrs)
    averaged.attrs = dict(dataset.attrs)
    if categorical:
        already = averaged.attrs.get(OMITTED_INTERVAL_MEAN_ATTRIBUTE, "")
        names = {name for name in str(already).split(",") if name}
        averaged.attrs[OMITTED_INTERVAL_MEAN_ATTRIBUTE] = ",".join(
            sorted(names | set(categorical)))
    return averaged


def _is_categorical(variable: xr.DataArray) -> bool:
    """Return whether ``variable`` holds integer or boolean values.

    The same test jax-gcm's ``ModelPredictions.to_xarray`` applies when it
    omits a variable from an interval mean.
    """
    return bool(np.issubdtype(variable.dtype, np.integer)
                or np.issubdtype(variable.dtype, np.bool_))


def output_file_name(name: str, first_step: int) -> str:
    """Return the file name a chunk of ``name`` starting at ``first_step`` takes.

    The one place the naming rule lives, so that everything which *reads* the
    directory back -- :func:`output_file_step`, and through it the driver's
    sweep of stale output on a resume -- agrees with what :func:`write_chunk`
    wrote, by construction rather than by two copies of a format string.

    Parameters
    ----------
    name : str
        The dataset's name, normally a component's; anything in it that cannot
        appear in a file name becomes an underscore.
    first_step : int
        The coupled step the chunk starts at.

    Returns
    -------
    str
        ``<component>-<first_step:08d>.nc``.

    """
    return f"{_safe_name(name)}-{first_step:08d}.nc"


def output_file_step(path: Path | str, names: Iterable[str]) -> int | None:
    """Return the coupled step ``path`` holds output for, or None if it is not ours.

    The inverse of :func:`output_file_name`, and deliberately a *strict* one:
    a candidate is accepted only when rebuilding the name from one of
    ``names`` and the step reproduces the file name character for character.
    So a file the run did not write -- another component's, a hand-made
    ``ocn.nc``, a reanalysis someone dropped in the output directory -- comes
    back as None rather than as a step, which is what lets the driver reason
    about the output a resume would overlap without ever having to guess
    whether a file is its own.

    ``names`` is required for the same reason: a step alone cannot say whose
    output a file is, and "every file whose name ends in eight digits" is not
    a set a run should be refusing to start over.

    Parameters
    ----------
    path : pathlib.Path or str
        The file to examine; only its name is looked at, and nothing is read
        from disk.
    names : Iterable[str]
        The dataset names whose files count -- the run's component names.

    Returns
    -------
    int or None
        The coupled step the chunk in ``path`` starts at, or None.

    """
    filename = Path(path).name
    match = _OUTPUT_FILE_PATTERN.fullmatch(filename)
    if match is None:
        return None
    step = int(match["step"])
    for name in names:
        if filename == output_file_name(name, step):
            return step
    return None


def write_chunk(
    datasets: Mapping[str, xr.Dataset], output_dir: Path | str, first_step: int
) -> list[Path]:
    """Write one chunk's datasets as netCDF and return the paths, in a stable order.

    One file per dataset, named by :func:`output_file_name`
    (``<component>-<first_step:08d>.nc``): the component first so a directory
    listing groups a component's files together, and the coupled step the
    chunk starts at -- zero-padded so the listing sorts in run order --
    second. That step is the run's own clock, so the name is unique however
    the run was chunked; see the module docstring for why a chunk index is
    not.

    An existing file is overwritten, with a warning. Writing a second run into
    a directory that already holds one is a deliberate act (a rerun, or a
    configuration changed and repeated), and refusing it would be worse than
    saying so. A resumed run normally does not collide, because it starts from
    the step its checkpoint holds; it does when the chunk it already wrote was
    never checkpointed -- because the run was killed in between, because the
    health gate rejected that chunk, which is deliberately not checkpointed, or
    because ``run_chunked``'s ``checkpoint_interval`` spaces the saves out and
    the run was killed after one of the chunks in between -- so the warning reports
    a fact and does not assert which of them happened. The rewrite is of the
    same name from the same starting state, because a file is named after the
    coupled step its chunk starts at.

    Overwriting is only half of what a resume owes the directory, and this
    function cannot do the other half: a file an earlier pass wrote is only
    rewritten if the resumed run writes a chunk starting at that same step,
    which it need not, since the chunk belongs to the run and not to the
    checkpoint. So :func:`jem.driver.run_chunked` checks the directory before
    it integrates anything -- using :func:`output_file_step` to pick out its
    own files -- and refuses a resume that would leave any of them behind,
    rather than interleaving one pass's files with another's holding records
    for the same simulated time.

    A dataset with **no records** is not written at all, and is reported at
    INFO. That is what a chunk holding no coupled step on the ``subsample``
    stride reduces to, and an empty netCDF file is worse than no file: it
    carries nothing, and a zero-length dimension makes
    ``xr.open_mfdataset(files)`` -- how a run's output is read back -- fail
    outright rather than skip it. The chunk's simulated time is not missing
    from the output, it is simply not on the cadence the run asked to keep.
    A dataset with no time dimension at all is a different thing (a grid, a
    mask) and is written as it always was.

    A file already at that name is **removed**, because this pass's output
    for that chunk is nothing and the name is this chunk's. That is the one
    thing here that deletes, and it deletes only a name this call is itself
    responsible for: the same name it would otherwise have overwritten. It
    matters for a rechunked resume, where an earlier pass's file can sit on
    the new chunk grid -- :func:`jem.driver.run_chunked` has already checked
    that every file at or after the restart point is one this run writes
    again -- and would otherwise survive holding a record this pass also
    writes under a different name.

    ``Coupler.to_xarray`` has already flattened a nested coupler's output
    into this mapping under its inner components' own names, so a name is
    normally a plain identifier; anything in one that cannot appear in a file
    name is replaced with an underscore.

    Parameters
    ----------
    datasets : Mapping[str, xarray.Dataset]
        The chunk's datasets, keyed by component name.
    output_dir : pathlib.Path or str
        Directory to write into. Created, with its parents, if it does not
        exist.
    first_step : int
        The coupled step this chunk starts at -- the ``step`` of the carry it
        was integrated from, and the same number
        :func:`datasets_for_chunk` labels its records from.

    Returns
    -------
    list[pathlib.Path]
        The paths written, sorted by dataset name so a caller's log, and a
        test, see one order. Shorter than ``datasets`` when one of them had
        no records to write.

    Raises
    ------
    ValueError
        If ``first_step`` is negative, or two dataset names reduce to the
        same file name.

    """
    if first_step < 0:
        raise ValueError(f"first_step must not be negative; got {first_step!r}.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    paths: dict[Path, str] = {}
    written: list[Path] = []
    for name in sorted(datasets):
        path = directory / output_file_name(name, first_step)
        if path in paths:
            raise ValueError(
                f"Datasets {paths[path]!r} and {name!r} would both be written to "
                f"{path.name!r}; rename one of the components."
            )
        paths[path] = name
        if _dataset_is_empty(datasets[name]):
            # This pass's record of this chunk, for this component, is
            # nothing -- usually because the `subsample` stride keeps none of
            # the chunk's coupled steps. Removing any file already at the
            # name is the degenerate case of rewriting it: the name belongs
            # to this chunk of this run, so leaving an earlier pass's file
            # there would leave the directory holding records this pass has
            # replaced with nothing, which a rechunked resume turns into a
            # duplicate of a record it writes under another name.
            removed = path.exists()
            path.unlink(missing_ok=True)
            logger.info(
                "Coupled step %d: %r holds no records, so %s was not "
                "written%s. The usual cause is a subsample stride that keeps "
                "none of this chunk's coupled steps.",
                first_step, name, path.name,
                " and the file already there was removed" if removed else "",
            )
            continue
        if path.exists():
            logger.warning(
                "%s already exists and is being overwritten: this directory "
                "already holds output for coupled step %d. Either this is a "
                "rerun into the same output directory, or an earlier run "
                "wrote this chunk without checkpointing it -- killed in "
                "between, or stopped by the health gate -- so the resume is "
                "repeating the chunk.",
                path, first_step,
            )
        datasets[name].to_netcdf(path, engine="netcdf4")
        written.append(path)
    logger.debug(
        "Coupled step %d: wrote %d file(s) to %s.", first_step, len(written), directory
    )
    return written


def chunk_datasets(
    coupler: "Coupler",
    diagnostics: Mapping[str, Any],
    *,
    first_step: int = 0,
) -> dict[str, xr.Dataset]:
    """Label a chunk's diagnostics, keeping every record.

    ``coupler.to_xarray(diagnostics, first_step=first_step)``, and nothing
    else. ``first_step`` is the coupled step the chunk's first record covers
    -- the ``step`` of the carry the chunk started from -- and passing it is
    what stops every chunk from being labelled with the first chunk's dates.

    This is the form anything that *inspects* a chunk must be given: it still
    holds the chunk's last record, and every value in it, which neither
    reduction in :func:`postprocess` promises to preserve. Reduce with
    :func:`postprocess_datasets` only what is written out.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the diagnostics came from.
    diagnostics : Mapping[str, Any]
        What the chunk's trajectory function returned.
    first_step : int
        The coupled step the first record of this chunk covers.

    Returns
    -------
    dict[str, xarray.Dataset]
        One unreduced dataset per component that writes output.

    """
    return coupler.to_xarray(dict(diagnostics), first_step=first_step)


def postprocess_datasets(
    datasets: Mapping[str, xr.Dataset],
    *,
    output_averages: bool = False,
    subsample: int = 1,
    first_step: int = 0,
    steps: int | None = None,
) -> dict[str, xr.Dataset]:
    """Apply :func:`postprocess` to every dataset of a chunk.

    The reductions are the same ones, applied with the same options to each
    component: a chunk is one interval of the run's clock, so a run whose
    output is chunk means wants them from every component that wrote any.

    ``first_step`` and ``steps`` describe the **chunk**, so they are one pair
    for all of its datasets even though the components need not record at the
    same rate: :func:`postprocess` reads each component's records-per-step off
    its own record count, and a component the workflow runs ``n`` times per
    coupled step keeps or drops all ``n`` of a step's records together with
    everyone else's.

    Nothing here copies. With neither reduction asked for this is the
    identity and the returned datasets **are** the given ones, so a caller
    holding both -- :func:`jem.driver.run_chunked` holds the chunk for its
    health gate and the reduction for the file -- must treat them as one
    object and not modify either in place.

    Parameters
    ----------
    datasets : Mapping[str, xarray.Dataset]
        A chunk's datasets, keyed by component name -- from
        :func:`chunk_datasets`.
    output_averages : bool
        Passed to :func:`postprocess`.
    subsample : int
        Passed to :func:`postprocess`.
    first_step : int
        The coupled step the chunk starts at; passed to :func:`postprocess`,
        where it places the chunk in the run's stride period.
    steps : int, optional
        Coupled steps in the chunk; passed to :func:`postprocess`.

    Returns
    -------
    dict[str, xarray.Dataset]
        One postprocessed dataset per entry, under the same names.

    """
    return {
        name: postprocess(
            dataset,
            output_averages=output_averages,
            subsample=subsample,
            first_step=first_step,
            steps=steps,
        )
        for name, dataset in datasets.items()
    }


def datasets_for_chunk(
    coupler: "Coupler",
    diagnostics: Mapping[str, Any],
    *,
    first_step: int = 0,
    steps: int | None = None,
    output_averages: bool = False,
    subsample: int = 1,
) -> dict[str, xr.Dataset]:
    """Label a chunk's diagnostics and postprocess them, in one call.

    :func:`chunk_datasets` followed by :func:`postprocess_datasets`, for a
    caller that wants only the reduced output.

    A caller that also inspects the chunk -- a health gate, a diagnostic --
    should call the two separately and inspect the unreduced datasets, as
    :func:`jem.driver.run_chunked` does; see the module docstring for why
    reduced output is the wrong thing to judge a chunk by.

    Parameters
    ----------
    coupler : jem.base.coupler.Coupler
        The coupled model the diagnostics came from.
    diagnostics : Mapping[str, Any]
        What the chunk's trajectory function returned.
    first_step : int
        The coupled step the first record of this chunk covers. It labels the
        records *and* places the chunk in the run's ``subsample`` period, so
        one number does both and the two cannot disagree.
    steps : int, optional
        Coupled steps in the chunk; passed to :func:`postprocess_datasets`.
        Defaults to one coupled step per record.
    output_averages : bool
        Passed to :func:`postprocess`.
    subsample : int
        Passed to :func:`postprocess`.

    Returns
    -------
    dict[str, xarray.Dataset]
        One postprocessed dataset per component that writes output.

    """
    return postprocess_datasets(
        chunk_datasets(coupler, diagnostics, first_step=first_step),
        output_averages=output_averages,
        subsample=subsample,
        first_step=first_step,
        steps=steps,
    )
