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
  ``<output_dir>/<component>-<first step>.nc``,
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

The **coupled step a chunk starts at** is what labels a file, rather than a
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

What ``output_averages`` means here
-----------------------------------
JCM's ``run.output_averages`` switches each saved record from an
instantaneous snapshot at the end of its save interval to the **mean over
that interval**, still labelled at its end (``jcm.model``'s averaged outer
step; ``jcm/config/run/default.yaml``). The coupler's records are already one
per coupling step, so the same idea one level up: the coupler's output
interval is the **chunk**, and ``output_averages=True`` replaces a chunk's
records with their time mean -- one record, labelled with the chunk's last
time, carrying the CF ``cell_methods = "time: mean"`` that says so.

That keeps JCM's rule ("one record per output interval, the mean over it,
labelled at its end") rather than inventing a second meaning for the same
word, and it is the reduction a long run actually needs: monthly-mean output
from a 30-day chunk. Note that in a coupled run the atmosphere's per-step
records are *already* step means -- the JCM wrapper integrates each coupling
step with ``output_averages=True`` -- so averaging a chunk of them is the
chunk mean exactly, with no double counting.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from jem.base.coupler import Coupler

logger = logging.getLogger(__name__)

#: The dimension the reductions here act on. Every component's ``to_xarray``
#: labels its records with it (:class:`jem.base.component.TimeAxis`).
TIME_DIMENSION = "time"

#: CF cell method :func:`postprocess` stamps on an averaged variable.
TIME_MEAN_CELL_METHOD = f"{TIME_DIMENSION}: mean"

# Characters allowed in the component part of an output file name. A dataset
# name is a component name, which a user chooses, and a nested coupler's
# components arrive under their own names -- so nothing stops a name that is
# a path fragment or worse. Anything else becomes an underscore.
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")


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
    """Return the data variables that have a time dimension."""
    return [
        str(name)
        for name, variable in dataset.data_vars.items()
        if TIME_DIMENSION in variable.dims
    ]


def _with_cell_method(attrs: Mapping[str, Any], method: str) -> dict[str, Any]:
    """Return ``attrs`` with ``method`` appended to its CF ``cell_methods``."""
    updated = dict(attrs)
    existing = str(updated.get("cell_methods", "")).strip()
    # CF cell methods are a space-separated list, applied in order.
    updated["cell_methods"] = f"{existing} {method}".strip()
    return updated


def check_subsample(subsample: Any) -> int:
    """Return ``subsample`` if it is a valid record stride, else raise.

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
        record", which is not what anyone writing it meant.

    """
    if not isinstance(subsample, int) or isinstance(subsample, bool) or subsample < 1:
        raise ValueError(f"subsample must be a positive integer; got {subsample!r}.")
    return subsample


def postprocess(
    dataset: xr.Dataset, *, output_averages: bool = False, subsample: int = 1
) -> xr.Dataset:
    """Thin and/or average one component's records for a chunk.

    The two reductions compose in the order they are listed: ``subsample``
    chooses which records are kept, and ``output_averages`` then reduces
    whatever is left to its mean. Asking for both is legal but unusual --
    the mean is then over the retained records only, which is a worse
    estimate of the chunk mean than averaging all of them; normally a run
    sets one or the other.

    Variables without a time dimension (grid masks, layer thicknesses) are
    passed through untouched by both reductions.

    Parameters
    ----------
    dataset : xarray.Dataset
        One component's chunk of output, from ``Coupler.to_xarray``.
    output_averages : bool
        Replace the records with their time mean: one record, labelled with
        the last time in the chunk, with ``cell_methods = "time: mean"`` on
        every variable that was averaged. See the module docstring for why
        the chunk is the averaging interval.
    subsample : int
        Keep every ``subsample``-th record, starting with the first. ``1``
        (the default) keeps all of them.

    Returns
    -------
    xarray.Dataset

    Raises
    ------
    ValueError
        If ``subsample`` is not a positive integer, or a reduction was asked
        for and the dataset has no time dimension to reduce.

    """
    check_subsample(subsample)
    if subsample == 1 and not output_averages:
        return dataset
    if TIME_DIMENSION not in dataset.dims:
        raise ValueError(
            f"Cannot subsample or average a dataset with no {TIME_DIMENSION!r} "
            f"dimension (it has {sorted(map(str, dataset.dims))!r})."
        )

    if subsample > 1:
        dataset = dataset.isel({TIME_DIMENSION: slice(None, None, subsample)})
    if not output_averages:
        return dataset

    timed = _timed_variables(dataset)
    # `Dataset.mean` drops the dimension it reduces, so the label has to be
    # put back by hand: the chunk's last time, which is the end of the
    # interval the mean covers -- JCM's labelling convention, and the one
    # `TimeAxis.datetimes` already applied to the records being averaged.
    last_time = dataset[TIME_DIMENSION].isel({TIME_DIMENSION: slice(-1, None)})
    averaged = (
        dataset[timed]
        .mean(dim=TIME_DIMENSION, keep_attrs=True)
        .expand_dims({TIME_DIMENSION: last_time.values})
    )
    averaged[TIME_DIMENSION].attrs = dict(dataset[TIME_DIMENSION].attrs)
    for name in timed:
        averaged[name].attrs = _with_cell_method(
            dataset[name].attrs, TIME_MEAN_CELL_METHOD
        )
    for variable in dataset.data_vars:
        if str(variable) not in timed:
            averaged[variable] = dataset[variable]
    averaged.attrs = dict(dataset.attrs)
    return averaged


def write_chunk(
    datasets: Mapping[str, xr.Dataset], output_dir: Path | str, first_step: int
) -> list[Path]:
    """Write one chunk's datasets as netCDF and return the paths, in a stable order.

    One file per dataset, named ``<component>-<first_step:08d>.nc``: the
    component first so a directory listing groups a component's files
    together, and the coupled step the chunk starts at -- zero-padded so the
    listing sorts in run order -- second. That step is the run's own clock, so
    the name is unique however the run was chunked; see the module docstring
    for why a chunk index is not.

    An existing file is overwritten, with a warning. Writing a second run into
    a directory that already holds one is a deliberate act (a rerun, or a
    configuration changed and repeated), and refusing it would be worse than
    saying so. A resumed run normally does not collide, because it starts from
    the step its checkpoint holds; it does when the chunk it already wrote was
    never checkpointed -- either because the run was killed in between, or
    because the health gate rejected that chunk, which is deliberately not
    checkpointed -- so the warning reports a fact and does not assert which of
    them happened.

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
        test, see one order.

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
        path = directory / f"{_safe_name(name)}-{first_step:08d}.nc"
        if path in paths:
            raise ValueError(
                f"Datasets {paths[path]!r} and {name!r} would both be written to "
                f"{path.name!r}; rename one of the components."
            )
        paths[path] = name
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
) -> dict[str, xr.Dataset]:
    """Apply :func:`postprocess` to every dataset of a chunk.

    The reductions are the same ones, applied with the same options to each
    component: a chunk is one interval of the run's clock, so a run whose
    output is chunk means wants them from every component that wrote any.

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

    Returns
    -------
    dict[str, xarray.Dataset]
        One postprocessed dataset per entry, under the same names.

    """
    return {
        name: postprocess(
            dataset, output_averages=output_averages, subsample=subsample
        )
        for name, dataset in datasets.items()
    }


def datasets_for_chunk(
    coupler: "Coupler",
    diagnostics: Mapping[str, Any],
    *,
    first_step: int = 0,
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
        The coupled step the first record of this chunk covers.
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
    )
