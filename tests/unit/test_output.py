"""Tests for ``jem.output`` -- reducing a chunk of a run and writing it out.

:func:`~jem.output.postprocess` is exercised on hand-built datasets, where the
records are small integers and the answer can be written down; the two-slab
coupler at the end is there for :func:`~jem.output.datasets_for_chunk`, whose
whole job is to sit on top of a real ``Coupler.to_xarray``.
"""

import logging
import pathlib

import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr

from jem.base.coupler import Coupler
from jem.components.slab import SlabOceanModel, SlabSeaiceModel
from jem.exchangers import default_exchangers
from jem.output import (
    chunk_datasets,
    datasets_for_chunk,
    output_file_name,
    output_file_step,
    postprocess,
    postprocess_datasets,
    write_chunk,
)
from tests.unit.slab_test_utils import make_grid

START_DATE = jdt.to_datetime("2001-01-01")
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")


def simple_dataset(n_records: int = 6) -> xr.Dataset:
    """Return a dataset shaped like a component's chunk of output."""
    values = np.arange(n_records * 2, dtype=np.float64).reshape(n_records, 2)
    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lon"),
                values,
                {"units": "K", "jem_role": "state"},
            ),
            "mask": (("lon",), np.array([0.0, 1.0]), {"units": "1"}),
        },
        coords={
            "time": (
                "time",
                np.arange(n_records).astype("datetime64[D]").astype("datetime64[ns]"),
                {"standard_name": "time"},
            ),
            "lon": ("lon", np.array([0.0, 180.0])),
        },
        attrs={"title": "a chunk"},
    )


def bounded_dataset(start: str, n_records: int = 3) -> xr.Dataset:
    """Return a dataset shaped like a JCM component's chunk, ``time_bounds`` included.

    One daily record per day from ``start``, labelled at each interval's
    midpoint and carrying the CF ``time_bounds`` jax-gcm's own
    ``ModelPredictions.to_xarray`` publishes for an averaged trajectory --
    exactly the shape :func:`postprocess`'s ``time_bounds``-aware branch reads.
    """
    starts = np.datetime64(start) + np.arange(n_records).astype("timedelta64[D]")
    ends = starts + np.timedelta64(1, "D")
    midpoints = (
        starts.astype("datetime64[ms]")
        + (ends - starts).astype("timedelta64[ms]") // 2
    )
    bounds = np.stack(
        [starts.astype("datetime64[ms]"), ends.astype("datetime64[ms]")], axis=1
    )
    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lon"),
                np.arange(n_records * 2, dtype=np.float64).reshape(n_records, 2),
                {"units": "K", "jem_role": "state"},
            ),
            "time_bounds": (
                ("time", "bounds"),
                bounds,
                {"long_name": "time interval bounds"},
            ),
        },
        coords={
            "time": ("time", midpoints, {"standard_name": "time", "bounds": "time_bounds"}),
            "lon": ("lon", np.array([0.0, 180.0])),
        },
        attrs={"title": "a chunk"},
    )


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------


def test_postprocess_without_reductions_returns_the_dataset_unchanged():
    dataset = simple_dataset()
    assert postprocess(dataset) is dataset


def test_postprocess_subsample_keeps_every_kth_record():
    dataset = simple_dataset(6)
    thinned = postprocess(dataset, subsample=2)

    assert thinned.sizes["time"] == 3
    np.testing.assert_array_equal(
        thinned["time"].values, dataset["time"].values[::2]
    )
    np.testing.assert_array_equal(
        thinned["temperature"].values, dataset["temperature"].values[::2]
    )
    # A variable without a time axis is untouched, and the metadata survives.
    np.testing.assert_array_equal(thinned["mask"].values, dataset["mask"].values)
    assert thinned["temperature"].attrs["jem_role"] == "state"
    assert thinned.attrs == {"title": "a chunk"}


def test_postprocess_averages_the_chunk_and_labels_it_at_its_midpoint():
    dataset = simple_dataset(4)
    averaged = postprocess(dataset, output_averages=True)

    assert averaged.sizes["time"] == 1
    # This dataset carries no `time_bounds` (see `simple_dataset`), so there
    # is no bound to read the chunk's own interval from directly -- but its
    # midpoint is still exact, as the average of the chunk's first and last
    # record's own labels (days 0 and 3 here, so day 1.5 at noon).
    assert averaged["time"].values[0] == np.datetime64("1970-01-02T12:00", "ms")
    np.testing.assert_allclose(
        averaged["temperature"].values[0],
        dataset["temperature"].values.mean(axis=0),
    )
    assert averaged["temperature"].attrs["cell_methods"] == "time: mean"
    assert averaged["temperature"].attrs["units"] == "K"
    assert averaged["temperature"].attrs["jem_role"] == "state"
    assert averaged["time"].attrs == {"standard_name": "time"}
    assert averaged.attrs == {"title": "a chunk"}
    # The time-independent variable is carried through, not averaged into a
    # one-record time series.
    assert "time" not in averaged["mask"].dims
    np.testing.assert_array_equal(averaged["mask"].values, dataset["mask"].values)
    np.testing.assert_array_equal(averaged["lon"].values, dataset["lon"].values)


def test_postprocess_appends_to_an_existing_cell_methods():
    dataset = simple_dataset(3)
    dataset["temperature"].attrs["cell_methods"] = "area: mean"
    averaged = postprocess(dataset, output_averages=True)
    assert averaged["temperature"].attrs["cell_methods"] == "area: mean time: mean"


def test_postprocess_does_not_duplicate_an_identical_cell_method_already_present():
    """JCM's own per-step "time: mean" must not double up.

    JCM writes each of its own per-coupling-step records already averaged
    over the physics sub-steps (``run.output_averages``, see the module
    docstring), so by the time :func:`postprocess` sees a JCM variable it
    *already* carries ``cell_methods = "time: mean"``. Appending the bare,
    unannotated method text a second time would give
    ``"time: mean time: mean"``, which is what shows up on every atm
    variable of a real coupled earth-slab run if left unguarded. Two
    identical, unannotated entries carry no more information than one, so
    the second is a duplicate rather than a description of a genuine
    further reduction -- see :func:`~jem.output._with_cell_method`'s own
    docstring.
    """
    dataset = simple_dataset(3)
    dataset["temperature"].attrs["cell_methods"] = "time: mean"
    averaged = postprocess(dataset, output_averages=True)
    assert averaged["temperature"].attrs["cell_methods"] == "time: mean"


def test_postprocess_annotates_cell_methods_honestly_when_subsampled():
    """A subsampled mean's ``cell_methods`` says so.

    With ``subsample > 1`` the recorded label (and, for a dataset that has
    one, ``time_bounds``) still spans the chunk's *whole* interval (see
    ``test_postprocess_time_bounds_survive_a_subsample_then_average``) while
    the mean itself is only over the records the stride *kept* -- so a bare,
    unannotated "time: mean" would overstate what interval actually informed
    the number (the metadata would read as if every record of the chunk had
    been averaged). :func:`postprocess` instead appends an explicit CF
    comment saying so -- one that is valid CF (see
    ``test_postprocess_subsample_comment_is_valid_cf``) and does not assume
    this dataset has a ``time_bounds`` to point at, since ``simple_dataset``
    (every non-JCM component) never carries one.
    """
    dataset = simple_dataset(4)
    averaged = postprocess(dataset, output_averages=True, subsample=2)
    cell_methods = averaged["temperature"].attrs["cell_methods"]
    assert cell_methods.startswith("time: mean (comment:")
    assert "subsampl" in cell_methods or "every" in cell_methods
    assert "chunk" in cell_methods

    # This annotated text is never textually identical to a bare, upstream
    # "time: mean" -- so unlike the plain (subsample=1) case, both entries
    # survive: one describes JCM's own per-step average, the other this
    # subsampled chunk average, and they say different things.
    dataset2 = simple_dataset(4)
    dataset2["temperature"].attrs["cell_methods"] = "time: mean"
    averaged2 = postprocess(dataset2, output_averages=True, subsample=2)
    assert averaged2["temperature"].attrs["cell_methods"] == f"time: mean {cell_methods}"


def test_postprocess_subsample_comment_is_valid_cf():
    """The subsample comment must parse as valid CF.

    CF's ``(comment: ...)`` extra-info block does not nest -- a stray, inner
    ``(``/``)`` pair (e.g. "step(s)") would close the block early, so
    anything written after it would not actually be inside the comment any
    CF reader would parse. The comment also must not say "the label/
    time_bounds above", which points at nothing meaningful from inside an
    attribute string, and must not claim a ``time_bounds`` exists on every
    dataset, since a non-JCM component's (``simple_dataset``'s) never does.
    """
    dataset = simple_dataset(4)
    averaged = postprocess(dataset, output_averages=True, subsample=2)
    cell_methods = averaged["temperature"].attrs["cell_methods"]
    # Exactly one parenthesised extra-info block: a second, inner pair would
    # close CF's own block early rather than nest inside it.
    assert cell_methods.count("(") == 1
    assert cell_methods.count(")") == 1
    assert cell_methods.endswith(")")
    assert "above" not in cell_methods
    assert "time_bounds" not in cell_methods


def test_postprocess_averages_time_bounds_correctly():
    """A `time_bounds`-carrying chunk gets the chunk's own exact midpoint and bound.

    Averaging `time_bounds` like an ordinary variable (mean of the three
    1-day bounds) and labelling with the chunk's *last* record's own
    midpoint would give a real 3-day chunk starting 2000-02-02 the label
    ``02-04T12:00`` (neither the correct midpoint, ``02-03T12:00``, nor the
    end, ``02-05``) with ``time_bounds=[02-03, 02-04]`` -- a false one-day
    interval for what is actually a 3-day mean.
    """
    dataset = bounded_dataset("2000-02-02", n_records=3)
    averaged = postprocess(dataset, output_averages=True)

    assert averaged.sizes["time"] == 1
    assert averaged["time"].values[0] == np.datetime64("2000-02-03T12:00", "ms")
    np.testing.assert_array_equal(
        averaged["time_bounds"].values,
        np.array(
            [[np.datetime64("2000-02-02", "ms"), np.datetime64("2000-02-05", "ms")]]
        ),
    )
    # `time_bounds` is a bound, not a sampled quantity: no `cell_methods`.
    assert "cell_methods" not in averaged["time_bounds"].attrs
    assert averaged["time_bounds"].attrs["long_name"] == "time interval bounds"
    # The data variable is still the plain mean, and still stamped.
    np.testing.assert_allclose(
        averaged["temperature"].values[0],
        dataset["temperature"].values.mean(axis=0),
    )
    assert averaged["temperature"].attrs["cell_methods"] == "time: mean"


def test_postprocess_time_bounds_survive_a_subsample_then_average():
    """The chunk's own bounds, not the kept records', even when subsample thins."""
    dataset = bounded_dataset("2000-02-02", n_records=4)
    averaged = postprocess(dataset, output_averages=True, subsample=2)

    assert averaged.sizes["time"] == 1
    # The chunk is still all 4 days, 2 Feb through 6 Feb, whether or not the
    # stride kept every record of it.
    np.testing.assert_array_equal(
        averaged["time_bounds"].values,
        np.array(
            [[np.datetime64("2000-02-02", "ms"), np.datetime64("2000-02-06", "ms")]]
        ),
    )
    assert averaged["time"].values[0] == np.datetime64("2000-02-04T00:00", "ms")
    np.testing.assert_allclose(
        averaged["temperature"].values[0],
        dataset["temperature"].values[::2].mean(axis=0),
    )


def test_postprocess_with_no_time_bounds_still_finds_the_chunk_midpoint():
    """A dataset with no `time_bounds` gets the chunk's true midpoint too.

    Labelling with the chunk's last record's own label instead would be an
    inconsistent approximation, since jax-gcm PR 878 made every record's own
    label a midpoint rather than an end-of-interval instant. The chunk's
    true midpoint is exact here too: it is the average of
    the first and last record's own midpoint labels, which equals the
    chunk's midpoint for any equal-length, contiguous run of records (see
    :func:`~jem.output.postprocess`'s body for the derivation) -- it does not
    depend on `dt`, so it is checked here with records at *uneven* spacing
    (day 0 and day 10) to show the formula does not assume one.

    The end-to-end version of this -- two real components, one with
    `time_bounds` and one without, whose averaged chunks must carry the
    *same* label so `xr.merge(..., join="exact")` succeeds -- is
    `test_postprocess_averages_jcm_and_slab_chunks_to_matching_labels` in
    `tests/unit/test_coupled.py`, which is where a real JCM model lives.
    """
    dataset = simple_dataset(2)
    dataset = dataset.assign_coords(
        time=("time", np.array(["1970-01-01", "1970-01-11"], dtype="datetime64[ns]"))
    )
    assert "time_bounds" not in dataset.variables
    averaged = postprocess(dataset, output_averages=True)
    assert averaged["time"].values[0] == np.datetime64("1970-01-06", "ms")


def test_postprocess_composes_subsample_then_average():
    """Both together average the retained records, in that order.

    The label stays the **chunk's** own midpoint (days 0 through 5, so day
    2.5) even though the stride dropped some of those records from the mean:
    it says which interval the mean covers, which is the whole chunk, so a
    run that sets both still writes one mean per chunk evenly spaced with the
    chunks.
    """
    dataset = simple_dataset(6)
    result = postprocess(dataset, output_averages=True, subsample=2)

    assert result.sizes["time"] == 1
    np.testing.assert_allclose(
        result["temperature"].values[0],
        dataset["temperature"].values[::2].mean(axis=0),
    )
    assert result["time"].values[0] == np.datetime64("1970-01-03T12:00", "ms")


def test_postprocess_stride_counts_coupled_steps_of_the_whole_run():
    """A chunk part-way through the stride continues its phase, not restarts it.

    Six records in two three-step chunks: the stride is in coupled steps of
    the run, so the pair keeps global steps 0, 2 and 4 -- the records an
    unchunked run keeps -- and not 0, 2, 3, 5, which is what a slice that
    restarts at each chunk's first record gives (an irregular cadence, and
    more output than was asked for).
    """
    whole = simple_dataset(6)
    chunks = [
        postprocess(
            whole.isel(time=slice(0, 3)), subsample=2, first_step=0, steps=3
        ),
        postprocess(
            whole.isel(time=slice(3, 6)), subsample=2, first_step=3, steps=3
        ),
    ]

    kept_times = np.concatenate([chunk["time"].values for chunk in chunks])
    np.testing.assert_array_equal(kept_times, whole["time"].values[[0, 2, 4]])
    np.testing.assert_array_equal(
        kept_times, postprocess(whole, subsample=2)["time"].values
    )
    np.testing.assert_array_equal(
        np.concatenate([chunk["temperature"].values for chunk in chunks]),
        whole["temperature"].values[::2],
    )


def test_postprocess_keeps_every_record_of_a_kept_coupled_step():
    """A component recording twice a step is thinned by step, not by record."""
    dataset = simple_dataset(6)  # three coupled steps, two records each

    first = postprocess(dataset, subsample=2, first_step=0, steps=3)
    # Steps 0 and 2 are kept, both of each one's records with them.
    np.testing.assert_array_equal(
        first["temperature"].values, dataset["temperature"].values[[0, 1, 4, 5]]
    )

    # Steps 3, 4 and 5: only step 4 is on the stride, and it keeps both.
    second = postprocess(dataset, subsample=2, first_step=3, steps=3)
    np.testing.assert_array_equal(
        second["temperature"].values, dataset["temperature"].values[[2, 3]]
    )


def test_postprocess_needs_a_whole_number_of_records_per_coupled_step():
    """Without that, the coupled step a record belongs to is undefined."""
    with pytest.raises(ValueError, match="cannot hold 5 record"):
        postprocess(simple_dataset(5), subsample=2, steps=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"steps": 0}, "steps must be a positive"), ({"first_step": -1}, "negative")],
)
def test_postprocess_rejects_a_nonsensical_chunk(kwargs, message):
    with pytest.raises(ValueError, match=message):
        postprocess(simple_dataset(6), subsample=2, **kwargs)


def test_postprocess_refuses_to_average_a_chunk_that_does_not_divide_by_steps():
    """Averaging alone must be checked too, not just the stride.

    ``_kept_records`` already refuses this when ``subsample > 1`` triggers
    it; ``output_averages`` on its own (``subsample=1``, the default) must
    not skip that check and label the chunk anyway, since the chunk-mean
    derivation equally assumes a whole number of equal-length coupled
    steps.
    """
    with pytest.raises(ValueError, match="cannot hold 5 record"):
        postprocess(simple_dataset(5), output_averages=True, steps=3)


def test_postprocess_refuses_to_average_records_with_an_irregular_gap():
    """A non-contiguous set of records must not get a silently wrong label.

    ``postprocess``'s chunk label -- the average of the first and last
    record's own midpoint -- is only exact for a contiguous run of
    equal-length intervals (see its own docstring's derivation). Widening the
    last gap here breaks that assumption without changing the record count,
    so nothing about ``steps`` would have caught it.
    """
    dataset = simple_dataset(4)  # days 0, 1, 2, 3, evenly spaced
    times = dataset["time"].values.copy()
    times[-1] = np.datetime64("1970-01-11")  # days 0, 1, 2, 10
    dataset = dataset.assign_coords(time=("time", times))
    with pytest.raises(ValueError, match="not evenly spaced"):
        postprocess(dataset, output_averages=True)


def test_postprocess_refuses_to_average_time_bounds_with_a_gap():
    """The ``time_bounds`` variant: a gap between bounded intervals.

    Unlike the label-only case above, a ``time_bounds``-carrying dataset
    (JCM's) has an explicit per-record interval to check for contiguity
    directly -- record k's own bound must end exactly where record k+1's
    begins.
    """
    dataset = bounded_dataset("2000-02-02", n_records=3)
    bounds = dataset["time_bounds"].values.copy()
    # Push the last record's interval a day later, opening a one-day gap
    # between it and the record before it.
    bounds[-1] += np.timedelta64(1, "D")
    dataset = dataset.copy()
    dataset["time_bounds"] = (("time", "bounds"), bounds, dataset["time_bounds"].attrs)
    with pytest.raises(ValueError, match="not contiguous"):
        postprocess(dataset, output_averages=True)


def test_postprocess_keeps_nothing_from_a_chunk_the_stride_skips():
    """A chunk with no step on the stride is skipped rather than bent.

    Keeping a record here anyway -- the chunk's first, say -- is exactly what
    would break the cadence the stride promises; no records is the honest
    answer, `output_averages` has nothing to average in it, and
    :func:`~jem.output.write_chunk` writes no file for it.
    """
    dataset = simple_dataset(3)  # coupled steps 6, 7, 8 of the run

    assert postprocess(dataset, subsample=5, first_step=6, steps=3).sizes["time"] == 0
    assert (
        postprocess(
            dataset, subsample=5, first_step=6, steps=3, output_averages=True
        ).sizes["time"]
        == 0
    )
    # The chunk before it does hold a multiple of the stride -- step 5 -- and
    # keeps that one record.
    kept = postprocess(dataset, subsample=5, first_step=3, steps=3)
    np.testing.assert_array_equal(kept["time"].values, dataset["time"].values[[2]])


@pytest.mark.parametrize("subsample", [0, -1, 1.5, True])
def test_postprocess_rejects_a_bad_subsample(subsample):
    with pytest.raises(ValueError, match="positive integer"):
        postprocess(simple_dataset(), subsample=subsample)


def test_postprocess_needs_a_time_dimension_to_reduce():
    dataset = simple_dataset().isel(time=0, drop=True)
    with pytest.raises(ValueError, match="no 'time' dimension"):
        postprocess(dataset, output_averages=True)


# ---------------------------------------------------------------------------
# The file name, and reading it back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["ocn", "sea-ice", "atm2", "sea ice", "a-00000001"]
)
@pytest.mark.parametrize("step", [0, 12, 99999999, 123456789])
def test_a_written_name_reads_back_as_the_step_it_was_written_for(name, step):
    """Every name the writer produces is recognised, with its step, by the reader.

    The two are used at opposite ends of a run -- one names a chunk's file,
    the other decides what an existing file in the directory is -- so the only
    thing that keeps a resume honest about which files are its own is that
    they are exact inverses. The awkward cases are the point: a name with a
    hyphen or a space in it (a component name is whatever a user called it),
    and a step past the eight digits of the padding, where the name is a digit
    longer.
    """
    filename = output_file_name(name, step)

    assert output_file_step(filename, [name]) == step
    # A path is accepted as readily as a bare name, and only the name is read.
    assert output_file_step(pathlib.Path("/somewhere/else") / filename, [name]) == step
    # A name this run does not write is not this run's file, whatever the step.
    assert output_file_step(filename, ["something_else"]) is None


@pytest.mark.parametrize(
    "filename",
    [
        # The padding is eight digits, so a ninth here is a different name --
        # not step 12 written by anything this module wrote.
        "ocn-000000012.nc",
        # A different component.
        "ocean-00000004.nc",
        # No step at all.
        "ocn.nc",
        # One digit short of the padding.
        "ocn-0000004.nc",
        # netCDF, but not the extension `write_chunk` writes.
        "ocn-00000004.nc4",
    ],
)
def test_a_name_this_module_would_not_have_written_is_not_recognised(filename):
    """Recognition is by rebuilding the name, so near misses are misses.

    Anything the check cannot prove is a run's own output has to come back as
    "not ours": the driver uses it to decide which files in a directory a
    resume is responsible for, and a false positive there is a run refusing to
    start because of a stranger's file.
    """
    assert output_file_step(filename, ["ocn"]) is None


def test_a_component_name_that_ends_like_a_step_is_read_as_its_own_name():
    """A hyphen and digits inside a component name do not become the step.

    The name is whatever the component was registered as, so it can end in
    something that looks like a padded step. The file is that component's,
    starting at step 4 -- and it is emphatically not component `a`'s, which is
    what a lazy split on the first hyphen would have made it.
    """
    assert output_file_step("a-00000001-00000004.nc", ["a-00000001"]) == 4
    assert output_file_step("a-00000001-00000004.nc", ["a"]) is None


# ---------------------------------------------------------------------------
# write_chunk
# ---------------------------------------------------------------------------


def test_write_chunk_names_files_by_component_and_first_step(tmp_path):
    """The name is the component and the coupled step the chunk starts at."""
    datasets = {"ocn": simple_dataset(3), "atm": simple_dataset(3)}
    paths = write_chunk(datasets, tmp_path / "output", 7)

    assert [path.name for path in paths] == ["atm-00000007.nc", "ocn-00000007.nc"]
    assert all(path.parent == tmp_path / "output" for path in paths)
    assert all(path.exists() for path in paths)


def test_write_chunk_names_sort_in_run_order(tmp_path):
    """Zero padding is what makes a directory listing read in run order.

    Unpadded, step 10 would sort before step 9, and a user reading the files
    in listing order would read the run out of sequence.
    """
    for first_step in (0, 9, 10, 1000):
        write_chunk({"ocn": simple_dataset(2)}, tmp_path, first_step)
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == [
        "ocn-00000000.nc", "ocn-00000009.nc", "ocn-00000010.nc", "ocn-00001000.nc",
    ]


def test_write_chunk_round_trips_through_open_dataset(tmp_path):
    dataset = postprocess(simple_dataset(4), output_averages=True)
    (path,) = write_chunk({"ocn": dataset}, tmp_path, 0)

    with xr.open_dataset(path) as written:
        np.testing.assert_allclose(
            written["temperature"].values, dataset["temperature"].values
        )
        np.testing.assert_array_equal(
            written["time"].values, dataset["time"].values
        )
        assert written["temperature"].attrs["cell_methods"] == "time: mean"
        assert written["temperature"].attrs["jem_role"] == "state"
        assert written.attrs["title"] == "a chunk"


def test_write_chunk_creates_the_directory_and_keeps_chunks_apart(tmp_path):
    directory = tmp_path / "deeply" / "nested"
    first = write_chunk({"ocn": simple_dataset(2)}, directory, 0)
    second = write_chunk({"ocn": simple_dataset(2)}, directory, 2)

    assert first != second
    assert sorted(p.name for p in directory.iterdir()) == [
        "ocn-00000000.nc", "ocn-00000002.nc",
    ]


def test_write_chunk_warns_when_it_overwrites(tmp_path, caplog):
    """Writing over an existing file is allowed, and said out loud.

    A resumed run never lands on a step it has already written, so a
    collision means a rerun into the same output directory -- deliberate,
    but worth knowing about when the two runs were not configured the same.
    """
    write_chunk({"ocn": simple_dataset(2)}, tmp_path, 4)
    with caplog.at_level(logging.WARNING, logger="jem.output"):
        write_chunk({"ocn": simple_dataset(3)}, tmp_path, 4)
    assert "already exists and is being overwritten" in caplog.text

    # The second write won, so the file holds the second dataset.
    with xr.open_dataset(tmp_path / "ocn-00000004.nc") as written:
        assert written.sizes["time"] == 3


def test_write_chunk_does_not_warn_on_a_first_write(tmp_path, caplog):
    """A run writing into an empty directory says nothing about overwriting."""
    with caplog.at_level(logging.WARNING, logger="jem.output"):
        write_chunk({"ocn": simple_dataset(2)}, tmp_path, 0)
    assert caplog.text == ""


def test_write_chunk_makes_a_name_safe_for_the_filesystem(tmp_path):
    (path,) = write_chunk({"fast/atm": simple_dataset(2)}, tmp_path, 3)
    assert path.name == "fast_atm-00000003.nc"
    assert path.parent == tmp_path


def test_write_chunk_refuses_names_that_would_collide(tmp_path):
    with pytest.raises(ValueError, match="rename one of the components"):
        write_chunk(
            {"a/b": simple_dataset(2), "a:b": simple_dataset(2)}, tmp_path, 0
        )


def test_write_chunk_writes_no_file_for_a_dataset_with_no_records(
    tmp_path, caplog
):
    """A chunk the stride keeps nothing from is skipped, not written empty.

    `xr.open_mfdataset` -- how a run's output is read back -- refuses a
    zero-length dimension, so one empty file would cost the reader the whole
    directory; and an empty file is not the chunk's output, it is the absence
    of any.
    """
    empty = postprocess(simple_dataset(3), subsample=5, first_step=6, steps=3)
    with caplog.at_level(logging.INFO, logger="jem.output"):
        written = write_chunk({"ocn": empty, "seaice": simple_dataset(2)}, tmp_path, 6)

    assert [path.name for path in written] == ["seaice-00000006.nc"]
    assert not (tmp_path / "ocn-00000006.nc").exists()
    assert "'ocn' holds no records, so ocn-00000006.nc was not written" in caplog.text


def test_write_chunk_removes_a_file_at_the_name_of_an_empty_dataset(
    tmp_path, caplog
):
    """Leave nothing at a name whose chunk this pass keeps no record for.

    The name belongs to this chunk of this run, and an earlier pass's file at
    it holds records this one has replaced with none -- which a rechunked
    resume can turn into a duplicate of a record written under another name.
    Only that one name is touched: a non-empty dataset is overwritten as ever.
    """
    write_chunk({"ocn": simple_dataset(2), "seaice": simple_dataset(2)}, tmp_path, 6)
    empty = postprocess(simple_dataset(3), subsample=5, first_step=6, steps=3)

    with caplog.at_level(logging.INFO, logger="jem.output"):
        written = write_chunk({"ocn": empty, "seaice": simple_dataset(3)}, tmp_path, 6)

    assert [path.name for path in written] == ["seaice-00000006.nc"]
    assert not (tmp_path / "ocn-00000006.nc").exists()
    assert "the file already there was removed" in caplog.text
    # The component that did have records still simply overwrote its file.
    with xr.open_dataset(tmp_path / "seaice-00000006.nc") as rewritten:
        assert rewritten.sizes["time"] == 3


def test_write_chunk_writes_a_dataset_with_no_time_dimension(tmp_path):
    """"Nothing to write" is an empty time axis, not the absence of one."""
    timeless = simple_dataset(2).isel(time=0, drop=True)
    (path,) = write_chunk({"ocn": timeless}, tmp_path, 0)
    with xr.open_dataset(path) as written:
        assert "time" not in written.dims


def test_write_chunk_still_refuses_colliding_names_when_one_is_empty(tmp_path):
    """The name check is about names, so an empty dataset does not dodge it."""
    empty = postprocess(simple_dataset(3), subsample=5, first_step=6, steps=3)
    with pytest.raises(ValueError, match="rename one of the components"):
        write_chunk({"a/b": empty, "a:b": simple_dataset(2)}, tmp_path, 0)


def test_write_chunk_rejects_a_negative_first_step(tmp_path):
    with pytest.raises(ValueError, match="must not be negative"):
        write_chunk({"ocn": simple_dataset(2)}, tmp_path, -1)


# ---------------------------------------------------------------------------
# datasets_for_chunk, on a real coupler
# ---------------------------------------------------------------------------


@pytest.fixture
def two_slab_coupler():
    """Return an ocean and a sea-ice slab on the 4x3 grid, default wiring."""
    grid = make_grid()
    components = {
        "ocn": SlabOceanModel(grid),
        "seaice": SlabSeaiceModel(grid, name="seaice"),
    }
    return Coupler(
        components,
        default_exchangers(components),
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )


def test_chunk_datasets_and_postprocess_datasets_are_the_two_halves(
    two_slab_coupler,
):
    """The labelling keeps every record; the reduction is a separate choice.

    `run_chunked` relies on being able to take the two separately -- it shows
    the health gate the unreduced chunk and writes the reduced one -- so the
    split is part of the contract, not an implementation detail of
    `datasets_for_chunk`.
    """
    run = two_slab_coupler.generate_trajectory_function(4)
    _, diagnostics = run(two_slab_coupler.initialize())

    labelled = chunk_datasets(two_slab_coupler, diagnostics, first_step=0)
    assert set(labelled) == {"ocn", "seaice"}
    assert all(dataset.sizes["time"] == 4 for dataset in labelled.values())

    reduced = postprocess_datasets(labelled, output_averages=True, subsample=2)
    assert all(dataset.sizes["time"] == 1 for dataset in reduced.values())
    # The reduction is not in place: the chunk it was given still has all
    # four records for whoever else is looking at them.
    assert all(dataset.sizes["time"] == 4 for dataset in labelled.values())

    composed = datasets_for_chunk(
        two_slab_coupler, diagnostics, first_step=0, output_averages=True,
        subsample=2,
    )
    for name, dataset in composed.items():
        np.testing.assert_array_equal(
            dataset["time"].values, reduced[name]["time"].values
        )


def test_datasets_for_chunk_labels_and_postprocesses(two_slab_coupler, tmp_path):
    """One call per chunk: label the records, then reduce them."""
    carry = two_slab_coupler.initialize()
    run = two_slab_coupler.generate_trajectory_function(4)
    carry, diagnostics = run(carry)

    datasets = datasets_for_chunk(two_slab_coupler, diagnostics, first_step=0)
    assert set(datasets) == {"ocn", "seaice"}
    assert all(dataset.sizes["time"] == 4 for dataset in datasets.values())

    thinned = datasets_for_chunk(
        two_slab_coupler, diagnostics, first_step=0, subsample=2
    )
    assert all(dataset.sizes["time"] == 2 for dataset in thinned.values())

    averaged = datasets_for_chunk(
        two_slab_coupler, diagnostics, first_step=0, output_averages=True
    )
    for name, dataset in averaged.items():
        assert dataset.sizes["time"] == 1, name
        # The chunk's own true midpoint: the average of its first and last
        # record's own midpoint labels (no `time_bounds` on a slab dataset,
        # so this is the exact-but-bound-free branch of `postprocess`).
        first_ms = datasets[name]["time"].values[0].astype("datetime64[ms]").astype("int64")
        last_ms = datasets[name]["time"].values[-1].astype("datetime64[ms]").astype("int64")
        expected = np.asarray(
            first_ms + (last_ms - first_ms) // 2, dtype="datetime64[ms]"
        )
        assert dataset["time"].values[0] == expected
        np.testing.assert_allclose(
            dataset["sea_surface_temperature"].values[0]
            if name == "ocn"
            else dataset["ice_thickness"].values[0],
            datasets[name][
                "sea_surface_temperature" if name == "ocn" else "ice_thickness"
            ].values.mean(axis=0),
            rtol=1e-6,
        )


@pytest.fixture
def sub_stepped_coupler():
    """Return the same pair with the ocean run twice per coupled step.

    The classic weaving, in miniature: one component on a faster clock inside
    the coupled step, so its chunk holds two records per step while the other
    holds one -- which is what makes the stride's unit (coupled steps, not
    records) observable.
    """
    grid = make_grid()
    components = {
        "ocn": SlabOceanModel(grid),
        "seaice": SlabSeaiceModel(grid, name="seaice"),
    }
    exchangers = default_exchangers(components)
    return Coupler(
        components,
        exchangers,
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=[list(exchangers), ["ocn"] * 2, "seaice"],
    )


def test_components_recording_at_different_rates_are_thinned_in_step(
    sub_stepped_coupler,
):
    """One `(first_step, steps)` pair thins every component by the same steps."""
    run = sub_stepped_coupler.generate_trajectory_function(3)
    _, diagnostics = run(sub_stepped_coupler.initialize())
    full = chunk_datasets(sub_stepped_coupler, diagnostics, first_step=0)
    assert full["ocn"].sizes["time"] == 6
    assert full["seaice"].sizes["time"] == 3

    reduced = postprocess_datasets(full, subsample=2, first_step=0, steps=3)

    # Coupled steps 0 and 2 are kept: both of the ocean's records for each of
    # them, and the sea ice's single record for each.
    np.testing.assert_array_equal(
        reduced["ocn"]["time"].values, full["ocn"]["time"].values[[0, 1, 4, 5]]
    )
    np.testing.assert_array_equal(
        reduced["seaice"]["time"].values, full["seaice"]["time"].values[[0, 2]]
    )


def test_datasets_for_chunk_carries_the_chunk_through_to_the_stride(
    two_slab_coupler,
):
    """`first_step` labels the records and places them in the run's stride."""
    run = two_slab_coupler.generate_trajectory_function(3)
    carry, first = run(two_slab_coupler.initialize())
    _, second = run(carry)

    kept = [
        datasets_for_chunk(two_slab_coupler, first, first_step=0, steps=3,
                           subsample=2)["ocn"],
        datasets_for_chunk(two_slab_coupler, second, first_step=3, steps=3,
                           subsample=2)["ocn"],
    ]
    # Midpoints, not ends (jax-gcm PR 878; see TimeAxis.datetimes): record
    # `step` covers `[start + step*day, start + (step+1)*day)` and is
    # labelled at `start + (step + 1/2)*day`.
    half_day = np.timedelta64(12, "h").astype("timedelta64[ms]")
    day = np.timedelta64(1, "D").astype("timedelta64[ms]")
    start = np.datetime64("2001-01-01", "ms")
    np.testing.assert_array_equal(
        np.concatenate([dataset["time"].values for dataset in kept]),
        np.array([start + step * day + half_day for step in (0, 2, 4)]),
    )


def test_datasets_for_chunk_labels_a_later_chunk_from_its_first_step(
    two_slab_coupler,
):
    """``first_step`` is what stops every chunk carrying the first one's dates."""
    run = two_slab_coupler.generate_trajectory_function(2)
    carry, first = run(two_slab_coupler.initialize())
    _, second = run(carry)

    first_chunk = datasets_for_chunk(two_slab_coupler, first, first_step=0)
    second_chunk = datasets_for_chunk(two_slab_coupler, second, first_step=2)

    day = np.timedelta64(1, "D").astype("timedelta64[ns]")
    np.testing.assert_array_equal(
        second_chunk["ocn"]["time"].values,
        first_chunk["ocn"]["time"].values + 2 * day,
    )


def test_a_chunked_run_writes_one_file_per_component_per_chunk(
    two_slab_coupler, tmp_path
):
    """The two halves of this module, used the way a run loop uses them."""
    run = two_slab_coupler.generate_trajectory_function(2)
    carry = two_slab_coupler.initialize()
    written = []
    for _ in range(2):
        first_step = int(carry.step)
        carry, diagnostics = run(carry)
        datasets = datasets_for_chunk(
            two_slab_coupler, diagnostics, first_step=first_step
        )
        written += write_chunk(datasets, tmp_path, first_step)

    assert sorted(path.name for path in written) == [
        "ocn-00000000.nc", "ocn-00000002.nc",
        "seaice-00000000.nc", "seaice-00000002.nc",
    ]
    with xr.open_mfdataset(
        [path for path in written if path.name.startswith("ocn")],
        combine="by_coords",
    ) as combined:
        assert combined.sizes["time"] == 4
