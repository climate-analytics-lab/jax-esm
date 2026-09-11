"""Tests for ``jem.output`` -- reducing a chunk of a run and writing it out.

:func:`~jem.output.postprocess` is exercised on hand-built datasets, where the
records are small integers and the answer can be written down; the two-slab
coupler at the end is there for :func:`~jem.output.datasets_for_chunk`, whose
whole job is to sit on top of a real ``Coupler.to_xarray``.
"""

import logging

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


def test_postprocess_averages_the_chunk_and_labels_it_at_the_end():
    dataset = simple_dataset(4)
    averaged = postprocess(dataset, output_averages=True)

    assert averaged.sizes["time"] == 1
    # Labelled with the end of the interval it covers, as JCM labels an
    # averaged record.
    assert averaged["time"].values[0] == dataset["time"].values[-1]
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


def test_postprocess_composes_subsample_then_average():
    """Both together average the retained records, in that order."""
    dataset = simple_dataset(6)
    result = postprocess(dataset, output_averages=True, subsample=2)

    assert result.sizes["time"] == 1
    np.testing.assert_allclose(
        result["temperature"].values[0],
        dataset["temperature"].values[::2].mean(axis=0),
    )
    assert result["time"].values[0] == dataset["time"].values[4]


@pytest.mark.parametrize("subsample", [0, -1, 1.5, True])
def test_postprocess_rejects_a_bad_subsample(subsample):
    with pytest.raises(ValueError, match="positive integer"):
        postprocess(simple_dataset(), subsample=subsample)


def test_postprocess_needs_a_time_dimension_to_reduce():
    dataset = simple_dataset().isel(time=0, drop=True)
    with pytest.raises(ValueError, match="no 'time' dimension"):
        postprocess(dataset, output_averages=True)


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
        assert dataset["time"].values[0] == datasets[name]["time"].values[-1]
        np.testing.assert_allclose(
            dataset["sea_surface_temperature"].values[0]
            if name == "ocn"
            else dataset["ice_thickness"].values[0],
            datasets[name][
                "sea_surface_temperature" if name == "ocn" else "ice_thickness"
            ].values.mean(axis=0),
            rtol=1e-6,
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
