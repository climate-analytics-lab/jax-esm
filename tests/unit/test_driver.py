"""Tests for :mod:`jem.driver` -- the chunked run loop.

Almost everything here runs on a two-slab coupler with **no atmosphere**: a
4x3 grid, an ocean and a sea ice, coupled by the default wiring. That is
deliberate, and not only because it is fast. The loop's job is arithmetic on
the coupled step counter -- how many chunks, where each one starts, what is
left after a resume -- and a toy coupler makes the answers exact, so
``test_continuous_chunked_resumed_agree`` can insist on 1e-12 rather than on
"close enough for an atmosphere". The same comparison with the real
atmosphere is the slow test at the bottom, which is what proves the toy is
not the only thing this works on.
"""

import logging
import pathlib

import jax
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr

from jem.base.coupler import Coupler
from jem.checkpoint import CARRY_FILENAME
from jem.components.slab import SlabOceanModel, SlabSeaiceModel
from jem.driver import RunResult, default_health_check, run_chunked
from jem.exchangers import default_exchangers
from tests.unit.slab_test_utils import make_grid

START_DATE = jdt.to_datetime("2001-01-01")
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")


def two_slabs() -> Coupler:
    """Return an ocean and a sea ice on the 4x3 grid, with the default wiring."""
    grid = make_grid()
    components = {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)}
    return Coupler(
        components,
        default_exchangers(components),
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )


@pytest.fixture
def coupler() -> Coupler:
    return two_slabs()


def carry_leaves(carry):
    """Return the carry's leaves as numpy, for an exact comparison."""
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(carry)]


def assert_carries_agree(left, right, atol):
    """Assert two carries hold the same numbers to ``atol``, leaf by leaf."""
    left_leaves, right_leaves = carry_leaves(left), carry_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for index, (one, other) in enumerate(zip(left_leaves, right_leaves)):
        np.testing.assert_allclose(one, other, atol=atol, rtol=0, err_msg=f"leaf {index}")


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


def test_run_chunked_python_api(coupler, tmp_path):
    """Four days in two chunks: two files per component, and the clock advanced."""
    result = run_chunked(
        coupler, total_time="4 days", chunk="2 days", output_dir=tmp_path
    )

    assert isinstance(result, RunResult)
    assert result.completed
    assert result.steps_completed == 4
    assert int(result.final_carry.step) == 4

    # One file per component per chunk, named after the coupled step each
    # chunk starts at: steps 0 and 2 of a four-step run in two-day chunks.
    names = sorted(path.name for path in result.paths)
    assert names == [
        "ocn-00000000.nc", "ocn-00000002.nc",
        "seaice-00000000.nc", "seaice-00000002.nc",
    ]
    assert all(path.exists() for path in result.paths)

    # The default health check has no atmosphere to look at, so it abstains --
    # once per chunk, and without stopping the run.
    assert [report["skipped"] for report in result.reports] == ["no atmosphere"] * 2
    assert [report["chunk"] for report in result.reports] == [0, 1]

    # Each chunk is labelled with its own dates rather than the first chunk's.
    first = xr.open_dataset(tmp_path / "ocn-00000000.nc")
    second = xr.open_dataset(tmp_path / "ocn-00000002.nc")
    assert second["time"].values[0] > first["time"].values[-1]


def test_run_chunked_accepts_days_as_numbers(coupler, tmp_path):
    """A duration may be a number of days as well as a string."""
    result = run_chunked(coupler, total_time=2, chunk=1, output_dir=tmp_path)
    assert result.steps_completed == 2
    assert len(result.paths) == 4


def test_run_chunked_rejects_partial_chunk(coupler, tmp_path):
    """A run that is not a whole number of chunks is refused, naming both."""
    with pytest.raises(ValueError, match="not a whole number of chunks"):
        run_chunked(
            coupler, total_time="5 days", chunk="2 days", output_dir=tmp_path
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_time": "10 days", "chunk": "12 hours"}, "chunk="),
        ({"total_time": "36 hours", "chunk": "1 day"}, "total_time="),
    ],
)
def test_run_chunked_rejects_a_duration_that_is_not_whole_steps(
    coupler, tmp_path, kwargs, message
):
    """Both durations must be whole multiples of the coupling timestep."""
    with pytest.raises(ValueError, match=message):
        run_chunked(coupler, output_dir=tmp_path, **kwargs)


@pytest.mark.parametrize("subsample", [0, -1, 1.0, True])
def test_run_chunked_rejects_a_bad_subsample_before_it_integrates(
    coupler, tmp_path, subsample
):
    """`subsample` is checked up front, not after a chunk has been integrated.

    It is only read once a chunk's output exists, so a run configured with a
    nonsensical stride would otherwise compile a trajectory and integrate a
    whole chunk -- for an atmosphere, hours -- before raising. The trajectory
    factory is replaced with one that explodes, so the test fails if the
    check ever moves back behind it.
    """
    def must_not_be_called(iterations):
        raise AssertionError(
            f"a {iterations}-step trajectory was built despite subsample="
            f"{subsample!r}"
        )

    coupler.generate_trajectory_function = must_not_be_called
    with pytest.raises(ValueError, match="subsample must be a positive integer"):
        run_chunked(
            coupler, total_time="2 days", chunk="2 days",
            output_dir=tmp_path, subsample=subsample,
        )
    assert list(tmp_path.iterdir()) == []


def test_run_chunked_writes_nothing_when_the_run_is_already_done(coupler, tmp_path):
    """A carry already at `total_time` completes with no chunks and no files."""
    carry = coupler.initialize()
    run = coupler.generate_trajectory_function(3)
    carry, _ = run(carry)

    result = run_chunked(
        coupler,
        total_time="3 days",
        chunk="3 days",
        initial_carry=carry,
        output_dir=tmp_path,
    )
    assert result.completed
    assert result.steps_completed == 3
    assert result.paths == []
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The health gate
# ---------------------------------------------------------------------------


def test_unhealthy_chunk_stops_the_run(coupler, tmp_path):
    """The gate stops the run at the chunk it rejects, keeping that chunk's files."""
    def fails_on_the_second_chunk(datasets, chunk_index, elapsed_days):
        return chunk_index < 1, {"chunk": chunk_index, "elapsed_days": elapsed_days}

    result = run_chunked(
        coupler,
        total_time="6 days",
        chunk="2 days",
        output_dir=tmp_path,
        health_check=fails_on_the_second_chunk,
    )
    assert not result.completed
    # Two chunks ran: the good one and the one that failed. The third never did.
    assert result.steps_completed == 4
    assert len(result.reports) == 2
    assert sorted(path.name for path in result.paths) == [
        "ocn-00000000.nc", "ocn-00000002.nc",
        "seaice-00000000.nc", "seaice-00000002.nc",
    ]


def test_unhealthy_chunk_can_be_logged_and_ignored(coupler, tmp_path):
    """`bail_on_unhealthy=False` integrates an unhealthy state and says so."""
    result = run_chunked(
        coupler,
        total_time="4 days",
        chunk="2 days",
        output_dir=tmp_path,
        health_check=lambda datasets, index, days: (False, {"chunk": index}),
        bail_on_unhealthy=False,
    )
    assert result.completed
    assert result.steps_completed == 4
    assert len(result.reports) == 2


def test_no_health_check_collects_no_reports(coupler, tmp_path):
    """`health_check=None` is no gate at all, not a gate that always passes."""
    result = run_chunked(
        coupler,
        total_time="2 days",
        chunk="2 days",
        output_dir=tmp_path,
        health_check=None,
    )
    assert result.completed
    assert result.reports == []


def test_default_health_check_skips_without_an_atmosphere():
    """A coupled model with no `atm` dataset gets an abstention, not a pass."""
    ok, report = default_health_check({"ocn": xr.Dataset()}, 3, 12.0)
    assert ok
    assert report == {"chunk": 3, "elapsed_days": 12.0, "skipped": "no atmosphere"}


class BlowsUpInTheLastRecord(Coupler):
    """A two-slab coupler whose ocean's chunk ends with one NaN point.

    A slab ocean integrated for a few days does not blow up, and tuning one
    until it did would stop it being a slab ocean. What a health gate has to
    cope with is the *shape* of a blow-up rather than its cause -- a bad
    value in the chunk's final record -- so that is injected here, at the one
    place the driver takes a chunk's datasets from.
    """

    def to_xarray(self, diagnostics, time=None, *, first_step=0):
        datasets = super().to_xarray(diagnostics, time, first_step=first_step)
        ocean = datasets["ocn"]["sea_surface_temperature"]
        values = np.asarray(ocean.values).copy()
        values[-1, 0, 0] = np.nan
        datasets["ocn"]["sea_surface_temperature"] = ocean.copy(data=values)
        return datasets


def two_slabs_blowing_up() -> Coupler:
    """Return :func:`two_slabs`' model, with a NaN at the end of every chunk."""
    grid = make_grid()
    components = {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)}
    return BlowsUpInTheLastRecord(
        components,
        default_exchangers(components),
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )


def rejects_a_nan_in_the_last_record(datasets, chunk_index, elapsed_days):
    """Ask of the ocean what `jcm.diagnostics.check_health` asks of the atmosphere.

    The real gate reads `isel(time=-1)` and fails on a NaN there; this is the
    same question, put to the one component a two-slab coupler has.
    """
    last = datasets["ocn"]["sea_surface_temperature"].isel(time=-1)
    unhealthy = bool(np.isnan(np.asarray(last)).any())
    return not unhealthy, {"chunk": chunk_index, "nan_in_last_record": unhealthy}


@pytest.mark.parametrize(
    ("options", "records"),
    [({}, 4), ({"output_averages": True}, 1), ({"subsample": 2}, 2)],
)
def test_health_gate_sees_the_chunk_unreduced(tmp_path, options, records):
    """A state that goes bad at the end of a chunk is caught however output is reduced.

    Both reductions destroy the evidence a gate reads: `output_averages`
    averages the chunk, and xarray's mean skips NaNs and dilutes a finite
    extreme, while `subsample=2` drops the last of four records outright. So
    the gate is given the chunk as integrated, and only the copy written to
    disk is reduced -- which the file's own record count and the NaN's
    absence from it check, so the fix cannot be "stop reducing the output".
    """
    result = run_chunked(
        two_slabs_blowing_up(),
        total_time="4 days",
        chunk="4 days",
        output_dir=tmp_path,
        health_check=rejects_a_nan_in_the_last_record,
        **options,
    )

    assert not result.completed
    assert result.reports == [{"chunk": 0, "nan_in_last_record": True}]

    with xr.open_dataset(tmp_path / "ocn-00000000.nc") as written:
        assert written.sizes["time"] == records
        # The file is the reduced form, and in both reduced forms the bad
        # point is no longer in it -- which is exactly what the gate used to
        # be shown.
        nan_in_file = bool(np.isnan(written["sea_surface_temperature"].values).any())
        assert nan_in_file == (not options)


# ---------------------------------------------------------------------------
# Chunking and resuming give the same run
# ---------------------------------------------------------------------------


def test_continuous_chunked_resumed_agree(tmp_path):
    """Ten steps in one chunk, in two chunks, and across a restart, all agree.

    This is the property the whole loop exists to have: how a run is *divided*
    -- by chunk, by checkpoint, by process -- must not change the trajectory.
    The step counter lives in the carry, so the seasonal cycle and every
    component's clock continue across a boundary rather than restarting.
    """
    continuous = run_chunked(
        two_slabs(), total_time="10 days", chunk="10 days",
        output_dir=tmp_path / "continuous",
    )
    chunked = run_chunked(
        two_slabs(), total_time="10 days", chunk="5 days",
        output_dir=tmp_path / "chunked",
    )

    checkpoint = tmp_path / "checkpoint"
    first_half = run_chunked(
        two_slabs(), total_time="5 days", chunk="5 days",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
    )
    # A second, independently built coupler, resuming from the file alone:
    # what a new process does, without the carry the first run returned.
    resumed = run_chunked(
        two_slabs(), total_time="10 days", chunk="5 days",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
    )

    assert first_half.steps_completed == 5
    for result in (chunked, resumed):
        assert result.steps_completed == continuous.steps_completed == 10
        assert_carries_agree(result.final_carry, continuous.final_carry, atol=1e-12)

    # The resumed run wrote the second chunk's file, not the first one again.
    assert sorted(path.name for path in resumed.paths) == [
        "ocn-00000005.nc", "seaice-00000005.nc"
    ]
    assert (tmp_path / "restarted" / "ocn-00000000.nc").exists()


def test_checkpoint_is_one_directory_rewritten_each_chunk(coupler, tmp_path):
    """The checkpoint is a single directory holding the newest state only."""
    checkpoint = tmp_path / "checkpoint"
    result = run_chunked(
        coupler, total_time="4 days", chunk="2 days",
        output_dir=tmp_path, checkpoint_path=checkpoint,
    )
    assert (checkpoint / CARRY_FILENAME).exists()
    assert sorted(p.name for p in checkpoint.iterdir()) == [CARRY_FILENAME]

    restored = two_slabs().load_state(checkpoint)
    assert int(restored.step) == 4
    assert_carries_agree(restored, result.final_carry, atol=1e-12)


def test_resume_skips_incomplete_checkpoint(coupler, tmp_path, caplog):
    """A checkpoint directory with no carry file is stepped over, not loaded.

    `jem.checkpoint` publishes the carry file last, so a directory without one
    is what an interrupted save leaves behind: its component states belong to
    a step nothing records. The run starts from the initial carry instead, and
    says so.
    """
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )
    (checkpoint / CARRY_FILENAME).unlink()

    with caplog.at_level(logging.WARNING, logger="jem.driver"):
        result = run_chunked(
            coupler, total_time="2 days", chunk="2 days",
            output_dir=tmp_path / "second", checkpoint_path=checkpoint,
        )
    assert "not a complete checkpoint" in caplog.text
    # Started from step 0, so the run integrated its two days again.
    assert result.steps_completed == 2


def test_resume_with_a_different_chunk_length_still_stops_on_time(tmp_path):
    """A checkpoint part-way through a chunk is finished off in a short batch.

    The chunk length is a choice of the run, not a property of the checkpoint,
    so a run resumed with a longer chunk starts mid-chunk. What is left is
    computed from the restored step counter, so the run still stops exactly at
    `total_time` -- and, because `remaining_batches` puts the short batch
    LAST, the eight days are integrated as 3 + 4 + 1. The control run is what
    makes that a statement about the trajectory rather than about the counter:
    the separately-compiled short final batch has to produce the same numbers
    as one continuous eight-day integration, which is the property a bug in
    `remaining_batches` or in the driver's per-length trajectory cache would
    break while still stopping at step 8.
    """
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="3 days", chunk="3 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )
    resumed = run_chunked(
        two_slabs(), total_time="8 days", chunk="4 days",
        output_dir=tmp_path / "second", checkpoint_path=checkpoint,
    )
    assert resumed.completed
    assert resumed.steps_completed == 8

    continuous = run_chunked(
        two_slabs(), total_time="8 days", chunk="8 days",
        output_dir=tmp_path / "continuous",
    )
    assert_carries_agree(resumed.final_carry, continuous.final_carry, atol=1e-12)


def test_resume_with_a_different_chunk_length_keeps_the_earlier_files(tmp_path):
    """A different chunk length on resume must not write over earlier output.

    Three days in one chunk, then a resume with four-day chunks into the SAME
    output directory. Every file is named after the coupled step its chunk
    starts at -- 0 for the first run, then 3 and 7 (a full four-day chunk and
    then the short one that stops the run exactly at eight days) -- so all
    three survive. Under a chunk *index* the resumed run's first chunk would
    have been index `3 // 4 == 0` again, and the first run's three days of
    output would have been silently replaced.
    """
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    first = run_chunked(
        two_slabs(), total_time="3 days", chunk="3 days",
        output_dir=output, checkpoint_path=checkpoint,
    )
    resumed = run_chunked(
        two_slabs(), total_time="8 days", chunk="4 days",
        output_dir=output, checkpoint_path=checkpoint,
    )
    assert resumed.steps_completed == 8

    assert [path.name for path in first.paths] == [
        "ocn-00000000.nc", "seaice-00000000.nc"
    ]
    assert sorted(path.name for path in resumed.paths) == [
        "ocn-00000003.nc", "ocn-00000007.nc",
        "seaice-00000003.nc", "seaice-00000007.nc",
    ]
    # The first run's files are still there, and still hold its three days.
    assert all(path.exists() for path in first.paths)
    with xr.open_dataset(output / "ocn-00000000.nc") as written:
        assert written.sizes["time"] == 3

    # The whole run reads back as one continuous eight-day series.
    with xr.open_mfdataset(
        sorted(output.glob("ocn-*.nc")), combine="by_coords"
    ) as combined:
        assert combined.sizes["time"] == 8


# ---------------------------------------------------------------------------
# Output options reach the files
# ---------------------------------------------------------------------------


def test_output_options_reach_the_files(coupler, tmp_path):
    """`output_averages` and `subsample` are passed through to the postprocessing."""
    averaged = run_chunked(
        coupler, total_time="4 days", chunk="4 days",
        output_dir=tmp_path / "averaged", output_averages=True,
    )
    dataset = xr.open_dataset(averaged.paths[0])
    assert dataset.sizes["time"] == 1
    assert "time: mean" in dataset["sea_surface_temperature"].attrs["cell_methods"]

    thinned = run_chunked(
        two_slabs(), total_time="4 days", chunk="4 days",
        output_dir=tmp_path / "thinned", subsample=2,
    )
    assert xr.open_dataset(thinned.paths[0]).sizes["time"] == 2


# ---------------------------------------------------------------------------
# The same properties with the real atmosphere
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_continuous_chunked_resumed_agree_with_jcm(tmp_path):
    """The chunk/restart invariance holds for a real coupled model too.

    The toy coupler above pins the arithmetic exactly; this pins the thing the
    arithmetic is for -- an atmosphere whose cross-step physics carry, dycore
    state and forcing all have to survive a chunk boundary and a round trip
    through a checkpoint file. The tolerance is 1e-6 because the atmosphere
    integrates in float32.
    """
    import jcm
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.terrain import TerrainData

    from jem.components import JCMComponent
    from jem.components.slab import SlabGrid

    def build() -> Coupler:
        coords = get_speedy_coords(layers=5, spectral_truncation=21)
        model = jcm.model.Model(
            coords=coords,
            terrain=TerrainData.aquaplanet(coords),
            start_date=START_DATE,
            log_level=50,
        )
        atm = JCMComponent(model)
        components = {
            "atm": atm,
            "ocn": SlabOceanModel(SlabGrid.from_coords(coords.horizontal)),
        }
        return Coupler(
            components,
            default_exchangers(components),
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=START_DATE,
        )

    continuous = run_chunked(
        build(), total_time="4 days", chunk="4 days",
        output_dir=tmp_path / "continuous",
    )
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        build(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
    )
    resumed = run_chunked(
        build(), total_time="4 days", chunk="2 days",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
    )

    assert resumed.steps_completed == continuous.steps_completed == 4
    assert_carries_agree(resumed.final_carry, continuous.final_carry, atol=1e-6)
    # The atmosphere is there, so the default gate actually looked at it.
    assert all("skipped" not in report for report in continuous.reports)


@pytest.mark.slow
def test_run_smoke_cli(tmp_path):
    """`python -m jem.main` runs the shipped smoke configuration end to end.

    A subprocess, in a scratch working directory, because that is what a user
    types: it exercises Hydra's composition from the installed package, the
    resolvers the shipped configurations use, the run directory Hydra makes
    and the driver's own logging -- none of which an in-process call of
    `runners.run` would touch.
    """
    import os
    import subprocess
    import sys

    repository = pathlib.Path(__file__).resolve().parents[2]
    environment = dict(os.environ, JAX_PLATFORMS="cpu")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    finished = subprocess.run(
        [sys.executable, "-m", "jem.main",
         "+configuration=aquaplanet-slab", "coupled_run=smoke"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=1800,
    )
    assert finished.returncode == 0, finished.stderr[-4000:]

    run_directories = sorted((tmp_path / "outputs").glob("*/*"))
    assert len(run_directories) == 1, run_directories
    written = sorted(path.name for path in run_directories[0].glob("*.nc"))
    assert written == [
        "atm-00000000.nc", "ocn-00000000.nc", "seaice-00000000.nc",
    ]

    # The run said what it did, at INFO, through the logger rather than print.
    log = (run_directories[0] / "main.log").read_text()
    assert "Chunk 0:" in log
    assert "Finished: 2 coupled steps, completed=True, 3 file(s) written." in log
