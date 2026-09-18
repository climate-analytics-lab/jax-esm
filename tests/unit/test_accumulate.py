"""In-scan diagnostic reduction: the ``accumulate`` hook and the binned means.

The model is a two-slab coupler -- an idealized atmosphere over a relaxing
slab ocean on a 4x3 grid -- run for a whole 365-day year, which is the
shortest run in which every month exists and the boundary cases (a step whose
interval ends exactly on the first of a month, and the last step of the year,
which is labelled 1 January of the next one) actually occur. A year is also
exactly 73 pentads, so the same run fills a ``windowed_mean`` accumulator once
with nothing wrapping.

The last two sections run the same pair of slabs *weaved*: the atmosphere
stepped hourly within the daily coupling, once as a repeated workflow and once
as a nested hourly coupler. Those runs are 40 days from 1 January, so they
cross a month boundary -- the case in which a coupled step's sub-steps do not
all belong to the same month, and the only case that can tell the two binning
conventions apart.

Every test here compares the reduction computed *inside* the ``lax.scan``
with the same reduction computed on the host from the stacked diagnostics,
because the point of the hook is to be indistinguishable from the obvious
thing while never materialising the trajectory.
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest

from jem.accumulate import (
    MONTHS_PER_YEAR,
    month_lengths,
    monthly_mean,
    windowed_mean,
)
from jem.base.coupler import Coupler
from jem.components.slab import (
    SlabAtmosphereModel,
    SlabOceanModel,
    SlabOceanParameters,
)
from tests.unit.slab_test_utils import (
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    make_grid,
    write_climatology,
)

START_DATE = jdt.to_datetime("2001-01-01")
CALENDAR = "365_day"
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")
STEPS_PER_YEAR = 365

#: The sub-seasonal window the `windowed_mean` tests use, and how many of them
#: a 365-day year holds exactly -- so a whole year's records fill the
#: accumulator once, with nothing wrapping, which is the case a forecast
#: scored on pentads is run in.
PENTAD_DAYS = 5
PENTADS_PER_YEAR = STEPS_PER_YEAR // PENTAD_DAYS

#: Short enough that the ocean tracks its seasonal climatology within the
#: year, so the twelve monthly means differ from each other by far more than
#: float32 noise -- and short enough that the gradient with respect to it is
#: well clear of that noise too.
RELAXATION_TIME = 20.0 * 86400.0


def slab_exchange(components, time):
    """Wire the slab atmosphere to the slab ocean, and back.

    The atmosphere computes the surface heat flux from the sea surface
    temperature the ocean handed it; that same flux is what the ocean is
    cooled or warmed by, and what the atmosphere reports as its own forcing.
    """
    del time
    atmosphere = components["atm"]
    ocean = components["ocn"]
    heat_flux = atmosphere["derived"].internal_total_heat_flux
    return dict(
        components,
        atm=dict(
            atmosphere,
            forcing=atmosphere["forcing"].replace(
                sea_surface_temperature=ocean["state"].sea_surface_temperature,
                total_heat_flux=heat_flux,
            ),
        ),
        ocn=dict(ocean, forcing=ocean["forcing"].replace(total_heat_flux=heat_flux)),
    )


def seasonal_sst_climatology(path) -> str:
    """Write a 12-month SST climatology with a real annual cycle on the test grid."""
    monthly = 288.0 + 10.0 * np.cos(2 * np.pi * (np.arange(12) - 6) / 12.0)
    latitude_shape = (12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES))
    values = np.broadcast_to(monthly[:, None, None], latitude_shape)
    return write_climatology(
        path / "sst.nc", "sst", np.array(values, dtype=np.float32)
    )


def build_coupler(climatology_file, relaxation_time=RELAXATION_TIME) -> Coupler:
    """Return the two-slab coupler the tests run."""
    grid = make_grid()
    ocean = SlabOceanModel(
        grid,
        SlabOceanParameters(
            forcing_method="relaxation", relaxation_time=relaxation_time
        ),
        sst_clim_file=climatology_file,
    )
    return Coupler(
        {"atm": SlabAtmosphereModel(grid), "ocn": ocean},
        {"exchange": slab_exchange},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )


@pytest.fixture(scope="module")
def climatology_file(tmp_path_factory):
    return seasonal_sst_climatology(tmp_path_factory.mktemp("climatology"))


@pytest.fixture(scope="module")
def coupler(climatology_file):
    return build_coupler(climatology_file)


@pytest.fixture(scope="module")
def record_months(coupler):
    """Return the 0-based calendar month of each of the year's output records.

    Taken from the ``datetime64`` labels ``Coupler.to_xarray`` puts on the
    records -- the end of each coupling interval -- so this is the binning a
    user reading the written output would do, and what the accumulator has to
    agree with.
    """
    labels = coupler.time_axis(0, STEPS_PER_YEAR).datetimes()
    return labels.astype("datetime64[M]").astype(int) % MONTHS_PER_YEAR


@pytest.fixture(scope="module")
def stacked_year(coupler):
    """Run the year the ordinary way, keeping every step's diagnostics."""
    trajectory = coupler.generate_trajectory_function(STEPS_PER_YEAR)
    return trajectory(coupler.initialize())


@pytest.fixture(scope="module")
def accumulated_year(coupler):
    """Run the same year reducing to monthly means inside the scan."""
    monthly = monthly_mean(coupler)
    trajectory = coupler.generate_trajectory_function(
        STEPS_PER_YEAR, accumulate=monthly
    )
    carry, accumulator = trajectory(coupler.initialize())
    return monthly, carry, accumulator


def host_monthly_means(diagnostics, record_months):
    """Return the monthly means of stacked diagnostics, binned on the host."""

    def mean_by_month(leaf):
        leaf = np.asarray(leaf)
        return np.stack(
            [leaf[record_months == month].mean(axis=0) for month in range(MONTHS_PER_YEAR)]
        )

    return jax.tree_util.tree_map(mean_by_month, diagnostics)


# ---------------------------------------------------------------------------
# The hook itself
# ---------------------------------------------------------------------------


def test_without_accumulate_the_trajectory_is_unchanged(coupler, stacked_year):
    """``accumulate=None`` is the function it was before the hook existed."""
    carry, diagnostics = stacked_year

    assert int(carry.step) == STEPS_PER_YEAR
    assert set(diagnostics) == {"atm", "ocn"}
    for component_diagnostics in diagnostics.values():
        for leaf in jax.tree_util.tree_leaves(component_diagnostics):
            assert jnp.shape(leaf)[0] == STEPS_PER_YEAR


def test_accumulating_does_not_stack_the_diagnostics(accumulated_year):
    """The second return value is the accumulator, not a trajectory."""
    _, carry, (sums, counts) = accumulated_year

    assert int(carry.step) == STEPS_PER_YEAR
    assert counts.shape == (MONTHS_PER_YEAR,)
    # Every step of the year is counted exactly once, and the month lengths
    # are the calendar's, not the run's chunking.
    np.testing.assert_array_equal(
        np.asarray(counts), [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    )
    for leaf in jax.tree_util.tree_leaves(sums):
        assert jnp.shape(leaf)[0] == MONTHS_PER_YEAR


def test_the_accumulated_run_takes_the_same_trajectory(stacked_year, accumulated_year):
    """Reducing in the scan changes what is returned, not what is integrated."""
    stacked_carry, _ = stacked_year
    _, accumulated_carry, _ = accumulated_year

    stacked_leaves = jax.tree_util.tree_leaves(stacked_carry)
    accumulated_leaves = jax.tree_util.tree_leaves(accumulated_carry)
    assert jax.tree_util.tree_structure(stacked_carry) == jax.tree_util.tree_structure(
        accumulated_carry
    )
    for stacked, accumulated in zip(stacked_leaves, accumulated_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(stacked), np.asarray(accumulated))


# ---------------------------------------------------------------------------
# The monthly mean itself
# ---------------------------------------------------------------------------


def test_monthly_means_match_the_host_side_binning(
    stacked_year, accumulated_year, record_months
):
    """Every leaf equals the mean of the records the datetime labels bin there.

    This is the whole contract of ``monthly_mean``: the convention it bins by
    is the one the output labels imply, so the accumulated mean and a mean
    taken from the written output are the same number.
    """
    _, diagnostics = stacked_year
    monthly, _, accumulator = accumulated_year

    expected = host_monthly_means(diagnostics, record_months)
    actual = monthly.finalize(accumulator)

    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert actual_structure == expected_structure
    for index, (got, want) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-4,
            err_msg=f"leaf {index}",
        )


def test_monthly_means_match_an_xarray_groupby(coupler, stacked_year, accumulated_year):
    """And they match what a user would compute from the netCDF output.

    The tie to ``xarray`` matters because the labelling convention lives in
    ``TimeAxis.datetimes``, not in this module: if the two ever drifted apart,
    an accumulated monthly mean and a ``groupby("time.month")`` of the same
    run would quietly disagree.
    """
    _, diagnostics = stacked_year
    monthly, _, accumulator = accumulated_year

    ocean = coupler.to_xarray(diagnostics)["ocn"]
    from_output = ocean.sea_surface_temperature.groupby("time.month").mean("time")
    np.testing.assert_array_equal(from_output.month.values, np.arange(1, 13))

    accumulated = monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature
    np.testing.assert_allclose(
        np.asarray(accumulated), from_output.values, rtol=1e-5, atol=1e-4
    )


def test_a_month_with_no_steps_is_nan(coupler):
    """Ten empty bins in a two-month run are NaN, not a zero that plots as data."""
    monthly = monthly_mean(coupler)
    trajectory = coupler.generate_trajectory_function(58, accumulate=monthly)
    _, accumulator = trajectory(coupler.initialize())
    _, counts = accumulator

    means = monthly.finalize(accumulator)
    sea_surface_temperature = np.asarray(
        means["ocn"]["state"].sea_surface_temperature
    )
    # 58 daily steps from 1 January are labelled 2 January to 28 February --
    # 30 records in January and 28 in February -- so March onwards is empty.
    np.testing.assert_array_equal(np.asarray(counts)[:2], [30, 28])
    assert np.all(np.isfinite(sea_surface_temperature[:2]))
    assert np.all(np.isnan(sea_surface_temperature[2:]))


def test_a_chunked_run_accumulates_the_same_means(
    coupler, accumulated_year, record_months
):
    """The accumulator crosses a chunk boundary untouched.

    The two chunks are different lengths, which is the case that matters: a
    driver chunks by whatever suits the machine, not by month, and the
    accumulator must not care where the boundary fell.
    """
    del record_months
    monthly, _, one_call = accumulated_year

    first = coupler.generate_trajectory_function(200, accumulate=monthly)
    second = coupler.generate_trajectory_function(
        STEPS_PER_YEAR - 200, accumulate=monthly
    )
    carry, accumulator = first(coupler.initialize())
    carry, accumulator = second(carry, accumulator)

    assert int(carry.step) == STEPS_PER_YEAR
    chunked_leaves = jax.tree_util.tree_leaves(monthly.finalize(accumulator))
    single_leaves = jax.tree_util.tree_leaves(monthly.finalize(one_call))
    for index, (chunked, single) in enumerate(
        zip(chunked_leaves, single_leaves, strict=True)
    ):
        np.testing.assert_allclose(
            np.asarray(chunked), np.asarray(single), rtol=1e-6, atol=1e-5,
            err_msg=f"leaf {index}",
        )


def test_remat_does_not_change_the_accumulated_means(coupler, record_months):
    """``remat=True`` recomputes the step; it must not recompute a different one."""
    monthly = monthly_mean(coupler)
    steps = int(np.sum(record_months < 2))
    plain = coupler.generate_trajectory_function(steps, accumulate=monthly)
    remat = coupler.generate_trajectory_function(steps, accumulate=monthly, remat=True)

    _, plain_accumulator = plain(coupler.initialize())
    _, remat_accumulator = remat(coupler.initialize())

    for got, want in zip(
        jax.tree_util.tree_leaves(monthly.finalize(remat_accumulator)),
        jax.tree_util.tree_leaves(monthly.finalize(plain_accumulator)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


# ---------------------------------------------------------------------------
# Means over fixed-length windows
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def accumulated_pentads(coupler):
    """Run the same year reducing to 5-day means inside the scan."""
    pentads = windowed_mean(coupler, "5 days", n_windows=PENTADS_PER_YEAR)
    trajectory = coupler.generate_trajectory_function(
        STEPS_PER_YEAR, accumulate=pentads
    )
    carry, accumulator = trajectory(coupler.initialize())
    return pentads, carry, accumulator


def test_a_window_is_the_records_its_interval_ends_with(accumulated_pentads):
    """The first pentad is days 1 to 5 of the run, and every pentad holds five.

    This is the convention the docstring commits to -- a step is binned by the
    label of the record it produces, and a window is closed at its end, so the
    record labelled exactly day 5 finishes the first pentad rather than
    starting the second. A forecast's "first pentad" is days 1-5, and an
    off-by-one here would make it days 1-4.
    """
    _, carry, (_, counts) = accumulated_pentads

    assert int(carry.step) == STEPS_PER_YEAR
    assert counts.shape == (PENTADS_PER_YEAR,)
    np.testing.assert_array_equal(
        np.asarray(counts), np.full(PENTADS_PER_YEAR, PENTAD_DAYS)
    )


def test_windowed_means_match_an_xarray_groupby(coupler, stacked_year, accumulated_pentads):
    """Every window equals the mean of the records whose labels fall in it.

    The same tie to the written output the monthly test makes, for the bins
    the *run* defines rather than the ones the calendar does: the window index
    is computed here from the `datetime64` labels alone, so it fails if the
    accumulator's step-to-window arithmetic and `TimeAxis.datetimes` ever stop
    agreeing.
    """
    _, diagnostics = stacked_year
    pentads, _, accumulator = accumulated_pentads

    ocean = coupler.to_xarray(diagnostics)["ocn"]
    elapsed_days = (
        ocean["time"].values - np.datetime64("2001-01-01")
    ) / np.timedelta64(1, "D")
    # `ceil(elapsed / window) - 1`: the window a label belongs to when a window
    # is the half-open interval (w*window, (w+1)*window] -- closed at the end,
    # which is where JEM labels a record covering an interval.
    window = np.ceil(elapsed_days / PENTAD_DAYS).astype(int) - 1
    from_output = (
        ocean.sea_surface_temperature.assign_coords(window=("time", window))
        .groupby("window")
        .mean("time")
    )
    np.testing.assert_array_equal(
        from_output.window.values, np.arange(PENTADS_PER_YEAR)
    )

    accumulated = pentads.finalize(accumulator)["ocn"]["state"].sea_surface_temperature
    np.testing.assert_allclose(
        np.asarray(accumulated), from_output.values, rtol=1e-5, atol=1e-4
    )


def test_a_window_with_no_steps_is_nan(coupler):
    """An accumulator sized for a year, run for a fortnight, is mostly NaN."""
    pentads = windowed_mean(coupler, "5 days", n_windows=PENTADS_PER_YEAR)
    trajectory = coupler.generate_trajectory_function(14, accumulate=pentads)
    _, accumulator = trajectory(coupler.initialize())
    _, counts = accumulator

    sea_surface_temperature = np.asarray(
        pentads.finalize(accumulator)["ocn"]["state"].sea_surface_temperature
    )
    # Days 1-5, 6-10 and then 11-14: the third pentad is short, and it is a
    # mean of the four records that fell in it rather than of five.
    np.testing.assert_array_equal(np.asarray(counts)[:4], [5, 5, 4, 0])
    assert np.all(np.isfinite(sea_surface_temperature[:3]))
    assert np.all(np.isnan(sea_surface_temperature[3:]))


def test_a_run_longer_than_the_accumulator_wraps(coupler):
    """Window `w` composites windows `w`, `w + n_windows`, ... of a long run.

    The accumulator's size is fixed at trace time -- that is what makes the
    reduction cost nothing per step -- so a run that outlasts it wraps, the
    way the monthly table wraps years. The counts are what show it: with ten
    pentads and 73 pentads of run, the first three bins collect eight windows
    each and the rest seven.
    """
    pentads = windowed_mean(coupler, "5 days", n_windows=10)
    trajectory = coupler.generate_trajectory_function(
        STEPS_PER_YEAR, accumulate=pentads
    )
    _, (_, counts) = trajectory(coupler.initialize())

    np.testing.assert_array_equal(
        np.asarray(counts), [40, 40, 40, 35, 35, 35, 35, 35, 35, 35]
    )
    assert int(np.sum(np.asarray(counts))) == STEPS_PER_YEAR


@pytest.mark.parametrize(
    ("window", "expected"),
    [("5 days", PENTADS_PER_YEAR), ("7 days", 53), (365, 1)],
)
def test_total_time_counts_the_windows_of_the_run(coupler, window, expected):
    """`total_time` sizes the accumulator: enough windows to cover the run.

    Seven days do not divide a 365-day year, so the year needs 53 weekly
    windows and the last one holds a single day -- rounding down would drop it
    into the first window instead, which is the silent corruption the ceiling
    avoids.
    """
    accumulator = windowed_mean(coupler, window, total_time="1 year")
    _, counts = accumulator.init()
    assert counts.shape == (expected,)


def test_total_time_and_n_windows_agree(coupler):
    """The two ways of sizing the accumulator build the same thing."""
    from_total = windowed_mean(coupler, "5 days", total_time="1 year")
    from_count = windowed_mean(coupler, "5 days", n_windows=PENTADS_PER_YEAR)

    trajectory = coupler.generate_trajectory_function(50, accumulate=from_total)
    _, first = trajectory(coupler.initialize())
    _, second = coupler.generate_trajectory_function(50, accumulate=from_count)(
        coupler.initialize()
    )
    for got, want in zip(
        jax.tree_util.tree_leaves(from_total.finalize(first)),
        jax.tree_util.tree_leaves(from_count.finalize(second)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_a_window_that_is_not_whole_coupling_steps_is_refused(coupler):
    """Half a step cannot be attributed to either side of the boundary."""
    with pytest.raises(ValueError, match="whole number of coupling steps"):
        windowed_mean(coupler, "36 hours", n_windows=4)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"n_windows": 4, "total_time": "20 days"}],
)
def test_exactly_one_of_n_windows_and_total_time_is_required(coupler, kwargs):
    """Neither is unanswerable and both is two answers to one question."""
    with pytest.raises(ValueError, match="exactly one of n_windows and total_time"):
        windowed_mean(coupler, "5 days", **kwargs)


@pytest.mark.parametrize("n_windows", [0, -3, 1.0, True])
def test_a_bad_n_windows_is_refused(coupler, n_windows):
    """`True` is an `int` that would silently mean "one window" -- the whole run."""
    with pytest.raises(ValueError, match="n_windows must be a positive integer"):
        windowed_mean(coupler, "5 days", n_windows=n_windows)


# ---------------------------------------------------------------------------
# Windows of more than one length
# ---------------------------------------------------------------------------

#: The variable-window pattern the tests cycle through: a 10-day window and a
#: 20-day one, so the bins are neither all the same length nor aligned with
#: anything the calendar defines.
PATTERN_DAYS = (10, 20)

#: Coupled steps of the calendar-month-window run: 1 January 2001 to 5
#: February 2002, so it passes a whole year. January 2002 is then a bin of its
#: own rather than a second helping of the first -- which is the point of
#: sizing the accumulator to the run -- and the last bin is a short one.
MONTH_WINDOW_STEPS = 400


def cycled_boundaries(lengths, bins):
    """Return the ends, in days, of `bins` windows cycling through `lengths`."""
    return np.cumsum([lengths[index % len(lengths)] for index in range(bins)])


def window_of_label(boundaries_days, times):
    """Return the window each `datetime64` label falls in, binned on the host.

    A window is closed at its end -- the record labelled exactly on a boundary
    is the last of the window it closes, not the first of the next one -- so
    the window of a label is the first boundary at or after its elapsed time,
    which is what `searchsorted(..., side="left")` returns.
    """
    elapsed_days = (times - np.datetime64("2001-01-01")) / np.timedelta64(1, "D")
    return np.searchsorted(boundaries_days, elapsed_days, side="left")


def test_a_pattern_of_windows_cycles_over_the_run(coupler, stacked_year):
    """Alternating 10- and 20-day windows bin the year as the labels say.

    The whole of the variable-window contract in one run: the lengths cycle,
    `total_time` counts enough of them to cover the run (rounding up into a
    short last one), and every record lands in the window its own label falls
    in -- checked against a plain numpy binning of the stacked diagnostics and
    against an xarray reduction of the same run's output.
    """
    _, diagnostics = stacked_year
    pattern = [f"{days} days" for days in PATTERN_DAYS]
    means = windowed_mean(coupler, pattern, total_time="1 year")
    trajectory = coupler.generate_trajectory_function(
        STEPS_PER_YEAR, accumulate=means
    )
    _, accumulator = trajectory(coupler.initialize())
    _, counts = accumulator

    # Twelve whole 30-day cycles cover 360 days, and one more 10-day window
    # has to exist to hold the last five days of the year.
    bins = 25
    assert counts.shape == (bins,)
    boundaries = cycled_boundaries(PATTERN_DAYS, bins)
    expected = np.diff(np.concatenate([[0], boundaries]))
    expected[-1] = STEPS_PER_YEAR - boundaries[-2]
    np.testing.assert_array_equal(np.asarray(counts), expected)

    means_by_window = means.finalize(accumulator)
    ocean = coupler.to_xarray(diagnostics)["ocn"]
    window = window_of_label(boundaries, ocean["time"].values)

    # Against a numpy binning of the stacked diagnostics...
    stacked = np.asarray(diagnostics["atm"]["state"].mean_air_temperature)
    host = np.stack([stacked[window == index].mean(axis=0) for index in range(bins)])
    np.testing.assert_allclose(
        np.asarray(means_by_window["atm"]["state"].mean_air_temperature),
        host,
        rtol=1e-5,
        atol=1e-4,
    )

    # ... and against the same binning expressed as an xarray reduction of the
    # written output, which is how a user would check it.
    from_output = (
        ocean.sea_surface_temperature.assign_coords(window=("time", window))
        .groupby("window")
        .mean("time")
    )
    np.testing.assert_array_equal(from_output.window.values, np.arange(bins))
    np.testing.assert_allclose(
        np.asarray(means_by_window["ocn"]["state"].sea_surface_temperature),
        from_output.values,
        rtol=1e-5,
        atol=1e-4,
    )


def test_a_pattern_shorter_than_the_accumulator_repeats_and_wraps(coupler):
    """`n_windows` beyond the pattern repeats it; the run still wraps at the end.

    Three windows of 10, 20 and 10 days are 40 days of accumulator, so a
    60-day run wraps its last 20 days back into the first two bins -- the
    fixed-size accumulator behaving exactly as it does for equal windows.
    """
    means = windowed_mean(coupler, ["10 days", "20 days"], n_windows=3)
    trajectory = coupler.generate_trajectory_function(60, accumulate=means)
    _, (_, counts) = trajectory(coupler.initialize())

    np.testing.assert_array_equal(np.asarray(counts), [20, 30, 10])
    assert int(np.sum(np.asarray(counts))) == 60


def test_a_pattern_given_no_size_is_used_once(coupler):
    """A sequence and neither `n_windows` nor `total_time` is one cycle of it.

    The one case in which giving neither is answerable: the pattern itself
    says how many windows there are. A single window length still has to be
    told, since one window is never what was meant.
    """
    means = windowed_mean(coupler, ["10 days", "20 days"])
    _, counts = means.init()
    assert counts.shape == (2,)

    trajectory = coupler.generate_trajectory_function(30, accumulate=means)
    _, (_, counts) = trajectory(coupler.initialize())
    np.testing.assert_array_equal(np.asarray(counts), [10, 20])


@pytest.mark.parametrize("pattern", [["5 days", "36 hours"], [5, 1.5]])
def test_a_pattern_element_that_is_not_whole_steps_is_refused(coupler, pattern):
    """Every length is held to the rule a single window is held to."""
    with pytest.raises(ValueError, match=r"window\[1\].*whole number of coupling"):
        windowed_mean(coupler, pattern, n_windows=4)


def test_an_empty_pattern_is_refused(coupler):
    """No windows to cycle through is no accumulator to build."""
    with pytest.raises(ValueError, match="no windows"):
        windowed_mean(coupler, [], n_windows=2)


def test_a_pattern_cannot_be_given_both_sizes(coupler):
    """Two answers to one question, sequence or not."""
    with pytest.raises(ValueError, match="exactly one of n_windows and total_time"):
        windowed_mean(coupler, ["10 days"], n_windows=4, total_time="20 days")


@pytest.fixture(scope="module")
def calendar_month_windows(coupler):
    """Run 400 days binned into calendar-month *windows*, stacked and reduced."""
    months = windowed_mean(
        coupler, month_lengths(coupler), total_time=f"{MONTH_WINDOW_STEPS} days"
    )
    _, accumulator = coupler.generate_trajectory_function(
        MONTH_WINDOW_STEPS, accumulate=months
    )(coupler.initialize())
    _, diagnostics = coupler.generate_trajectory_function(MONTH_WINDOW_STEPS)(
        coupler.initialize()
    )
    return months, accumulator, diagnostics


def test_calendar_month_windows_do_not_wrap_into_a_climatology(
    coupler, calendar_month_windows
):
    """Fourteen bins over 400 days: every month of the run, not twelve of them.

    This is "monthly averages without drifting": the windows are the calendar's
    own month lengths, so no bin spans parts of two months the way a fixed
    30-day window does, and the accumulator is sized to the run, so January
    2002 is bin 12 rather than more records in bin 0 -- which is what
    `monthly_mean`'s twelve-bin climatology would make it.
    """
    _, (_, counts), _ = calendar_month_windows
    lengths = month_lengths(coupler)

    # The example the docstrings and the README give, counted: ten years of
    # calendar months is 120 bins, not twelve.
    _, decade_counts = windowed_mean(
        coupler, lengths, total_time="10 years"
    ).init()
    assert decade_counts.shape == (10 * MONTHS_PER_YEAR,)

    assert counts.shape == (MONTHS_PER_YEAR + 2,)
    np.testing.assert_array_equal(
        np.asarray(counts),
        [*lengths, lengths[0], MONTH_WINDOW_STEPS - sum(lengths) - lengths[0]],
    )


def test_calendar_month_windows_match_a_year_month_grouping(
    coupler, calendar_month_windows
):
    """Each bin is the mean of one month of the output, year by year.

    The comparison is the reduction a user would write on the written output,
    and it also pins down what a calendar-month *window* is in the output's own
    terms: the records whose **interval** lies in that month, which is the
    label grouped by month after stepping back one coupling step.
    """
    months, accumulator, diagnostics = calendar_month_windows
    boundaries = cycled_boundaries(month_lengths(coupler), MONTHS_PER_YEAR + 2)
    ocean = coupler.to_xarray(diagnostics)["ocn"]
    window = window_of_label(boundaries, ocean["time"].values)

    interval_months = (
        ocean["time"].values - np.timedelta64(1, "D")
    ).astype("datetime64[M]")
    np.testing.assert_array_equal(
        window, np.unique(interval_months, return_inverse=True)[1]
    )

    from_output = (
        ocean.sea_surface_temperature.assign_coords(window=("time", window))
        .groupby("window")
        .mean("time")
    )
    np.testing.assert_array_equal(
        from_output.window.values, np.arange(MONTHS_PER_YEAR + 2)
    )
    np.testing.assert_allclose(
        np.asarray(
            months.finalize(accumulator)["ocn"]["state"].sea_surface_temperature
        ),
        from_output.values,
        rtol=1e-5,
        atol=1e-4,
    )


def test_a_month_window_and_a_month_differ_by_the_boundary_record(coupler):
    """The documented one-record difference between the two reductions.

    A window closes at its end and a calendar month closes at its start, so
    the record labelled 00:00 on 1 February is the **last** of January's
    window and the **first** of February's month. Nothing else separates
    `windowed_mean(coupler, month_lengths(coupler), ...)` from
    `monthly_mean(coupler)` bin for bin, so a user choosing between them is
    choosing between per-month bins and agreement with
    `groupby("time.month")` record for record.
    """
    labels = coupler.time_axis(0, MONTH_WINDOW_STEPS).datetimes()
    boundary = np.flatnonzero(labels == np.datetime64("2001-02-01"))
    window = window_of_label(
        cycled_boundaries(month_lengths(coupler), MONTHS_PER_YEAR + 2), labels
    )
    month = labels.astype("datetime64[M]").astype(int) % MONTHS_PER_YEAR

    assert boundary.size == 1
    assert int(window[boundary[0]]) == 0
    assert int(month[boundary[0]]) == 1


# ---------------------------------------------------------------------------
# Components that record more than once per coupled step
# ---------------------------------------------------------------------------

#: Sub-steps per coupled step in the weaved runs: an hourly atmosphere inside
#: a daily coupling, which is the classic surface/ocean weaving.
HOURS_PER_DAY = 24

#: Coupled steps of the weaved runs: 1 January to 9 February, so the run holds
#: a whole month boundary and then some. The daily step that covers 31 January
#: produces hourly records labelled 01:00 on the 31st through 00:00 on 1
#: February -- 23 of them in January and one in February -- which is the
#: discrepancy the per-record binning exists to get right.
BOUNDARY_STEPS = 40
JANUARY_DAYS = 31


def build_weaved_coupler(climatology_file) -> Coupler:
    """Return the two slabs with the atmosphere weaved hourly into a daily step."""
    grid = make_grid()
    ocean = SlabOceanModel(
        grid,
        SlabOceanParameters(
            forcing_method="relaxation", relaxation_time=RELAXATION_TIME
        ),
        sst_clim_file=climatology_file,
    )
    return Coupler(
        {"atm": SlabAtmosphereModel(grid), "ocn": ocean},
        {"exchange": slab_exchange},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
        workflow=[["exchange", "atm"] * HOURS_PER_DAY, "ocn"],
    )


def build_nested_coupler(climatology_file) -> Coupler:
    """Return the same weaving expressed as an hourly coupler inside a daily one."""
    grid = make_grid()
    ocean = SlabOceanModel(
        grid,
        SlabOceanParameters(
            forcing_method="relaxation", relaxation_time=RELAXATION_TIME
        ),
        sst_clim_file=climatology_file,
    )
    hourly = Coupler(
        {"atm": SlabAtmosphereModel(grid), "ocn": ocean},
        {"exchange": slab_exchange},
        coupling_timestep=jdt.to_timedelta(1, "hour"),
        start_date=START_DATE,
        calendar=CALENDAR,
        name="fast",
    )
    return Coupler(
        {"fast": hourly},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )


def run_both_ways(coupler, reduction, steps=BOUNDARY_STEPS):
    """Return the stacked diagnostics and the accumulator of the same run."""
    _, diagnostics = coupler.generate_trajectory_function(steps)(coupler.initialize())
    _, accumulator = coupler.generate_trajectory_function(
        steps, accumulate=reduction
    )(coupler.initialize())
    return diagnostics, accumulator


def fold_sub_steps(means, counts):
    """Return the mean over every record of a bin, from the per-slot means.

    ``finalize`` keeps the sub-step axis: bin `b`, slot `j` is the mean of the
    records of call `j` that fell in `b`. Weighting each slot by its own count
    is what recovers the mean over all of the bin's records -- which is what a
    ``groupby`` of the written output computes. A straight mean over the slots
    would equal it only where every slot holds the same number of records,
    which is exactly what a month boundary breaks.
    """
    means = np.asarray(means)
    counts = np.asarray(counts)
    weights = counts.reshape(counts.shape + (1,) * (means.ndim - counts.ndim))
    total = np.sum(np.where(weights == 0, 0.0, means) * weights, axis=1)
    per_bin = counts.sum(axis=1)
    per_bin = per_bin.reshape(per_bin.shape + (1,) * (total.ndim - per_bin.ndim))
    return np.where(per_bin == 0, np.nan, total / np.where(per_bin == 0, 1, per_bin))


@pytest.fixture(scope="module")
def weaved(climatology_file):
    """Return the weaved coupler, its stacked run and its monthly accumulator."""
    coupler = build_weaved_coupler(climatology_file)
    monthly = monthly_mean(coupler)
    diagnostics, accumulator = run_both_ways(coupler, monthly)
    return coupler, diagnostics, monthly, accumulator


def test_a_sub_stepped_component_is_counted_record_by_record(weaved):
    """The 24 hourly records of a coupled step land in the months they label.

    The counts are what show the convention directly: 23 of the records of the
    day covering 31 January are January's and the 24th, labelled 1 February
    00:00, is February's. Binning the whole coupled step by its own label --
    which is what the accumulator did before -- would have put all 24 in
    February and disagreed with the written output by a day of records.
    """
    _, _, _, (_, counts) = weaved

    # One count array per component, because the two record at different rates
    # and so fill different bins as one coupled step is folded in.
    assert set(counts) == {"atm", "ocn"}
    assert counts["atm"].shape == (MONTHS_PER_YEAR, HOURS_PER_DAY)
    assert counts["ocn"].shape == (MONTHS_PER_YEAR,)

    atmosphere = np.asarray(counts["atm"])
    np.testing.assert_array_equal(atmosphere[0], [JANUARY_DAYS] * 23 + [30])
    february_days = BOUNDARY_STEPS - JANUARY_DAYS
    np.testing.assert_array_equal(
        atmosphere[1], [february_days] * 23 + [february_days + 1]
    )
    assert np.all(atmosphere[2:] == 0)
    assert int(atmosphere.sum()) == BOUNDARY_STEPS * HOURS_PER_DAY
    # The daily component is unaffected: its 40 records are labelled 2 January
    # to 10 February.
    np.testing.assert_array_equal(np.asarray(counts["ocn"])[:2], [30, 10])


def test_weaved_monthly_means_match_an_xarray_groupby(weaved):
    """Both components agree with a ``groupby`` of the records they emitted.

    The same contract the un-weaved run is held to, at the resolution the
    records are actually written at: the hourly stream is binned by hour and
    the daily stream by day, and both come out of one accumulator.
    """
    coupler, diagnostics, monthly, accumulator = weaved
    _, counts = accumulator
    means = monthly.finalize(accumulator)
    datasets = coupler.to_xarray(diagnostics)

    hourly = (
        datasets["atm"].mean_air_temperature.groupby("time.month").mean("time")
    )
    np.testing.assert_array_equal(hourly.month.values, [1, 2])
    folded = fold_sub_steps(
        means["atm"]["state"].mean_air_temperature, counts["atm"]
    )
    np.testing.assert_allclose(folded[:2], hourly.values, rtol=1e-5, atol=1e-4)
    # Every other month is empty, and empty means NaN rather than zero.
    assert np.all(np.isnan(folded[2:]))

    daily = (
        datasets["ocn"].sea_surface_temperature.groupby("time.month").mean("time")
    )
    np.testing.assert_allclose(
        np.asarray(means["ocn"]["state"].sea_surface_temperature)[:2],
        daily.values,
        rtol=1e-5,
        atol=1e-4,
    )


def test_weaved_windowed_means_match_the_host_binning(climatology_file):
    """Fixed windows bin the sub-steps by their own labels too.

    A pentad boundary falls at 00:00, i.e. between two hourly records of a
    coupled step, so the same discrepancy the month boundary shows would show
    here -- at every window boundary rather than at every month's.
    """
    coupler = build_weaved_coupler(climatology_file)
    windows = BOUNDARY_STEPS // PENTAD_DAYS
    pentads = windowed_mean(coupler, f"{PENTAD_DAYS} days", n_windows=windows)
    diagnostics, accumulator = run_both_ways(coupler, pentads)
    _, counts = accumulator

    # Every window holds five whole days of hourly records, in every slot.
    np.testing.assert_array_equal(
        np.asarray(counts["atm"]), np.full((windows, HOURS_PER_DAY), PENTAD_DAYS)
    )

    atmosphere = coupler.to_xarray(diagnostics)["atm"]
    elapsed_days = (
        atmosphere["time"].values - np.datetime64("2001-01-01")
    ) / np.timedelta64(1, "D")
    # `ceil(elapsed / window) - 1`: a window is closed at its end, which is
    # where JEM labels the record covering it.
    window = np.ceil(elapsed_days / PENTAD_DAYS).astype(int) - 1
    from_output = (
        atmosphere.mean_air_temperature.assign_coords(window=("time", window))
        .groupby("window")
        .mean("time")
    )
    np.testing.assert_array_equal(from_output.window.values, np.arange(windows))

    folded = fold_sub_steps(
        pentads.finalize(accumulator)["atm"]["state"].mean_air_temperature,
        counts["atm"],
    )
    np.testing.assert_allclose(folded, from_output.values, rtol=1e-5, atol=1e-4)


def test_a_nested_coupler_bins_its_inner_records_the_same_way(climatology_file):
    """A nested coupler's inner steps are records of their own, and bin as such.

    The outer coupler stacks an inner coupler's diagnostics on a leading axis
    of one entry per *inner* coupled step, and labels them at the inner rate --
    so they are the same kind of thing as a repeated component's sub-steps and
    are treated identically, one bin per record, with the counts following the
    inner structure.
    """
    coupler = build_nested_coupler(climatology_file)
    monthly = monthly_mean(coupler)
    diagnostics, accumulator = run_both_ways(coupler, monthly)
    _, counts = accumulator

    assert set(counts) == {"fast"}
    assert set(counts["fast"]) == {"atm", "ocn"}
    for component in ("atm", "ocn"):
        np.testing.assert_array_equal(
            np.asarray(counts["fast"][component])[0],
            [JANUARY_DAYS] * 23 + [30],
        )

    means = monthly.finalize(accumulator)
    hourly = (
        coupler.to_xarray(diagnostics)["ocn"]
        .sea_surface_temperature.groupby("time.month")
        .mean("time")
    )
    folded = fold_sub_steps(
        means["fast"]["ocn"]["state"].sea_surface_temperature,
        counts["fast"]["ocn"],
    )
    np.testing.assert_allclose(folded[:2], hourly.values, rtol=1e-5, atol=1e-4)


def test_weaved_variable_windows_bin_the_sub_steps_by_their_own_labels(
    climatology_file,
):
    """A pattern of windows over an hourly component, record by record.

    The variable-length rule is the same rule at any record rate: the boundary
    of a 2- or 3-day window falls at 00:00, which is the label of the last
    hourly record of the window's last day, and that record closes the window
    rather than opening the next.
    """
    coupler = build_weaved_coupler(climatology_file)
    pattern = (2, 3)
    windows = windowed_mean(
        coupler,
        [f"{days} days" for days in pattern],
        total_time=f"{BOUNDARY_STEPS} days",
    )
    diagnostics, accumulator = run_both_ways(coupler, windows)
    _, counts = accumulator

    # 40 days is eight whole cycles of the 5-day pattern, so nothing is short.
    bins = BOUNDARY_STEPS // sum(pattern) * len(pattern)
    boundaries = cycled_boundaries(pattern, bins)
    lengths = np.diff(np.concatenate([[0], boundaries]))
    np.testing.assert_array_equal(
        np.asarray(counts["atm"]),
        np.repeat(lengths[:, None], HOURS_PER_DAY, axis=1),
    )

    atmosphere = coupler.to_xarray(diagnostics)["atm"]
    window = window_of_label(boundaries, atmosphere["time"].values)
    from_output = (
        atmosphere.mean_air_temperature.assign_coords(window=("time", window))
        .groupby("window")
        .mean("time")
    )
    np.testing.assert_array_equal(from_output.window.values, np.arange(bins))

    folded = fold_sub_steps(
        windows.finalize(accumulator)["atm"]["state"].mean_air_temperature,
        counts["atm"],
    )
    np.testing.assert_allclose(folded, from_output.values, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_gradient_through_the_accumulated_mean(climatology_file, record_months):
    """``jax.grad`` of a monthly-mean SST reaches a carried ocean parameter.

    The reduction lives in the scan carry, so nothing about it is special to
    differentiate -- which is exactly what is asserted: the gradient of the
    accumulated July mean equals the gradient of the same quantity computed
    from the stacked diagnostics.
    """
    coupler = build_coupler(climatology_file)
    monthly = monthly_mean(coupler)
    accumulating = coupler.generate_trajectory_function(
        STEPS_PER_YEAR, accumulate=monthly
    )
    stacking = coupler.generate_trajectory_function(STEPS_PER_YEAR)
    july_records = np.flatnonzero(record_months == 6)

    def initial_carry(relaxation_time):
        return coupler.initialize(
            {
                "ocn": SlabOceanParameters(
                    forcing_method="relaxation", relaxation_time=relaxation_time
                )
            }
        )

    def july_mean_from_accumulator(relaxation_time):
        _, accumulator = accumulating(initial_carry(relaxation_time))
        means = monthly.finalize(accumulator)
        return jnp.mean(means["ocn"]["state"].sea_surface_temperature[6])

    def july_mean_from_stack(relaxation_time):
        _, diagnostics = stacking(initial_carry(relaxation_time))
        sea_surface_temperature = diagnostics["ocn"]["state"].sea_surface_temperature
        return jnp.mean(sea_surface_temperature[july_records])

    relaxation_time = jnp.float32(RELAXATION_TIME)
    np.testing.assert_allclose(
        float(july_mean_from_accumulator(relaxation_time)),
        float(july_mean_from_stack(relaxation_time)),
        rtol=1e-5,
    )

    accumulated_gradient = float(jax.grad(july_mean_from_accumulator)(relaxation_time))
    stacked_gradient = float(jax.grad(july_mean_from_stack)(relaxation_time))

    assert np.isfinite(accumulated_gradient)
    assert accumulated_gradient != 0.0
    np.testing.assert_allclose(accumulated_gradient, stacked_gradient, rtol=1e-4)


def test_calibrating_a_monthly_mean_against_a_target(climatology_file):
    """The calibration loop the design doc documents, run.

    ``docs/source/design/architecture.md`` answers "how do I apply a gradient
    to calibrate monthly values?" with a snippet: build the trajectory with
    ``accumulate=monthly``, take the squared error of
    ``monthly.finalize(...)``'s July mean against a target, differentiate it
    with respect to a carried ocean parameter, and take one plain descent
    step. This *is* that snippet, so the documentation cannot drift away from
    a loop that works: what it asserts is that the gradient is a usable
    number (finite, non-zero) and that the step it implies actually reduces
    the loss.
    """
    coupled = build_coupler(climatology_file)
    ocn = coupled.components["ocn"]
    monthly = monthly_mean(coupled)
    trajectory = coupled.generate_trajectory_function(
        STEPS_PER_YEAR, accumulate=monthly
    )
    JULY = 6  # `finalize`'s leading axis is January first.

    def loss(relaxation_time, target_july_sst):
        params = ocn.params.replace(relaxation_time=relaxation_time)
        _, accumulator = trajectory(coupled.initialize({"ocn": params}))
        july = monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature[
            JULY
        ]
        return jnp.mean((july - target_july_sst) ** 2)

    relaxation_time = jnp.float32(RELAXATION_TIME)
    # A target the run misses, so there is a gradient to follow at all: the
    # July mean this ocean settles at, one kelvin colder.
    target_july_sst = (
        monthly.finalize(trajectory(coupled.initialize())[1])["ocn"][
            "state"
        ].sea_surface_temperature[JULY]
        - 1.0
    )

    before = float(loss(relaxation_time, target_july_sst))
    gradient = float(jax.grad(loss)(relaxation_time, target_july_sst))
    assert before > 0.0
    assert np.isfinite(gradient)
    assert gradient != 0.0

    # One plain descent step. The learning rate is scaled by the parameter and
    # the gradient because `relaxation_time` is of order 1e6 seconds while the
    # loss is a few K^2 -- a bare constant would either do nothing or leave the
    # parameter's physical range. A real calibration hands the same gradient to
    # an optimizer, which does this scaling for it.
    learning_rate = 0.05 * float(relaxation_time) / abs(gradient)
    updated = relaxation_time - learning_rate * gradient
    assert float(loss(updated, target_july_sst)) < before


# ---------------------------------------------------------------------------
# The month-length table
# ---------------------------------------------------------------------------


def test_month_lengths_takes_the_calendar_from_whatever_it_is_given(coupler):
    """A coupler, a calendar name and a year length all name the same table.

    The coupler is the form the reviewer's use needs -- `windowed_mean(coupler,
    month_lengths(coupler), ...)` -- and the other two are what an analysis
    script has when it has no coupler in hand.
    """
    expected = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

    assert month_lengths(coupler) == expected
    assert month_lengths(CALENDAR) == expected
    assert month_lengths(365) == expected
    assert sum(month_lengths(coupler)) == STEPS_PER_YEAR
    assert month_lengths(360) == (30,) * MONTHS_PER_YEAR


def test_month_lengths_refuses_a_calendar_with_no_fixed_table():
    """Leap years change the table from year to year, so there is no table."""
    with pytest.raises(NotImplementedError, match="Gregorian"):
        month_lengths("gregorian")


# ---------------------------------------------------------------------------
# What monthly_mean refuses
# ---------------------------------------------------------------------------


def test_a_gregorian_run_is_refused_with_a_reason(climatology_file):
    """Leap years make the day-of-year to month table non-constant."""
    grid = make_grid()
    coupler = Coupler(
        {"atm": SlabAtmosphereModel(grid)},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar="gregorian",
    )
    with pytest.raises(NotImplementedError, match="Gregorian"):
        monthly_mean(coupler)


def test_a_timestep_that_does_not_divide_the_year_is_refused(climatology_file):
    """A month would then not be a function of the step counter alone."""
    grid = make_grid()
    coupler = Coupler(
        {"atm": SlabAtmosphereModel(grid)},
        coupling_timestep=jdt.to_timedelta(7, "day"),
        start_date=START_DATE,
        calendar=CALENDAR,
    )
    with pytest.raises(ValueError, match="divide the year"):
        monthly_mean(coupler)


def test_a_run_starting_mid_year_bins_from_its_own_dates(climatology_file):
    """The bin is the calendar month, not months since the run started."""
    grid = make_grid()
    coupler = Coupler(
        {"atm": SlabAtmosphereModel(grid)},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=jdt.to_datetime("2001-07-01"),
        calendar=CALENDAR,
    )
    monthly = monthly_mean(coupler)
    trajectory = coupler.generate_trajectory_function(31, accumulate=monthly)
    _, (_, counts) = trajectory(coupler.initialize())

    # 31 daily steps from 1 July are labelled 2 July to 1 August: 30 records
    # in July and one in August.
    expected = np.zeros(MONTHS_PER_YEAR, dtype=int)
    expected[6] = 30
    expected[7] = 1
    np.testing.assert_array_equal(np.asarray(counts), expected)
