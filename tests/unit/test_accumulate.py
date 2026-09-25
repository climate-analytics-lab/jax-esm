"""In-scan diagnostic reduction: the ``accumulate`` hook and the binned means.

The model is a two-slab coupler -- an idealized atmosphere over a relaxing
slab ocean on a 4x3 grid -- run for a whole 365-day year, which is the
shortest run in which every month exists and the boundary case (a step whose
interval's midpoint falls exactly at the start of a month) actually occurs. A
year is also exactly 73 pentads, so the same run fills a ``windowed_mean``
accumulator once with nothing wrapping.

The last two sections run the same pair of slabs *weaved*: the atmosphere
stepped hourly within the daily coupling, once as a repeated workflow and once
as a nested hourly coupler. Those runs are 40 days from 1 January, so they
cross a month boundary -- the case in which a coupled step's sub-steps do not
all belong to the same month.

A dedicated section near the end covers the ``"gregorian"`` calendar: real
leap years, no fixed month table, cross-checked against `pandas`'s own
Gregorian arithmetic rather than jem's own labelling function, so a bug
shared between the two could not hide a disagreement.

Every test here compares the reduction computed *inside* the ``lax.scan``
with the same reduction computed on the host from the stacked diagnostics,
because the point of the hook is to be indistinguishable from the obvious
thing while never materialising the trajectory.
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pandas as pd
import pytest

from jem.accumulate import (
    MONTHS_PER_YEAR,
    _midpoint_month_rule,
    fold_records,
    month_lengths,
    monthly_mean,
    windowed_mean,
)
from jem.base.calendar import max_safe_record
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


def build_coupler(
    climatology_file,
    relaxation_time=RELAXATION_TIME,
    start_date=START_DATE,
    coupling_timestep=COUPLING_TIMESTEP,
) -> Coupler:
    """Return the two-slab coupler the tests run, started on `start_date`."""
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
        coupling_timestep=coupling_timestep,
        start_date=start_date,
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

    Taken **directly** from the ``datetime64`` labels ``Coupler.to_xarray``
    puts on the records (``coupler.time_axis(...).datetimes()``), with no
    correction -- since jax-gcm v3 (PR 878) moved the written label to each
    interval's midpoint, and ``monthly_mean``'s own bin rule matches it (see
    that function's docstring's Breaking-change paragraph), the two are the
    *same instant* by construction, for every calendar. This fixture, and
    every test built on
    it, is therefore itself evidence of that equality: it is comparing
    ``monthly_mean``'s in-scan bins against a mean grouped by the plain,
    uncorrected written labels, which is exactly the user-facing claim
    (``monthly.finalize(accumulator) ==
    coupler.to_xarray(diagnostics).groupby("time.month").mean()``). It is the
    same binning as the model calendar's because this run starts in 2001 and
    so crosses no Gregorian 29 February; where the two calendars part company
    the accumulator follows the model's, which is what the leap-year tests
    below pin.
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
    taken from the written output are the same number -- for a run like this
    one, whose 2001 start crosses no Gregorian 29 February. The bins are the
    *model* calendar's months, and the leap-year tests below cover the year in
    which that is visible.
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

    A **plain** ``groupby("time.month")`` of the written time coordinate, with
    no correction: jax-gcm PR 878 labels an averaged record at its interval's
    **midpoint**, and ``monthly_mean``'s own bin rule matches it exactly (see
    ``jem.accumulate``'s "Which month a record counts in"), so the two are
    the same instant by construction and
    this is the direct user-facing check of that claim -- they agree here
    (also) because a 2001 run's labels cross no Gregorian 29 February -- the
    one year in which the labels and the model calendar disagree *as well* is
    covered below.
    """
    _, diagnostics = stacked_year
    monthly, _, accumulator = accumulated_year

    ocean = coupler.to_xarray(diagnostics)["ocn"]
    from_output = ocean.sea_surface_temperature.groupby("time.month").mean("time")
    # `groupby("time.month")` labels its groups 1-12; `finalize` is 0-indexed,
    # January first -- the two conventions this asserts line up.
    np.testing.assert_array_equal(from_output["month"].values, np.arange(1, 13))

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
    # 58 daily steps from 1 January have midpoints 1 January 12:00 to 27
    # February 12:00 -- 31 records in January, 27 in February -- so March
    # onwards is empty.
    np.testing.assert_array_equal(np.asarray(counts)[:2], [31, 27])
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
# One bin per month of the run
# ---------------------------------------------------------------------------

#: Coupled steps of the sequential-month runs: thirteen months and a few days,
#: so the run passes a whole year and the second July is a bin of its own
#: rather than a second helping of the first.
SEQUENTIAL_STEPS = 400

#: Two mid-year start dates, because the phase is the whole point: one on the
#: 1st of a month (the rotation of the month table is exercised, the offset
#: into the month is zero) and one mid-month at 12:00 (the offset is not even
#: a whole number of coupling steps).
SEQUENTIAL_STARTS = ("2001-07-01", "2001-07-15T12:00:00")


@pytest.fixture(scope="module", params=SEQUENTIAL_STARTS)
def sequential_months(request, climatology_file):
    """Run 400 days from a mid-year start, stacked and binned by month."""
    coupler = build_coupler(
        climatology_file, start_date=jdt.to_datetime(request.param)
    )
    monthly = monthly_mean(coupler, total_time=f"{SEQUENTIAL_STEPS} days")
    _, accumulator = coupler.generate_trajectory_function(
        SEQUENTIAL_STEPS, accumulate=monthly
    )(coupler.initialize())
    _, diagnostics = coupler.generate_trajectory_function(SEQUENTIAL_STEPS)(
        coupler.initialize()
    )
    return coupler, monthly, accumulator, diagnostics


def year_month_bins(times):
    """Return the 0-based position of each label's calendar month, in order."""
    return np.unique(times.astype("datetime64[M]"), return_inverse=True)[1]


def test_sequential_months_bin_the_run_month_by_month(sequential_months):
    """The bins are the months the run passes through, however it is phased.

    The counts are the assertion that matters: they are the number of output
    records the calendar puts in each month, so a bin that had drifted (a
    fixed 30-day window) or that had been phased to January rather than to the
    run's own start date would show up immediately. Binned on the plain,
    uncorrected written labels -- the same instant ``monthly_mean`` itself now
    bins by, exactly (see that function's docstring's "Which month a record
    counts in").
    """
    coupler, _, (_, counts), diagnostics = sequential_months
    del diagnostics
    labels = (coupler.time_axis(0, SEQUENTIAL_STEPS)).datetimes()

    bins = year_month_bins(labels)
    assert counts.shape == (int(bins.max()) + 1,)
    np.testing.assert_array_equal(np.asarray(counts), np.bincount(bins))
    assert int(np.sum(np.asarray(counts))) == SEQUENTIAL_STEPS
    # Interior months hold their own true length, whatever the start date:
    # nothing drifts, and only the first and last bins are partial.
    interior = np.asarray(counts)[1:-1]
    first_full_month = labels[0].astype("datetime64[M]") + 1
    expected = [
        month_lengths(coupler)[
            (first_full_month + offset).astype(int) % MONTHS_PER_YEAR
        ]
        for offset in range(interior.size)
    ]
    np.testing.assert_array_equal(interior, expected)


def test_sequential_months_match_a_year_month_groupby(sequential_months):
    """Every bin equals the mean of that calendar month of the written output.

    The same contract `monthly_mean`'s twelve bins are held to, for the bins
    that do not composite the years: the reduction and a `groupby` of the
    output are the same number, month by month of the run, computed directly
    from the output's own written labels with no correction.
    """
    coupler, monthly, accumulator, diagnostics = sequential_months
    ocean = coupler.to_xarray(diagnostics)["ocn"]
    labels = (coupler.time_axis(0, SEQUENTIAL_STEPS)).datetimes()

    from_output = (
        ocean.sea_surface_temperature.assign_coords(
            year_month=("time", year_month_bins(labels))
        )
        .groupby("year_month")
        .mean("time")
    )
    accumulated = monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature
    np.testing.assert_allclose(
        np.asarray(accumulated), from_output.values, rtol=1e-5, atol=1e-4
    )


def test_a_run_spanning_exactly_one_calendar_month_gets_one_bin(
    coupler, climatology_file
):
    """``total_time`` naming exactly one calendar month never spills into the next.

    Every record is binned by its own **midpoint**, and the last record of a
    ``total_time``-sized run has its midpoint at ``total_time - dt/2`` -- half
    a step *short* of the boundary, never exactly on it. A 31-day run from 1
    January therefore keeps its last record (interval ``[30, 31)`` days,
    midpoint 31 January 12:00) in January, and the run needs only the one
    bin. This is a **breaking change**
    from the pre-2026-09 end-of-interval convention, under which the same
    run's last record's defining instant was exactly the boundary itself
    (``total_time``), so it opened a second, single-record bin -- see
    ``monthly_mean``'s own docstring for the worked ten-year version of this
    same fact.
    """
    del climatology_file
    monthly = monthly_mean(coupler, total_time="31 days")
    _, (_, counts) = coupler.generate_trajectory_function(31, accumulate=monthly)(
        coupler.initialize()
    )

    np.testing.assert_array_equal(np.asarray(counts), [31])
    # The same arithmetic at the length the docstrings quote: ten years of a
    # 365-day calendar is exactly 120 months now, not 121 -- the last
    # record's midpoint stays within December of year 10.
    _, decade_counts = monthly_mean(coupler, total_time="10 years").init()
    assert decade_counts.shape == (10 * MONTHS_PER_YEAR,)


def test_n_months_and_total_time_size_the_same_accumulator(climatology_file):
    """The two ways of sizing the sequential form build the same thing."""
    coupler = build_coupler(
        climatology_file, start_date=jdt.to_datetime("2001-07-01")
    )
    from_total = monthly_mean(coupler, total_time="100 days")
    from_count = monthly_mean(coupler, n_months=4)

    _, first = coupler.generate_trajectory_function(100, accumulate=from_total)(
        coupler.initialize()
    )
    _, second = coupler.generate_trajectory_function(100, accumulate=from_count)(
        coupler.initialize()
    )
    # 100 daily records from 1 July have midpoints 1 July 12:00 to 8 October
    # 12:00: whole July and August, all of September, and 8 days of October.
    np.testing.assert_array_equal(np.asarray(first[1]), [31, 31, 30, 8])
    for got, want in zip(
        jax.tree_util.tree_leaves(from_total.finalize(first)),
        jax.tree_util.tree_leaves(from_count.finalize(second)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_more_months_than_the_run_leaves_them_nan(coupler):
    """Bins past the end of the run are empty, and empty is NaN, not zero."""
    monthly = monthly_mean(coupler, n_months=6)
    _, accumulator = coupler.generate_trajectory_function(31, accumulate=monthly)(
        coupler.initialize()
    )
    _, counts = accumulator

    # A 31-day run from 1 January stays entirely within January under the
    # midpoint convention (its last record's midpoint is 31 January 12:00,
    # short of the February boundary), so February is empty too.
    np.testing.assert_array_equal(np.asarray(counts), [31, 0, 0, 0, 0, 0])
    sea_surface_temperature = np.asarray(
        monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature
    )
    assert np.all(np.isfinite(sea_surface_temperature[:1]))
    assert np.all(np.isnan(sea_surface_temperature[1:]))


def test_a_run_longer_than_n_months_wraps(coupler):
    """`n_months` bins repeat, as `n_windows` windows do.

    Two bins from 1 January are January and February, 59 days of accumulator;
    a 100-day run folds March back into the January bin and the first ten
    days of April into February's (a record's own midpoint, not the day
    number, is what decides it: the 100th record's midpoint is 9 April 12:00,
    still April's 10th day of this fold). That is the fixed-size accumulator
    behaving as it does everywhere else -- size it to the run to avoid it.
    """
    monthly = monthly_mean(coupler, n_months=2)
    _, (_, counts) = coupler.generate_trajectory_function(100, accumulate=monthly)(
        coupler.initialize()
    )

    np.testing.assert_array_equal(np.asarray(counts), [31 + 31, 28 + 10])
    assert int(np.sum(np.asarray(counts))) == 100


def test_the_climatology_is_what_neither_argument_gives(coupler):
    """No size means the twelve calendar months, which is what it always meant."""
    _, counts = monthly_mean(coupler).init()
    assert counts.shape == (MONTHS_PER_YEAR,)


def test_n_months_and_total_time_are_mutually_exclusive(coupler):
    """Two answers to one question."""
    with pytest.raises(ValueError, match="at most one of n_months and total_time"):
        monthly_mean(coupler, n_months=12, total_time="1 year")


@pytest.mark.parametrize("n_months", [0, -3, 1.0, True])
def test_a_bad_n_months_is_refused(coupler, n_months):
    """`True` is an `int` that would silently mean "one month"."""
    with pytest.raises(ValueError, match="n_months must be a positive integer"):
        monthly_mean(coupler, n_months=n_months)


def test_a_coupling_that_does_not_divide_a_month_still_bins_months(
    climatology_file,
):
    """Five-day coupling: the bins are calendar months it does not fit into.

    A five-day step divides the 365-day year but not 59 days of January and
    February, so the span the bins repeat with is not a whole number of steps.
    That span is rounded up to one -- the wrap moves by less than a step and
    every bin boundary stays exact -- rather than refused, which matters for
    an `n_months` accumulator that *is* wrapped into (see
    ``monthly_mean``'s docstring's wrap paragraph). A `total_time`-sized
    accumulator is never wrapped into, so the rounding cannot show up in the
    answer, and this is the test of that: it still equals the groupby.
    """
    coupler = build_coupler(
        climatology_file, coupling_timestep=jdt.to_timedelta(5, "day")
    )
    monthly = monthly_mean(coupler, total_time="1 year")
    steps = STEPS_PER_YEAR // 5
    _, accumulator = coupler.generate_trajectory_function(
        steps, accumulate=monthly
    )(coupler.initialize())
    _, diagnostics = coupler.generate_trajectory_function(steps)(
        coupler.initialize()
    )

    ocean = coupler.to_xarray(diagnostics)["ocn"]
    labels = (coupler.time_axis(0, steps)).datetimes()
    bins = year_month_bins(labels)
    # Exactly twelve months: the run's last record's own midpoint is short of
    # the one-year boundary (half a coupling step, unlike the pre-migration
    # end-of-interval convention, under which it sat exactly on it and opened
    # a thirteenth, single-record bin -- see `monthly_mean`'s docstring).
    assert accumulator[1].shape == (MONTHS_PER_YEAR,)
    np.testing.assert_array_equal(np.asarray(accumulator[1]), np.bincount(bins))

    from_output = (
        ocean.sea_surface_temperature.assign_coords(year_month=("time", bins))
        .groupby("year_month")
        .mean("time")
    )
    np.testing.assert_allclose(
        np.asarray(
            monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature
        ),
        from_output.values,
        rtol=1e-5,
        atol=1e-4,
    )
    # And the decade the docstrings quote builds on this coupling too, which
    # is what the refusal this replaced made impossible.
    _, decade_counts = monthly_mean(coupler, total_time="10 years").init()
    assert decade_counts.shape == (10 * MONTHS_PER_YEAR,)


def test_a_century_sequential_monthly_mean_is_exact_past_68_years(climatology_file):
    """A 100-year sequential ``monthly_mean`` must still work on the 365_day calendar.

    ``_midpoint_month_rule``'s bin rule compares in DAYS, not raw SECONDS,
    against a boundary table built from the pattern's own whole span;
    comparing in seconds instead would overflow int32 for a sequential
    accumulator, whose span is ``n_months`` months of the run -- for 100
    years on this calendar that is about 3.15e9 s, already past ``2**31``
    (the daily-coupling threshold is ``2**31 / 86400 / 365`` years, about
    68). Comparing in days instead (see that function's own docstring) stays
    int32-safe for millions of years. This is jitted,
    as a real trajectory would call it, and checked at the run's very last
    step -- the one a sequential accumulator sized exactly to the run
    actually reaches -- against the plain month arithmetic
    :func:`year_month_bins` also relies on.
    """
    years = 100
    coupler = build_coupler(climatology_file)
    monthly = monthly_mean(coupler, total_time=f"{years} years")
    sums, counts = monthly.init()
    assert counts.shape == (years * MONTHS_PER_YEAR,)

    # One real step gives the diagnostics' exact structure, shape and dtype
    # without integrating 36500 of them.
    _, diagnostics = jax.jit(coupler.generate_step_function())(coupler.initialize())

    late_step = years * STEPS_PER_YEAR - 1  # the run's very last coupled step
    time = coupler.coupling_time(jnp.int32(late_step))
    new_sums, new_counts = jax.jit(monthly.update)((sums, counts), diagnostics, time)

    label = coupler.time_axis(late_step, 1).datetimes()[0]
    start = np.datetime64(coupler.start_date.to_pydatetime())
    expected_bin = int(
        label.astype("datetime64[M]").astype(np.int64)
        - start.astype("datetime64[M]").astype(np.int64)
    )
    assert 0 <= expected_bin < years * MONTHS_PER_YEAR

    got_counts = np.asarray(new_counts)
    np.testing.assert_array_equal(np.flatnonzero(got_counts), [expected_bin])
    assert got_counts[expected_bin] == 1
    del new_sums  # only the bin placement is under test here


def test_midpoint_month_rule_refuses_a_pattern_gregorian_instant_cannot_resolve():
    """A pattern too long for its own record length is refused, not silently wrong.

    ``gregorian_instant`` cannot resolve an arbitrarily large number of
    records exactly for a long enough ``record_seconds`` (see
    ``jem.base.calendar.max_safe_record``); ``_midpoint_month_rule`` knows,
    in plain Python and before any bin is ever computed, exactly how many
    records of the pattern it is about to ask that function to resolve
    (``records_per_period``), so it is expected to raise here rather than
    let a real run silently drift into wrong bins for a century-scale
    accumulator (see
    ``test_a_century_sequential_monthly_mean_is_exact_past_68_years``, which
    is the same failure mode this construction-time check exists to catch
    before it happens).
    """
    record_seconds = 2_629_746  # a "1 month" Gregorian coupling step
    bound = max_safe_record(record_seconds, offset_seconds=record_seconds // 2)
    # One bin, deliberately sized to need ten more records per cycle than
    # `gregorian_instant` can resolve for this record length.
    period = record_seconds * (bound + 11)
    rule = _midpoint_month_rule(np.array([period], dtype=np.int64), 0)

    with pytest.raises(ValueError, match="too long"):
        rule(jnp.int32(0), record_seconds)


def test_gregorian_monthly_mean_refuses_a_total_time_gregorian_instant_cannot_resolve(
    gregorian_coupler,
):
    """The Gregorian sequential form gets the same construction-time refusal.

    `_midpoint_month_rule` (the fixed-calendar path, see the test above)
    already checks this; `_gregorian_monthly_mean`'s own sequential form
    must too, even though it is exactly the same situation -- `total_time`
    fixes the number of records `_gregorian_month_rule.bin_of_record` will
    be asked to resolve, known here in plain Python before any bin is ever
    computed. Refused rather than left to silently bin a too-long run
    wrong, exactly as `run_chunked`'s own equivalent check
    (`_check_step_counters_fit_int32`) is for a run of that length.
    `gregorian_instant`'s own limb-based decomposition makes this bound
    astronomically large for any realistic coupling timestep (millions of
    years), so the run `total_time` names here is deliberately far beyond
    even that.
    """
    dt_seconds = int(round(gregorian_coupler.dt_seconds))
    start = gregorian_coupler.start_date
    bound = max_safe_record(
        dt_seconds, offset_seconds=dt_seconds // 2,
        start_seconds=int(start.delta.seconds), start_days=int(start.delta.days),
    )
    too_many_days = (bound + 10) * dt_seconds // 86400

    with pytest.raises(ValueError, match="too long to bin exactly"):
        monthly_mean(gregorian_coupler, total_time=f"{too_many_days} days")


def test_gregorian_monthly_mean_refuses_cleanly_past_datetime_year_9999(
    gregorian_coupler,
):
    """A raw ``OverflowError`` is not a refusal.

    ``_gregorian_monthly_mean``'s ``total_time`` path counts the run's own
    span of calendar months on the host with Python's ``datetime`` (see the
    test above for the ``gregorian_instant`` int32 bound this is separate
    from, and astronomically larger than): ``datetime`` itself cannot
    represent a year past 9999, an ordinary Python limitation that binds
    thousands of years before ``max_safe_record``'s own int32 bound ever
    would. Left unguarded, a ``total_time`` whose last record's midpoint
    falls past year 9999 -- ``"3000000 days"`` from 2000-01-01 is about 8219
    years, well past it -- raises a raw ``OverflowError: date value out of
    range`` from ``datetime`` arithmetic instead of a clear ``ValueError``
    naming the actual limit.
    """
    with pytest.raises(ValueError, match="datetime"):
        monthly_mean(gregorian_coupler, total_time="3000000 days")


def test_a_wrapped_sequential_month_straddles_two_bins(coupler):
    """What wrapping an `n_months` accumulator really does, said honestly.

    The wrap is modular in elapsed time over the **span** of the bins, not in
    months, so it realigns with the calendar only when that span is a whole
    number of years -- `n_months` a multiple of twelve. Six bins from 1
    January span 181 days, so nine months of run folds the second August
    across two of them. This is why `total_time`, which is never wrapped into,
    is the form to prefer, and it is pinned here so the docstring cannot
    quietly go back to claiming that month `n_months` lands in bin 0.
    """
    steps = 270
    monthly = monthly_mean(coupler, n_months=6)
    _, (_, counts) = coupler.generate_trajectory_function(steps, accumulate=monthly)(
        coupler.initialize()
    )

    # The plain written labels, matching `monthly_mean`'s own bin math exactly
    # (both are the record's midpoint now).
    labels = (coupler.time_axis(0, steps)).datetimes()
    elapsed = (labels - np.datetime64("2001-01-01")) / np.timedelta64(1, "D")
    span = np.cumsum(month_lengths(coupler)[:6])
    # A month is closed at its start, so the bin of a label is the number of
    # boundaries at or before it -- of its elapsed time reduced modulo the span.
    host = np.searchsorted(span, elapsed % span[-1], side="right")
    np.testing.assert_array_equal(np.asarray(counts), np.bincount(host, minlength=6))

    august = labels.astype("datetime64[M]") == np.datetime64("2001-08")
    august_bins, august_counts = np.unique(host[august], return_counts=True)
    np.testing.assert_array_equal(august_bins, [1, 2])
    np.testing.assert_array_equal(august_counts, [28, 3])


def test_folding_records_of_a_component_without_sub_steps_changes_nothing(
    accumulated_year,
):
    """`fold_records` is safe on a component that records once per step.

    The documented promise is that a caller need not know which kind of
    component it has: with no sub-step axes there is nothing to fold, and the
    means come back as they went in.
    """
    monthly, _, accumulator = accumulated_year
    _, counts = accumulator
    means = monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature

    folded = fold_records(means, counts)
    np.testing.assert_array_equal(np.asarray(folded), np.asarray(means))


# ---------------------------------------------------------------------------
# The bins' calendar and the labels' calendar
# ---------------------------------------------------------------------------

#: A start date at which the two calendars in play part company. The run's
#: labels are proleptic Gregorian whatever the model calendar is
#: (`TimeAxis.datetimes`, JCM's convention), and 2000 is a Gregorian leap
#: year, so from 29 February on every label is a day behind the model's own
#: date. It is also where the shipped examples start, so this is the run a
#: user following the documentation makes.
LEAP_START_DATE = "2000-01-01"

#: The record whose label is 2000-02-29: its interval is ``[59, 60)`` model
#: days from the start (day 59, 0-indexed, is the leap day itself in the real
#: Gregorian calendar), so its midpoint -- the label
#: :meth:`~jem.base.component.TimeAxis.datetimes` writes -- is
#: ``2000-02-29T12:00``. The 365-day model calendar has no such date at all,
#: and calls that same instant (day 59 of its own fixed table, the 31 days of
#: January plus the first 28 of a Feb that never reaches a 29th) 1 March.
LEAP_DAY_RECORD = 59

#: The record whose instant the model calls 00:00 on 1 April: that is
#: 31 + 28 + 31 = 90 model days after the start, and record `k`'s midpoint is
#: at day `k + 0.5`, so it is record 90 (midpoint day 90.5, still within the
#: model's own March -- the model's April starts at day 90 exactly). The
#: Gregorian labels write its midpoint as 2000-03-31T12:00.
APRIL_RECORD = 90

#: Month lengths of the 365-day calendar, written out so that a test asserting
#: the accumulator follows the model calendar does not ask the module under
#: test what that calendar is.
MONTH_LENGTHS_365 = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@pytest.fixture(scope="module")
def leap_year(climatology_file):
    """Run a year from 1 January 2000, stacked and binned into twelve months."""
    coupler = build_coupler(
        climatology_file, start_date=jdt.to_datetime(LEAP_START_DATE)
    )
    monthly = monthly_mean(coupler)
    _, accumulator = coupler.generate_trajectory_function(
        STEPS_PER_YEAR, accumulate=monthly
    )(coupler.initialize())
    _, diagnostics = coupler.generate_trajectory_function(STEPS_PER_YEAR)(
        coupler.initialize()
    )
    return coupler, monthly, accumulator, diagnostics


def model_calendar_months(coupler, n_records):
    """Return each record's 0-based month in the *model* calendar.

    The host-side binning the accumulator has to reproduce: record `k`'s
    midpoint is at day `k + 0.5` of a run starting on 1 January, and the
    model's year is the fixed month table -- no `datetime64` anywhere. The
    modulo is the year wrapping into bin 0, which is what makes the twelve
    bins a climatology.
    """
    day_of_year_midpoint = (np.arange(n_records) + 0.5) % int(coupler.days_per_year)
    month_starts = np.cumsum((0,) + MONTH_LENGTHS_365[:-1])
    return np.searchsorted(month_starts, day_of_year_midpoint, side="right") - 1


def test_a_leap_year_start_bins_by_the_model_calendar_not_the_label(leap_year):
    """The record labelled 2000-02-29 is March's, because the model says so.

    The bins are the model calendar's months and the labels are proleptic
    Gregorian, so on a 365-day calendar the instant written `2000-02-29T12:00`
    is the model's 1 March and is accumulated into March. The twelve counts are
    therefore the 365-day month lengths whatever the labels read, and every
    mean is the mean of the records a host-side binning by model day-of-year
    puts in that month -- which is the binning the forcing and the seasonal
    cycle follow.
    """
    coupler, monthly, accumulator, diagnostics = leap_year
    _, counts = accumulator

    labels = (coupler.time_axis(0, STEPS_PER_YEAR)).datetimes()
    assert labels[LEAP_DAY_RECORD] == np.datetime64("2000-02-29T12:00")
    np.testing.assert_array_equal(np.asarray(counts), MONTH_LENGTHS_365)

    months = model_calendar_months(coupler, STEPS_PER_YEAR)
    assert months[LEAP_DAY_RECORD] == 2      # March, not the label's February
    expected = host_monthly_means(diagnostics, months)
    actual = monthly.finalize(accumulator)
    for index, (got, want) in enumerate(
        zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        )
    ):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-4,
            err_msg=f"leaf {index}",
        )


def test_a_run_stopping_on_the_leap_day_counts_that_record_in_march(
    climatology_file,
):
    """The accumulator itself, not just a host binning, puts it in bin 2.

    60 daily steps from 1 January 2000 have midpoints 1 January 12:00 to
    29 February 12:00 (real Gregorian); on the model calendar those same 60
    records are 31 full days of January, 28 of February, and one -- record 59,
    whose midpoint is the Gregorian leap day itself -- that opens March.
    """
    coupler = build_coupler(
        climatology_file, start_date=jdt.to_datetime(LEAP_START_DATE)
    )
    monthly = monthly_mean(coupler)
    _, (_, counts) = coupler.generate_trajectory_function(
        LEAP_DAY_RECORD + 1, accumulate=monthly
    )(coupler.initialize())

    expected = np.zeros(MONTHS_PER_YEAR, dtype=int)
    expected[:3] = (31, 28, 1)
    np.testing.assert_array_equal(np.asarray(counts), expected)


def test_leap_year_labels_run_a_day_behind_the_model_calendar(leap_year):
    """From the Gregorian 29 February on, a label is a day early.

    The consequence documented on `TimeAxis` and under `monthly_mean`'s **Leap
    days on the fixed calendars**: the labels are a plain count of Gregorian
    days (offset to each record's own midpoint), so a 365-day run loses a day
    to them at each leap day and keeps it for the rest of the Gregorian year.
    """
    coupler, _, _, _ = leap_year
    labels = (coupler.time_axis(0, STEPS_PER_YEAR)).datetimes()

    # Up to the leap day the two calendars still agree.
    assert labels[LEAP_DAY_RECORD - 1] == np.datetime64("2000-02-28T12:00")
    # And from it on the label is one day earlier than the model's own date:
    # the model calls these instants 1 March and 1 April.
    assert labels[LEAP_DAY_RECORD] == np.datetime64("2000-02-29T12:00")
    assert labels[APRIL_RECORD] == np.datetime64("2000-03-31T12:00")


def test_a_leap_year_groupby_of_the_written_output_differs_as_documented(leap_year):
    """`finalize` and `groupby("time.month")` part company, in the stated way.

    Under the midpoint convention this collapses to exactly two months, not
    the whole March-onward cascade the pre-migration end-of-interval
    convention produced (every record's REAL-calendar month now equals its
    MODEL-calendar month except across the one genuine difference between the
    two calendars: real February has a 29th day the model's never does).
    February's written labels therefore hold 29 records (its own real days,
    leap day included) against the model's 28; and since the real year (2000,
    a leap year) is 366 days while this run is only ``STEPS_PER_YEAR = 365``
    records long, the run's labels never reach real 31 December at all, so
    December's written labels hold only 30 (real 1-30 December) against the
    model's clean 31. Every other month -- including March, which the
    pre-migration test used as its worked example -- now agrees exactly,
    because the boundary convention that used to shift every month from March
    on by one record is gone (see ``monthly_mean``'s Breaking-change
    paragraph); what is left is only the leap-day / short-run difference this
    test now pins.
    """
    coupler, monthly, accumulator, diagnostics = leap_year
    _, counts = accumulator

    ocean = coupler.to_xarray(diagnostics)["ocn"]
    month = (
        (coupler.time_axis(0, STEPS_PER_YEAR)).datetimes()
        .astype("datetime64[M]").astype(int) % MONTHS_PER_YEAR
    )
    from_output = (
        ocean.sea_surface_temperature.assign_coords(month=("time", month))
        .groupby("month").mean("time")
    )
    from_labels = np.bincount(month, minlength=MONTHS_PER_YEAR)

    np.testing.assert_array_equal(np.asarray(counts)[:2], [31, 28])
    np.testing.assert_array_equal(from_labels[:2], [31, 29])
    # Every month but February and December now agrees exactly.
    np.testing.assert_array_equal(from_labels[2:11], np.asarray(counts)[2:11])
    assert from_labels[11] == np.asarray(counts)[11] - 1  # December: 30 vs 31

    accumulated = np.asarray(
        monthly.finalize(accumulator)["ocn"]["state"].sea_surface_temperature
    )
    # February is the clearest case now: 29 records against 28, and the
    # ocean's seasonal cycle makes that a difference far above float32 noise
    # (the accumulated means match the model-calendar binning to the float32
    # tolerance asserted above, rtol 1e-5 / atol 1e-4).
    assert np.max(np.abs(accumulated[1] - from_output.values[1])) > 0.01


def test_sequential_months_over_a_leap_year_keep_the_model_month_lengths(
    climatology_file,
):
    """The one-bin-per-month form bins by the model calendar too.

    Sized by `total_time` the bins never wrap: a run of exactly one model year
    (``STEPS_PER_YEAR`` days) gets exactly the twelve model month lengths, the
    same clean result as the twelve-bin climatology
    (`test_a_leap_year_start_bins_by_the_model_calendar_not_the_label`) --
    unaffected by the real calendar's leap day, which is a property of the
    *labels*, not of this reduction's own bin math.
    """
    coupler = build_coupler(
        climatology_file, start_date=jdt.to_datetime(LEAP_START_DATE)
    )
    monthly = monthly_mean(coupler, total_time=f"{STEPS_PER_YEAR} days")
    _, (_, counts) = coupler.generate_trajectory_function(
        STEPS_PER_YEAR, accumulate=monthly
    )(coupler.initialize())

    np.testing.assert_array_equal(np.asarray(counts), MONTH_LENGTHS_365)


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

    This is the convention the docstring commits to -- a record is binned
    against its interval's *end* (elapsed time from the run's start, not the
    record's own written label, which is its midpoint), and a window is
    closed at its end, so the record covering ``[day 4, day 5)`` -- labelled
    at its midpoint, day 4.5 -- finishes the first pentad rather than starting
    the second. A forecast's "first pentad" is days 1-5, and an off-by-one
    here would make it days 1-4.
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
    # `ceil(elapsed / window) - 1`: the window a label belongs to when a
    # window is the half-open interval (w*window, (w+1)*window] -- closed at
    # the end. `elapsed_days` is the record's own *written* label (its
    # midpoint), not its interval's end, but the two never straddle a
    # different multiple of `window` (the window is several records long),
    # so `ceil` gives the same answer either way -- see `windowed_mean`'s
    # docstring for why the two conventions agree at a grid-aligned boundary.
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

#: Coupled steps of the runs that cross a month boundary and then a year:
#: 1 January 2001 to 5 February 2002.
MONTH_WINDOW_STEPS = 400


def cycled_boundaries(lengths, bins):
    """Return the ends, in days, of `bins` windows cycling through `lengths`."""
    return np.cumsum([lengths[index % len(lengths)] for index in range(bins)])


def window_of_label(boundaries_days, times):
    """Return the window each `datetime64` label falls in, binned on the host.

    A window is closed at its end, so the window of a label is the first
    boundary at or after its elapsed time -- `searchsorted(..., side="left")`.
    A record's own written label is now its midpoint, never exactly on a
    whole-day boundary, so `side` cannot actually change the answer here (see
    `windowed_mean`'s docstring for why); it is kept explicit anyway, because
    it is still what the convention means.
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


def test_a_month_long_window_and_a_month_now_agree_at_their_shared_boundary(coupler):
    """The convention gap this test used to pin no longer exists.

    Before the 2026-09 jax-gcm-878 migration, `windowed_mean` and
    `monthly_mean` both bound a record by its interval's END but closed a
    shared boundary in *opposite* directions, so the record whose interval
    ended exactly at 00:00 on 1 February was the **last** of a 31-day window
    started on 1 January but the **first** of February's calendar-month bin.
    `monthly_mean`'s rule moved to the record's MIDPOINT to keep pace with
    ``TimeAxis``'s own new midpoint labels (see that function's docstring's
    Breaking-change paragraph); `windowed_mean`'s did not, since its own
    convention has nothing to do with the output label (see its own
    docstring). The two now agree at any *grid-aligned* shared boundary
    regardless -- this is the same record, still the last one before the
    boundary, in **both** builders now.
    """
    last_january_record = month_lengths(coupler)[0] - 1  # 0-indexed: day 30
    labels = (coupler.time_axis(0, MONTH_WINDOW_STEPS)).datetimes()
    window = window_of_label(
        cycled_boundaries(month_lengths(coupler), MONTHS_PER_YEAR + 2), labels
    )
    month = labels.astype("datetime64[M]").astype(int) % MONTHS_PER_YEAR

    assert int(window[last_january_record]) == 0
    assert int(month[last_january_record]) == 0


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

    This is `jem.accumulate.fold_records`, the helper the docstrings point a
    user at, so the tests below check the shipped fold rather than a second
    implementation of it that could agree with nothing.
    """
    return np.asarray(fold_records(means, counts))


@pytest.fixture(scope="module")
def weaved(climatology_file):
    """Return the weaved coupler, its stacked run and its monthly accumulator."""
    coupler = build_weaved_coupler(climatology_file)
    monthly = monthly_mean(coupler)
    diagnostics, accumulator = run_both_ways(coupler, monthly)
    return coupler, diagnostics, monthly, accumulator


def test_a_sub_stepped_component_is_counted_record_by_record(weaved):
    """Each hourly record lands in the month its own midpoint falls in.

    Every hour-of-day slot behaves identically here, which is itself the
    point: a slot's midpoint offset from the hour (30 minutes, for an hourly
    sub-step) is always under a day, so it can never itself cross the
    January/February boundary regardless of which hour-of-day the slot is --
    unlike the pre-migration end-labelled convention, under which the
    23:00-00:00 slot's record was labelled exactly on the boundary and so
    counted differently from the other 23. Binning the whole coupled step by
    one label (rather than each of its 24 sub-step records by its own) would
    still put all 24 of the boundary day's records on one side of it and
    disagree with the written output by a whole day of records.
    """
    _, _, _, (_, counts) = weaved

    # One count array per component, because the two record at different rates
    # and so fill different bins as one coupled step is folded in.
    assert set(counts) == {"atm", "ocn"}
    assert counts["atm"].shape == (MONTHS_PER_YEAR, HOURS_PER_DAY)
    assert counts["ocn"].shape == (MONTHS_PER_YEAR,)

    atmosphere = np.asarray(counts["atm"])
    february_days = BOUNDARY_STEPS - JANUARY_DAYS
    np.testing.assert_array_equal(atmosphere[0], [JANUARY_DAYS] * HOURS_PER_DAY)
    np.testing.assert_array_equal(atmosphere[1], [february_days] * HOURS_PER_DAY)
    assert np.all(atmosphere[2:] == 0)
    assert int(atmosphere.sum()) == BOUNDARY_STEPS * HOURS_PER_DAY
    # The daily component: its 40 records have midpoints 1 January 12:00
    # through 9 February 12:00, 31 in January and 9 in February.
    np.testing.assert_array_equal(
        np.asarray(counts["ocn"])[:2], [JANUARY_DAYS, february_days]
    )


def test_weaved_monthly_means_match_an_xarray_groupby(weaved):
    """Both components agree with a ``groupby`` of the records they emitted.

    The same contract the un-weaved run is held to, at the resolution the
    records are actually written at: the hourly stream is binned by hour and
    the daily stream by day, and both come out of one accumulator. No
    correction is needed on either stream's own written labels any more --
    ``monthly_mean`` bins every record by the same midpoint
    :class:`~jem.base.component.TimeAxis` labels it with, at whatever rate it
    records.
    """
    coupler, diagnostics, monthly, accumulator = weaved
    _, counts = accumulator
    means = monthly.finalize(accumulator)
    datasets = coupler.to_xarray(diagnostics)

    hourly = datasets["atm"].mean_air_temperature.groupby("time.month").mean("time")
    np.testing.assert_array_equal(hourly["month"].values, [1, 2])
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
    # `ceil(elapsed / window) - 1`: a window is closed at its end; `ceil` of
    # the record's own midpoint label agrees with `ceil` of its true interval
    # end here too (see `windowed_mean`'s docstring).
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
            [JANUARY_DAYS] * HOURS_PER_DAY,
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

    The coupler is the form a caller with one in hand needs --
    `windowed_mean(coupler, month_lengths(coupler), ...)` -- and the other
    two are what an analysis script has when it has no coupler in hand.
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


def test_a_timestep_that_does_not_divide_the_year_is_refused(climatology_file):
    """On a fixed calendar, a month would then not be a function of the step
    counter alone -- ``gregorian`` has no such restriction (below).
    """
    grid = make_grid()
    coupler = Coupler(
        {"atm": SlabAtmosphereModel(grid)},
        coupling_timestep=jdt.to_timedelta(7, "day"),
        start_date=START_DATE,
        calendar=CALENDAR,
    )
    with pytest.raises(ValueError, match="divide the year"):
        monthly_mean(coupler)


# ---------------------------------------------------------------------------
# The "gregorian" calendar
# ---------------------------------------------------------------------------

#: A run long enough to touch every one of the four Gregorian century cases
#: (`jem.base.calendar_test` cross-checks the vendored arithmetic itself over
#: 400+ years; this run only needs to be long enough to cross a real leap day,
#: which a single non-leap-adjacent year already does not, so this picks a
#: run straddling 2000's 29 February specifically).
GREGORIAN_START_DATE = "1999-11-15"
GREGORIAN_STEPS = 120  # 15 Nov 1999 to 14 Mar 2000: crosses the leap day.


@pytest.fixture(scope="module")
def gregorian_coupler(climatology_file):
    """Build a two-slab coupler on the (now default) ``gregorian`` calendar.

    Built without `build_coupler`'s own `calendar=CALENDAR` ("365_day"): this
    deliberately exercises `Coupler`'s own default rather than overriding it,
    since `"gregorian"` is what every undecorated `Coupler(...)` means now.
    """
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
        start_date=jdt.to_datetime(GREGORIAN_START_DATE),
    )


def test_gregorian_no_longer_needs_a_fixed_table_or_a_divides_the_year_check(
    gregorian_coupler,
):
    """``monthly_mean`` builds on ``gregorian`` directly now -- no refusal.

    Confirmed working, both forms, cross-checked below against `pandas`'s own
    Gregorian arithmetic (real leap years). This also needs no
    "coupling timestep divides the year" check at all -- unlike the fixed
    calendars, ``gregorian`` reads a record's real calendar month directly off
    its exact date rather than reducing a step counter modulo a table.

    ``"3650 days"``, not ``"10 years"``: a *literal* ``"10 years"`` is JEM's
    own duration parser's fixed-average-year approximation
    (``days_per_year("gregorian") == 365.2425``), so it is ``3652.425`` days
    -- never a whole number of this coupler's daily coupling steps, whatever
    the real (leap or non-leap) Gregorian years the run's actual dates would
    cross -- and ``monthly_mean`` refuses a ``total_time`` that is not a
    whole number of coupling steps, exactly as ``jem.driver.run_chunked``
    already refuses the same duration for its own ``total_time``/``chunk``.
    ``"3650 days"`` is a plain, exact duration that this daily-coupling
    ``gregorian_coupler`` can actually be run for.
    """
    monthly_mean(gregorian_coupler)  # twelve-bin form: does not raise
    monthly_mean(gregorian_coupler, total_time="3650 days")  # sequential: does not raise
    assert gregorian_coupler.calendar == "gregorian"


def test_gregorian_monthly_means_match_a_pandas_groupby_across_a_leap_day(
    gregorian_coupler,
):
    """The midpoint-binning equality, proven independently of jem's own labels.

    This is the equality ``monthly_mean``'s docstring claims -- bins equal
    ``groupby("time.month")`` of the written output *by construction* -- but
    checked against `pandas`'s own Gregorian ``Timestamp`` arithmetic rather
    than against ``TimeAxis.datetimes()`` itself, so a bug shared between the
    bin rule and the labelling function could not hide the disagreement. The
    run straddles the real 2000-02-29, which the twelve-bin form composites
    into February like any other February day (real leap years, not a fixed
    table) and the leap day itself is not a special case in the code at all.
    """
    monthly = monthly_mean(gregorian_coupler)
    trajectory = gregorian_coupler.generate_trajectory_function(
        GREGORIAN_STEPS, accumulate=monthly
    )
    _, accumulator = trajectory(gregorian_coupler.initialize())
    _, diagnostics = gregorian_coupler.generate_trajectory_function(GREGORIAN_STEPS)(
        gregorian_coupler.initialize()
    )

    start = pd.Timestamp(GREGORIAN_START_DATE)
    dt = pd.Timedelta(days=1)
    midpoints = pd.date_range(start, periods=GREGORIAN_STEPS, freq="D") + dt / 2
    month = np.asarray(midpoints.month) - 1  # 0-indexed, January first

    ocean = gregorian_coupler.to_xarray(diagnostics)["ocn"]
    # The written labels agree with the independent pandas computation, which
    # is itself part of the claim: jem's labels and jem's bins and pandas's
    # own Gregorian calendar are all the same calendar.
    np.testing.assert_array_equal(
        ocean["time"].values.astype("datetime64[us]"),
        midpoints.values.astype("datetime64[us]"),
    )

    expected = host_monthly_means(diagnostics, month)
    actual = monthly.finalize(accumulator)
    for index, (got, want) in enumerate(
        zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        )
    ):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-4,
            err_msg=f"leaf {index}",
        )

    from_output = ocean.sea_surface_temperature.groupby("time.month").mean("time")
    accumulated = actual["ocn"]["state"].sea_surface_temperature
    got_months = np.asarray(from_output["month"].values) - 1
    np.testing.assert_allclose(
        np.asarray(accumulated)[got_months],
        from_output.values,
        rtol=1e-5, atol=1e-4,
    )


def test_gregorian_sequential_months_size_and_bin_correctly(gregorian_coupler):
    """The sequential form's host-sized bin count, cross-checked against pandas.

    ``n_bins`` is computed on the host from the run's first and last records'
    own midpoints (see ``monthly_mean``'s **Sequential-form bin 0**); this
    checks that count, and every bin's membership, against an independent
    `pandas` computation of the same thing.
    """
    monthly = monthly_mean(gregorian_coupler, total_time=f"{GREGORIAN_STEPS} days")
    trajectory = gregorian_coupler.generate_trajectory_function(
        GREGORIAN_STEPS, accumulate=monthly
    )
    _, (_, counts) = trajectory(gregorian_coupler.initialize())

    start = pd.Timestamp(GREGORIAN_START_DATE)
    dt = pd.Timedelta(days=1)
    midpoints = pd.date_range(start, periods=GREGORIAN_STEPS, freq="D") + dt / 2
    year_month = midpoints.year * 12 + midpoints.month
    bins = year_month - int(year_month[0])

    expected_n_bins = int(bins.max()) + 1
    assert counts.shape == (expected_n_bins,)
    np.testing.assert_array_equal(np.asarray(counts), np.bincount(bins))


def test_the_documented_host_side_monthly_recipes(gregorian_coupler):
    """Every host-side monthly-binning recipe the docs point at actually runs.

    `jem/config/coupled_run/long_run.yaml`'s comment used to point at
    ``ds.groupby("time.year").groupby("time.month")`` as the host-side
    fallback for a Gregorian run (from before item A made `monthly_mean`
    itself handle `"gregorian"` in-scan) -- that recipe was never actually
    run and raises `AttributeError` (a `*GroupBy` object has no `.groupby` of
    its own). This pins the two recipes that replaced it, so neither can
    silently break the same way.
    """
    _, diagnostics = gregorian_coupler.generate_trajectory_function(GREGORIAN_STEPS)(
        gregorian_coupler.initialize()
    )
    ocean = gregorian_coupler.to_xarray(diagnostics)["ocn"]

    with pytest.raises(AttributeError):
        ocean.groupby("time.year").groupby("time.month")

    resampled = ocean.sea_surface_temperature.resample(time="MS").mean()
    n_calendar_months = resampled.sizes["time"]
    assert n_calendar_months > 0

    # `groupby` on a LIST of keys returns the full (year, month) cross
    # product, most of which is NaN for a run shorter than a year -- so this
    # compares them by picking out, for each of `resample`'s months, the
    # matching (year, month) slice of the `groupby` result, rather than by
    # shape (which differ) or a flattened sort (which pads with NaN).
    grouped = (
        ocean.sea_surface_temperature.groupby(["time.year", "time.month"]).mean()
    )
    assert int((~grouped.isnull()).any(("lat", "lon")).sum()) == n_calendar_months
    for label in resampled["time"].values:
        timestamp = pd.Timestamp(label)
        from_resample = resampled.sel(time=label)
        from_groupby = grouped.sel(year=timestamp.year, month=timestamp.month)
        np.testing.assert_allclose(
            from_resample.values, from_groupby.values, rtol=1e-5, atol=1e-4
        )


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

    # 31 daily steps from 1 July have midpoints 1 July 12:00 to 31 July
    # 12:00: all 31 stay in July, none spilling into August.
    expected = np.zeros(MONTHS_PER_YEAR, dtype=int)
    expected[6] = 31
    np.testing.assert_array_equal(np.asarray(counts), expected)


def test_a_misspelled_inclusive_is_refused():
    """``Literal`` does not check at runtime, so the rule builder must.

    Without this a misspelling would compare unequal to ``"right"`` and be
    treated as ``"left"``, moving every bin boundary by one record with no
    error anywhere.
    """
    from jem.accumulate import _variable_window_rule

    with pytest.raises(ValueError, match="left.*right.*rigth"):
        _variable_window_rule(np.array([5 * 86400]), 0, "rigth")  # type: ignore[arg-type]


def test_a_duration_that_is_not_whole_seconds_is_refused(coupler):
    """A fractional second is refused, not rounded into a moved boundary."""
    with pytest.raises(ValueError, match="not a whole number of seconds"):
        windowed_mean(coupler, "0.00001 days", n_windows=4)


def test_a_float_representation_of_whole_seconds_is_accepted():
    """A days value that is whole seconds up to float rounding is not refused."""
    from jem.accumulate import _exact_seconds

    assert _exact_seconds(11 / 86400 * 86400, "window") == 11   # 10.999999999999998
    assert _exact_seconds(3600.0, "window") == 3600
    with pytest.raises(ValueError, match="not a whole number of seconds"):
        _exact_seconds(10.5, "window")


# ---------------------------------------------------------------------------
# monthly_mean(total_time=...) must be a whole number of coupling steps,
# exactly like run_chunked's own total_time/chunk (jem.driver._whole_steps)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_time", ["36.5 hours", "1.5 days"])
def test_monthly_mean_refuses_a_total_time_that_is_not_whole_coupling_steps(
    coupler, total_time
):
    """A ``total_time`` shorter than a whole number of coupling steps is refused.

    Before this fix, ``total_time=`` was silently floor-divided by the
    coupling timestep (``n_steps = total_seconds // dt_seconds``), with no
    check that the division was exact -- so ``"36.5 hours"`` on this coupler's
    daily coupling built a one-bin accumulator as if it had been asked for
    exactly one day, discarding the extra 12.5 hours with no warning.
    ``jem.driver.run_chunked`` already refuses the same durations for its own
    ``total_time``/``chunk`` (:func:`jem.driver._whole_steps`); an
    accumulator sized by ``total_time`` is bound by the same coupled-step
    granularity a run is, so it is refused here the same way.
    """
    with pytest.raises(ValueError, match="whole"):
        monthly_mean(coupler, total_time=total_time)


def test_gregorian_monthly_mean_refuses_a_total_time_that_is_not_whole_coupling_steps(
    gregorian_coupler,
):
    """The same refusal, on the ``"gregorian"`` path (a separate code path).

    ``"10 years"`` is JEM's own duration parser's fixed-average-year
    approximation (``jem.base.component.days_per_year("gregorian") ==
    365.2425``), so it is ``3652.425`` days -- not a whole number of this
    coupler's daily coupling steps -- regardless of how many real (leap or
    non-leap) Gregorian years the run's actual dates would cross. Before this
    fix, ``_gregorian_monthly_mean`` silently floor-divided this to 3652
    steps and built a 120-bin accumulator as if the run were exactly that
    long.
    """
    with pytest.raises(ValueError, match="whole"):
        monthly_mean(gregorian_coupler, total_time="10 years")
    with pytest.raises(ValueError, match="whole"):
        monthly_mean(gregorian_coupler, total_time="36.5 hours")
    # A duration that IS a whole number of this coupler's (daily) steps is
    # unaffected -- this is a stricter check, not a more restrictive one.
    monthly_mean(gregorian_coupler, total_time="10 days")
