"""In-scan diagnostic reduction: the ``accumulate`` hook and monthly means.

The model is a two-slab coupler -- an idealized atmosphere over a relaxing
slab ocean on a 4x3 grid -- run for a whole 365-day year, which is the
shortest run in which every month exists and the boundary cases (a step whose
interval ends exactly on the first of a month, and the last step of the year,
which is labelled 1 January of the next one) actually occur.

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

from jem.accumulate import MONTHS_PER_YEAR, monthly_mean
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
