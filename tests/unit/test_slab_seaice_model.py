"""Tests for `jem.components.slab.slab_seaice_model`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jem import constants
from jem.components.slab.slab_seaice_model import (
    ICE_FREE_SST_EXCESS,
    SlabSeaiceModel,
    SlabSeaiceParameters,
)
from tests.unit.slab_test_utils import (
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    coupling_time,
    make_grid,
    tree_signature,
)


@pytest.fixture
def uniform_grid():
    """Return a tiny all-ocean 4x3 lon-lat grid the tests own."""
    return make_grid()


@pytest.fixture
def half_land_grid():
    """Return a 4x3 grid whose eastern half is land."""
    shape = (len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))
    fractional_mask = jnp.where(
        jnp.arange(shape[0])[:, None] >= shape[0] // 2,
        jnp.ones(shape),
        jnp.zeros(shape),
    )
    return make_grid(fractional_mask=fractional_mask)


def _step_with_sst(model, sea_surface_temperature, step=0):
    """Return the carry after one step forced with ``sea_surface_temperature``."""
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        sea_surface_temperature=jnp.broadcast_to(
            jnp.asarray(sea_surface_temperature, dtype=jnp.float32),
            model.grid.shape,
        )
    )
    stepped, _ = model.step(carry, coupling_time(step))
    return stepped


@pytest.mark.parametrize(
    "excess, expected",
    [
        (-5.0, 1.0),                         # well below freezing: full cover
        (0.0, 1.0),                          # at the freezing point: full cover
        (0.25 * ICE_FREE_SST_EXCESS, 0.75),  # on the ramp
        (0.5 * ICE_FREE_SST_EXCESS, 0.5),
        (ICE_FREE_SST_EXCESS, 0.0),          # the top of the ramp: ice-free
        (10.0, 0.0),                         # warm water: ice-free
    ],
)
def test_ice_fraction_ramps_from_the_freezing_point(uniform_grid, excess, expected):
    """Full cover at the freezing point, none `ICE_FREE_SST_EXCESS` above it."""
    model = SlabSeaiceModel(uniform_grid)
    stepped = _step_with_sst(
        model, constants.seawater_freezing_point_K + excess)

    np.testing.assert_allclose(
        np.asarray(stepped["state"].ice_fraction), expected, atol=1e-4
    )


def test_ice_fraction_follows_the_sst_cell_by_cell(uniform_grid):
    """Each cell is diagnosed from its own SST, colder meaning more ice."""
    model = SlabSeaiceModel(uniform_grid)
    excess = jnp.linspace(
        -1.0, ICE_FREE_SST_EXCESS + 1.0, uniform_grid.fractional_mask.size
    ).reshape(uniform_grid.shape)
    stepped = _step_with_sst(
        model, constants.seawater_freezing_point_K + excess)

    fraction = np.asarray(stepped["state"].ice_fraction)
    np.testing.assert_allclose(
        fraction,
        np.clip(1.0 - np.asarray(excess) / ICE_FREE_SST_EXCESS, 0.0, 1.0),
        atol=1e-4,
    )
    assert fraction.min() == 0.0 and fraction.max() == 1.0
    assert bool(np.all(np.diff(fraction.ravel()) <= 0.0))


def test_land_cells_carry_no_ice(half_land_grid):
    """A land cell has no ice however cold the SST it is handed."""
    model = SlabSeaiceModel(half_land_grid)
    stepped = _step_with_sst(
        model, constants.seawater_freezing_point_K - 10.0)

    fraction = np.asarray(stepped["state"].ice_fraction)
    land = np.asarray(half_land_grid.binary_mask == 1.0)
    np.testing.assert_allclose(fraction[land], 0.0)
    np.testing.assert_allclose(fraction[~land], 1.0)


def test_the_ice_has_no_memory(uniform_grid):
    """The fraction is a function of this step's SST alone.

    Whatever cover the carry arrives with -- here full cover -- one step
    under warm water leaves none: nothing is integrated from step to step.
    """
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    carry["state"] = carry["state"].replace(
        ice_fraction=jnp.ones(uniform_grid.shape))
    carry["forcing"] = carry["forcing"].replace(
        sea_surface_temperature=jnp.full(
            uniform_grid.shape,
            constants.seawater_freezing_point_K + 2.0 * ICE_FREE_SST_EXCESS,
        )
    )
    stepped, _ = model.step(carry, coupling_time(0))

    np.testing.assert_allclose(np.asarray(stepped["state"].ice_fraction), 0.0)


def test_params_default_equivalence(uniform_grid):
    """Constructing with no params is constructing with the defaults."""
    implicit = SlabSeaiceModel(uniform_grid).initialize()
    explicit = SlabSeaiceModel(
        uniform_grid, SlabSeaiceParameters.default()
    ).initialize()

    assert jax.tree_util.tree_structure(implicit) == jax.tree_util.tree_structure(
        explicit
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_state_has_no_clock(uniform_grid):
    """State carries no time, and the step does not depend on the date.

    The ice fraction is diagnosed from the SST the model is handed, so unlike
    the ocean and land models it reads nothing from the clock at all.
    """
    model = SlabSeaiceModel(uniform_grid)
    assert "sim_time" not in model.initialize()["state"].asdict()

    sst = constants.seawater_freezing_point_K + 0.5 * ICE_FREE_SST_EXCESS
    first = _step_with_sst(model, sst, step=0)
    later = _step_with_sst(model, sst, step=182)

    np.testing.assert_array_equal(
        np.asarray(first["state"].ice_fraction),
        np.asarray(later["state"].ice_fraction),
    )


def test_step_shapes_and_dtypes_stable(uniform_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_two_steps_scan_with_a_stable_carry(uniform_grid):
    """`lax.scan` accepts the step: the carry keeps its structure and dtypes."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        sea_surface_temperature=jnp.full(
            uniform_grid.shape, constants.seawater_freezing_point_K + 0.9
        )
    )

    def body(carry, step):
        new_carry, diagnostics = model.step(carry, coupling_time(0))
        del step
        return new_carry, diagnostics["state"].ice_fraction

    final, fractions = jax.lax.scan(body, carry, jnp.arange(2))

    assert tree_signature(final) == tree_signature(carry)
    assert fractions.shape == (2,) + uniform_grid.shape
    np.testing.assert_allclose(np.asarray(fractions), 0.5, atol=1e-4)


def test_step_is_differentiable_in_the_sst(uniform_grid):
    """d(fraction)/d(SST) is the ramp's slope on it and zero off it."""
    model = SlabSeaiceModel(uniform_grid)

    def mean_fraction(sst):
        carry = model.initialize()
        carry["forcing"] = carry["forcing"].replace(
            sea_surface_temperature=jnp.full(uniform_grid.shape, sst)
        )
        stepped, _ = model.step(carry, coupling_time(0))
        return jnp.mean(stepped["state"].ice_fraction)

    freezing = constants.seawater_freezing_point_K
    on_ramp = jax.grad(mean_fraction)(freezing + 0.5 * ICE_FREE_SST_EXCESS)
    np.testing.assert_allclose(
        float(on_ramp), -1.0 / ICE_FREE_SST_EXCESS, rtol=1e-4)
    # Saturated either way, the cover no longer responds to the SST -- and
    # the gradient there is exactly zero, not NaN.
    assert float(jax.grad(mean_fraction)(freezing - 3.0)) == 0.0
    assert float(jax.grad(mean_fraction)(freezing + 3.0 * ICE_FREE_SST_EXCESS)) == 0.0


def test_output_variables_and_roles(uniform_grid):
    """The dataset holds the fraction as state and the SST as a forcing."""
    import jax_datetime as jdt

    from jem.base.component import TimeAxis

    model = _bind(SlabSeaiceModel(uniform_grid), start="2000-01-01")
    _, diagnostics = model.step(
        _step_with_sst(model, constants.seawater_freezing_point_K + 0.9),
        coupling_time(0),
    )
    stacked = jax.tree.map(lambda leaf: leaf[None], diagnostics)
    dataset = model.to_xarray(stacked, TimeAxis(
        start_date=jdt.to_datetime("2000-01-01"),
        steps=np.array([0]),
        dt=jdt.to_timedelta(1, "day"),
    ))

    assert set(dataset.data_vars) == {
        "ice_fraction", "forcing_sea_surface_temperature"}
    assert dataset["ice_fraction"].attrs["jem_role"] == "state"
    assert dataset["forcing_sea_surface_temperature"].attrs["jem_role"] == "forcing"
    np.testing.assert_allclose(dataset["ice_fraction"].values, 0.5, atol=1e-4)


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------


MONTHLY_ICE_FRACTION = 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(12) / 12.0)


@pytest.fixture
def ice_clim_file(tmp_path, uniform_grid):
    """Write a 12-month `icec` concentration climatology the tests own."""
    from tests.unit.slab_test_utils import write_climatology

    shape = uniform_grid.shape[::-1]  # (lat, lon), the file's order
    seasonal = [np.full(shape, value, dtype=np.float32)
                for value in MONTHLY_ICE_FRACTION]
    return write_climatology(tmp_path / "icec.nc", "icec", np.array(seasonal))


def _bind(model, start="2000-04-01"):
    """Bind a model to a one-day coupler starting on ``start``."""
    import jax_datetime as jdt

    model.bind(
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=jdt.to_datetime(start),
    )
    return model


def test_initial_fraction_reproduces_the_climatology_at_the_start_date(
    uniform_grid, ice_clim_file
):
    """The fraction `initialize` returns is the file's, at the run's month.

    The climatology is read here the way the slab ocean reads its SST --
    `load_monthly_climatology` then `evaluate_cyclic_linear` -- so the file's
    own month-to-month interpolation is what is compared against.
    """
    from jem.components.slab.base import load_monthly_climatology
    from jem.utils.cycles import evaluate_cyclic_linear

    model = _bind(SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file))

    fraction = np.asarray(model.initialize()["state"].ice_fraction)
    expected = np.asarray(evaluate_cyclic_linear(
        model.start_year_fraction,
        load_monthly_climatology(ice_clim_file, "icec", uniform_grid),
    ))
    assert 0.0 < expected.max() < 0.95, expected.max()
    np.testing.assert_allclose(fraction, expected, atol=1e-6)

    # And it is the START date that selected it, not simply the first month:
    # a January start of the same file gives a different (near-full) cover.
    january = _bind(
        SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file),
        start="2000-01-01",
    ).initialize()["state"].ice_fraction
    assert float(np.asarray(january).min()) > fraction.max()


def test_without_a_climatology_the_run_starts_ice_free(uniform_grid):
    """No file, no ice: the first step's SST is what puts any there."""
    carry = _bind(SlabSeaiceModel(uniform_grid)).initialize()

    assert np.asarray(carry["state"].ice_fraction).shape == uniform_grid.shape
    np.testing.assert_array_equal(np.asarray(carry["state"].ice_fraction), 0.0)


def test_the_first_step_replaces_the_climatological_cover(
    uniform_grid, ice_clim_file
):
    """The climatology is an initial condition only: one step and it is gone."""
    model = _bind(
        SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file),
        start="2000-01-01",
    )
    assert float(np.asarray(model.initialize()["state"].ice_fraction).min()) > 0.9

    stepped = _step_with_sst(
        model, constants.seawater_freezing_point_K + 2.0 * ICE_FREE_SST_EXCESS)
    np.testing.assert_allclose(np.asarray(stepped["state"].ice_fraction), 0.0)


def test_land_cells_carry_no_climatological_ice(half_land_grid, ice_clim_file):
    """The ocean mask still wins: a land cell has no ice whatever the file says."""
    model = _bind(
        SlabSeaiceModel(half_land_grid, ice_clim_file=ice_clim_file),
        start="2000-01-01",
    )
    fraction = np.asarray(model.initialize()["state"].ice_fraction)

    land = np.asarray(half_land_grid.binary_mask == 1.0)
    assert fraction[land].max() == 0.0
    assert fraction[~land].min() > 0.9


def test_a_missing_climatology_file_is_reported_at_construction(uniform_grid):
    """A bad path fails where the traceback still points at the caller."""
    with pytest.raises(FileNotFoundError, match="no-such-file.nc"):
        SlabSeaiceModel(uniform_grid, ice_clim_file="no-such-file.nc")


def test_a_climatology_with_nans_over_ocean_is_refused(tmp_path, uniform_grid):
    """NaNs over ocean mean the file's land mask and the grid's disagree."""
    from tests.unit.slab_test_utils import write_climatology

    shape = (12,) + uniform_grid.shape[::-1]
    values = np.zeros(shape, dtype=np.float32)
    values[0, 0, 0] = np.nan
    path = write_climatology(tmp_path / "nan.nc", "icec", values)

    with pytest.raises(ValueError, match="NaNs over ocean"):
        SlabSeaiceModel(uniform_grid, ice_clim_file=path)


def test_a_nan_land_fill_value_stays_out_of_the_initial_state(
    half_land_grid, tmp_path
):
    """A file's NaN land fill value must not reach the carry.

    `__init__` accepts a NaN over land -- only a NaN over *ocean* is refused,
    since a real file's land cells routinely carry one -- so `initialize` has
    to select the land cells out, or the first exchange would hand the
    atmosphere a NaN ice cover over every continent.
    """
    from tests.unit.slab_test_utils import write_climatology

    # `half_land_grid`'s land half is the eastern two of its four longitudes
    # (see the fixture); write the file in its own (time, lat, lon) order and
    # NaN exactly those columns, leaving the ocean half a plain 0.5 everywhere.
    land_lat_lon = np.asarray(half_land_grid.binary_mask == 1.0).T
    values = np.full((12,) + half_land_grid.shape[::-1], 0.5, dtype=np.float32)
    values[:, land_lat_lon] = np.nan
    path = write_climatology(tmp_path / "icec_nan_land.nc", "icec", values)

    # Construction succeeds: the NaNs are over land, not ocean.
    model = _bind(SlabSeaiceModel(half_land_grid, ice_clim_file=path))
    fraction = np.asarray(model.initialize()["state"].ice_fraction)

    ocean = np.asarray(half_land_grid.binary_mask == 0.0)
    assert bool(np.all(np.isfinite(fraction)))
    np.testing.assert_array_equal(fraction[~ocean], 0.0)
    np.testing.assert_allclose(fraction[ocean], 0.5, atol=1e-6)
