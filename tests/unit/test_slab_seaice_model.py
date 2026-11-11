"""Tests for `jem.components.slab.slab_seaice_model`."""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest

from jem import constants
from jem.components.slab.base import load_monthly_climatology
from jem.components.slab.slab_seaice_model import SlabSeaiceModel, SlabSeaiceParameters
from jem.components.slab.slab_seaice_model.slab_seaice_model import ICE_FREE_SST_EXCESS
from jem.utils.cycles import evaluate_cyclic_linear
from tests.unit.slab_test_utils import (
    LONGITUDE_DEGREES,
    coupling_time,
    make_grid,
    tree_signature,
    write_climatology,
)

#: A seasonal ice cover: saturated in January, open in July.
MONTHLY_ICE_FRACTION = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


@pytest.fixture
def uniform_grid():
    """Return a tiny all-ocean 4x3 lon-lat grid the tests own."""
    return make_grid()


@pytest.fixture
def half_land_grid():
    """Return a 4x3 grid whose eastern half is land."""
    shape = (len(LONGITUDE_DEGREES), 3)
    land_fraction = jnp.where(
        jnp.arange(shape[0])[:, None] >= shape[0] // 2,
        jnp.ones(shape),
        jnp.zeros(shape),
    )
    return make_grid(land_fraction=land_fraction)


def _step_with_sst(model, sst, step=0):
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(sea_surface_temperature=sst)
    return model.step(carry, coupling_time(step))[0]


def test_ice_fraction_is_the_sst_ramp(uniform_grid):
    """Full cover at freezing, none ICE_FREE_SST_EXCESS above it, linear between."""
    freezing = constants.seawater_freezing_point_K
    offsets = np.array([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])[: uniform_grid.shape[0]]
    sst = jnp.broadcast_to(
        jnp.asarray(freezing + offsets * ICE_FREE_SST_EXCESS)[:, None], uniform_grid.shape
    )

    fraction = np.asarray(_step_with_sst(SlabSeaiceModel(uniform_grid), sst)["state"].ice_fraction)

    expected = np.clip(1.0 - offsets, 0.0, 1.0)[:, None] * np.ones(uniform_grid.shape)
    np.testing.assert_allclose(fraction, expected, atol=1e-5)


def test_land_cells_carry_no_ice(half_land_grid):
    """A cell with no ocean has no ice, however cold its SST forcing."""
    sst = jnp.full(half_land_grid.shape, constants.seawater_freezing_point_K - 5.0)
    fraction = np.asarray(_step_with_sst(SlabSeaiceModel(half_land_grid), sst)["state"].ice_fraction)

    no_ocean = np.asarray(half_land_grid.binary_ocean_mask) == 0
    assert fraction[no_ocean].max() == 0.0
    assert fraction[~no_ocean].min() == 1.0


def test_step_does_not_depend_on_the_date_or_the_previous_state(uniform_grid):
    """The fraction is a diagnostic of the current SST alone."""
    model = SlabSeaiceModel(uniform_grid)
    sst = jnp.full(uniform_grid.shape, constants.seawater_freezing_point_K + 0.5)
    carry = model.initialize()
    assert "sim_time" not in carry["state"].asdict()

    carry["forcing"] = carry["forcing"].replace(sea_surface_temperature=sst)
    first = model.step(carry, coupling_time(0))[0]
    carry["state"] = carry["state"].replace(ice_fraction=jnp.ones(uniform_grid.shape))
    later = model.step(carry, coupling_time(182))[0]

    np.testing.assert_array_equal(
        np.asarray(first["state"].ice_fraction), np.asarray(later["state"].ice_fraction)
    )


def test_step_shapes_and_dtypes_stable(uniform_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_ice_fraction_is_differentiable_in_the_sst(uniform_grid):
    """Inside the ramp the gradient is -1 / ICE_FREE_SST_EXCESS per cell."""
    model = SlabSeaiceModel(uniform_grid)
    sst = jnp.full(uniform_grid.shape, constants.seawater_freezing_point_K + 0.5)

    gradient = jax.grad(
        lambda t: _step_with_sst(model, t)["state"].ice_fraction.sum()
    )(sst)

    np.testing.assert_allclose(np.asarray(gradient), -1.0 / ICE_FREE_SST_EXCESS, rtol=1e-5)


def test_initialize_takes_parameters_and_defaults_to_the_models_own(uniform_grid):
    """The parameters `initialize` is given are the ones in the carry."""
    model = SlabSeaiceModel(uniform_grid)
    assert model.initialize()["params"] == model.params
    passed = SlabSeaiceParameters()
    assert model.initialize(passed)["params"] is passed


@pytest.fixture
def ice_clim_file(tmp_path, uniform_grid):
    """Write a 12-month `icec` concentration climatology the tests own."""
    shape = uniform_grid.shape[::-1]  # (lat, lon), the file's order
    seasonal = [np.full(shape, value, dtype=np.float32) for value in MONTHLY_ICE_FRACTION]
    return write_climatology(tmp_path / "icec.nc", "icec", np.array(seasonal))


def _bind(model, start="2000-04-01"):
    """Bind a model to a one-day coupler starting on ``start``."""
    model.bind(
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=jdt.to_datetime(start),
    )
    return model


def test_initial_fraction_is_the_climatology_at_the_start_date(uniform_grid, ice_clim_file):
    """The fraction `initialize` publishes is the file's, at the run's month."""
    model = _bind(SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file))

    fraction = np.asarray(model.initialize()["state"].ice_fraction)
    expected = np.asarray(evaluate_cyclic_linear(
        model.start_year_fraction,
        load_monthly_climatology(ice_clim_file, "icec", uniform_grid),
    ))
    assert 0.0 < expected.max() < 0.95, expected.max()
    np.testing.assert_allclose(fraction, expected, atol=1e-6)

    # It is the start date that selected it: a January start of the same
    # file gives a fuller cover.
    january = _bind(
        SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file), start="2000-01-01"
    ).initialize()["state"].ice_fraction
    assert float(np.asarray(january).min()) > fraction.max()


def test_without_a_climatology_the_ice_starts_ice_free(uniform_grid):
    """No file means no initial ice."""
    bare = _bind(SlabSeaiceModel(uniform_grid)).initialize()
    assert float(np.asarray(bare["state"].ice_fraction).max()) == 0.0


def test_land_cells_carry_no_climatological_ice(half_land_grid, ice_clim_file):
    """The ocean mask wins: a cell with no ocean has no ice whatever the file says."""
    model = _bind(
        SlabSeaiceModel(half_land_grid, ice_clim_file=ice_clim_file), start="2000-01-01"
    )
    fraction = np.asarray(model.initialize()["state"].ice_fraction)

    no_ocean = np.asarray(half_land_grid.binary_ocean_mask) == 0
    assert fraction[no_ocean].max() == 0.0


def test_a_nan_land_fill_value_is_accepted(half_land_grid, tmp_path):
    """NaN over land is a fill value, and it does not reach the initial state."""
    shape = (12,) + half_land_grid.shape[::-1]
    values = np.full(shape, 0.5, dtype=np.float32)
    no_ocean = np.asarray(half_land_grid.binary_ocean_mask).T == 0
    values[:, no_ocean] = np.nan
    path = write_climatology(tmp_path / "icec_nan_land.nc", "icec", values)

    fraction = _bind(SlabSeaiceModel(half_land_grid, ice_clim_file=path)).initialize()["state"].ice_fraction
    assert np.isfinite(np.asarray(fraction)).all()


def test_a_missing_climatology_file_is_reported_at_construction(uniform_grid):
    """A bad path fails where the traceback still points at the caller."""
    with pytest.raises(FileNotFoundError, match="no-such-file.nc"):
        SlabSeaiceModel(uniform_grid, ice_clim_file="no-such-file.nc")


def test_a_climatology_with_nans_over_ocean_is_refused(tmp_path, uniform_grid):
    """NaNs over ocean mean the file's land mask and the grid's disagree."""
    shape = (12,) + uniform_grid.shape[::-1]
    values = np.zeros(shape, dtype=np.float32)
    values[0, 0, 0] = np.nan
    path = write_climatology(tmp_path / "nan.nc", "icec", values)

    with pytest.raises(ValueError, match="NaNs over ocean"):
        SlabSeaiceModel(uniform_grid, ice_clim_file=path)


def test_output_has_the_fraction_and_the_sst_it_came_from(uniform_grid):
    """`to_xarray` writes the state and the forcing, prefixed and role-tagged."""
    from tests.unit.slab_test_utils import time_axis

    model = SlabSeaiceModel(uniform_grid)
    _, diagnostics = model.step(model.initialize(), coupling_time(0))
    stacked = jax.tree.map(lambda x: x[None], diagnostics)
    dataset = model.to_xarray(stacked, time_axis(1))

    assert dataset["ice_fraction"].attrs["jem_role"] == "state"
    assert dataset["forcing_sea_surface_temperature"].attrs["jem_role"] == "forcing"
