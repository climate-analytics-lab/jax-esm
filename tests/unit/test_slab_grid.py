"""Tests for `jem.components.slab.grid` and the slab models' xarray output."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import jem.data
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jcm.utils import data_to_xarray
from jem.components.slab import grid as grid_module
from jem.components.slab.grid import SlabGrid, to_degrees
from jem.components.slab.slab_ocean_model import SlabOceanModel
from tests.unit.slab_test_utils import (
    make_grid,
    run_steps,
    time_axis,
)

DATA = Path(jem.data.__file__).parent
T31_SCRIP = DATA / "JCM_T31.SCRIP.nc"
T31_TERRAIN = DATA / "terrain_JCM_T31.nc"
DISPLACED_POLE_SCRIP = DATA / "DisplacedPoleGrid.SCRIP.nc"

#: Nanoseconds in a day, as JCM computes the factor.
NANOSECONDS_PER_DAY = np.timedelta64(1, "D") / np.timedelta64(1, "ns")


@pytest.fixture(scope="module")
def t31_coords():
    """Return the T31 SPEEDY coordinate system a coupled run would use."""
    return get_speedy_coords(layers=8, spectral_truncation=31)


def test_from_coords_matches_scrip_t31(t31_coords):
    """The grid built from the atmosphere's coords is the packaged T31 grid.

    This is what makes dropping the ``"JCM::T31"`` specification DSL safe: the
    two independent descriptions of the same grid -- the spectral coordinate
    system JCM runs on, and the SCRIP file shipped for the regridding weights
    -- agree on every cell centre.
    """
    from_coords = SlabGrid.from_coords(t31_coords.horizontal)
    from_scrip = SlabGrid.from_scrip(str(T31_SCRIP))

    assert from_coords.shape == from_scrip.shape == (96, 48)
    np.testing.assert_allclose(
        np.asarray(from_coords.latitude_radian, dtype=np.float64),
        np.asarray(from_scrip.latitude_radian, dtype=np.float64),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(from_coords.longitude_radian, dtype=np.float64),
        np.asarray(from_scrip.longitude_radian, dtype=np.float64),
        atol=1e-6,
    )


def test_from_coords_matches_the_coordinates_jcm_writes(t31_coords):
    """The 1-D degree axes are bit-for-bit the ones JCM puts in its output."""
    grid = SlabGrid.from_coords(t31_coords.horizontal)
    reference = data_to_xarray(
        {"surface_pressure": np.zeros((96, 48), dtype=np.float32)},
        coords=t31_coords,
        times=None,
        serialize_coords_to_attrs=False,
    )

    np.testing.assert_array_equal(
        to_degrees(grid.longitude_axis_radian), reference["lon"].values
    )
    np.testing.assert_array_equal(
        to_degrees(grid.latitude_axis_radian), reference["lat"].values
    )


def test_from_coords_takes_the_mask_from_terrain_data(t31_coords):
    """``TerrainData.fmask`` is the mask source, and needs no reorientation.

    ``fmask`` is ``(n_lon, n_lat)`` with latitude ascending south to north,
    which is exactly SlabGrid's layout: Antarctica lands on the southernmost
    row and the Arctic ocean on the northernmost. A transposed or flipped mask
    would fail this.
    """
    fmask = TerrainData.from_file(str(T31_TERRAIN), t31_coords).fmask
    grid = SlabGrid.from_coords(t31_coords.horizontal, fractional_mask=fmask)

    latitudes = to_degrees(grid.latitude_axis_radian)
    assert latitudes[0] < -85.0 and latitudes[-1] > 85.0

    land_fraction = np.asarray(grid.fractional_mask)
    assert land_fraction[:, 0].mean() == pytest.approx(1.0)   # Antarctica
    assert land_fraction[:, -1].mean() == pytest.approx(0.0)  # Arctic ocean

    # binary_mask == 1 is land (jem CLAUDE.md).
    assert float(np.asarray(grid.binary_mask)[:, 0].mean()) == pytest.approx(1.0)
    assert float(np.asarray(grid.binary_mask)[:, -1].mean()) == pytest.approx(0.0)


def test_from_coords_defaults_to_all_ocean(t31_coords):
    """A grid built with no mask is all ocean, not all land."""
    grid = SlabGrid.from_coords(t31_coords.horizontal)
    assert float(jnp.sum(grid.binary_mask)) == 0.0


def test_from_coords_rejects_a_mask_that_is_not_on_the_grid(t31_coords):
    """A transposed mask is the likely mistake, so it is named in the error."""
    with pytest.raises(ValueError, match="n_lon, n_lat"):
        SlabGrid.from_coords(t31_coords.horizontal, fractional_mask=jnp.zeros((48, 96)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SlabGrid.from_coords(t31_coords.horizontal, fractional_mask=jnp.full((96, 48), 2.0))


def test_scrip_curvilinear_grid_is_not_separable():
    """A displaced-pole grid has no 1-D axes, and says so in its dims."""
    grid = SlabGrid.from_scrip(str(DISPLACED_POLE_SCRIP))

    assert not grid.is_separable
    assert grid.longitude_axis_radian is None
    assert grid.dims == ("x", "y")


def test_scrip_regular_grid_is_separable():
    """A SCRIP file that happens to hold a regular grid gets 1-D axes."""
    grid = SlabGrid.from_scrip(str(T31_SCRIP))

    assert grid.is_separable
    assert grid.dims == ("lon", "lat")
    assert grid.longitude_axis_radian.shape == (96,)


def test_grid_specification_dsl_is_gone():
    """The ``"JCM::T31"`` string DSL and its loaders were removed in v1.0."""
    for removed in (
        "generate_slab_grid",
        "generate_slab_grid_from_scrip",
        "load_jcm_fractional_mask",
        "_parse_grid_specification",
        "_generate_jcm_slab_grid",
    ):
        assert not hasattr(grid_module, removed)


def test_mismatched_shapes_are_rejected():
    """The three grid fields must describe one grid."""
    with pytest.raises(ValueError, match="latitude_radian"):
        SlabGrid(
            fractional_mask=jnp.zeros((4, 3)),
            latitude_radian=jnp.zeros((3, 4)),
            longitude_radian=jnp.zeros((4, 3)),
        )


def _jcm_like_dataset(coords, start_days, n_records):
    """Build a dataset with JCM's own coordinate construction, for merging.

    The values come from ``jcm.utils.data_to_xarray`` and the time axis from
    the same expression ``jcm.predictions.ModelPredictions`` uses, so this is
    the coordinate layout a real JCM run writes. The integration pass merges
    against genuine JCM output; this keeps the contract under test here.
    """
    times = start_days + 1.0 * (np.arange(n_records) + 1)
    dataset = data_to_xarray(
        {
            "surface_pressure": np.zeros(
                (n_records,) + coords.horizontal.nodal_shape, dtype=np.float32
            )
        },
        coords=coords,
        times=times - times[0],
        serialize_coords_to_attrs=False,
    )
    dataset["time"] = (times * NANOSECONDS_PER_DAY).astype("datetime64[ns]")
    return dataset


def test_to_xarray_dims_and_merge(t31_coords):
    """Slab output carries JCM's dims, coords and time axis, and merges with it."""
    grid = SlabGrid.from_coords(t31_coords.horizontal)
    model = SlabOceanModel(grid)
    _, diagnostics = run_steps(model, model.initialize(), 3)

    dataset = model.to_xarray(diagnostics, time_axis(3))

    assert dataset["sea_surface_temperature"].dims == ("time", "lon", "lat")
    assert dataset["lon"].dims == ("lon",) and dataset["lat"].dims == ("lat",)
    assert dataset["lon"].attrs["units"] == "degrees_east"
    assert dataset["time"].dtype == np.dtype("datetime64[ns]")
    # Record k covers step k and is stamped at its END, as JCM stamps its own
    # saved frames.
    assert str(dataset["time"].values[0]) == "2001-01-02T00:00:00.000000000"

    # 2001-01-01 is 11323 days after the epoch; JCM's own axis starts there.
    merged = xr.merge([dataset, _jcm_like_dataset(t31_coords, 11323.0, 3)], join="exact")
    assert dict(merged.sizes) == {"time": 3, "lon": 96, "lat": 48}
    assert "surface_pressure" in merged and "sea_surface_temperature" in merged


def test_to_xarray_curvilinear_uses_auxiliary_coordinates():
    """A curvilinear grid writes CF auxiliary coordinates instead of 1-D axes."""
    grid = SlabGrid.from_scrip(str(DISPLACED_POLE_SCRIP))
    model = SlabOceanModel(grid)
    _, diagnostics = run_steps(model, model.initialize(), 2)

    dataset = model.to_xarray(diagnostics, time_axis(2))

    assert dataset["sea_surface_temperature"].dims == ("time", "x", "y")
    assert dataset["lat"].dims == ("x", "y")
    assert dataset["sea_surface_temperature"].attrs["coordinates"] == "lat lon"
    # CF reserves ``axis`` for true coordinate variables.
    assert "axis" not in dataset["lat"].attrs


def test_to_xarray_rejects_a_mismatched_time_axis():
    """Stacked diagnostics and the time axis must describe the same records."""
    model = SlabOceanModel(make_grid())
    _, diagnostics = run_steps(model, model.initialize(), 3)

    with pytest.raises(ValueError, match="time records"):
        model.to_xarray(diagnostics, time_axis(2))


@pytest.mark.parametrize(
    "bad_value", [-1.0, 50.0, np.nan], ids=["fill_value", "percentage", "nan"]
)
def test_explicit_scrip_mask_is_validated(bad_value):
    """A mask given to ``from_scrip`` is checked like every other mask."""
    good = SlabGrid.from_scrip(str(T31_SCRIP))
    mask = np.zeros(good.shape)
    mask[0, 0] = bad_value
    with pytest.raises(ValueError, match="fractional_mask"):
        SlabGrid.from_scrip(str(T31_SCRIP), fractional_mask=mask)
    with pytest.raises(ValueError, match="fractional_mask"):
        SlabGrid(
            fractional_mask=jnp.asarray(mask),
            latitude_radian=good.latitude_radian,
            longitude_radian=good.longitude_radian,
        )
