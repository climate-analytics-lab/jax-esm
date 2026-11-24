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
from jem.components.slab.slab_seaice_model import SlabSeaiceModel
from tests.unit.slab_test_utils import (
    make_grid,
    run_steps,
    time_axis,
)

DATA = Path(jem.data.__file__).parent
T31_SCRIP = DATA / "JCM_T31.SCRIP.nc"
T31_TERRAIN = DATA / "terrain_JCM_T31.nc"
DISPLACED_POLE_SCRIP = DATA / "DisplacedPoleGrid.SCRIP.nc"

#: Milliseconds in a day, as JCM's own output-label arithmetic uses it.
MILLISECONDS_PER_DAY = 86_400_000


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

    assert bool(np.asarray(grid.land_mask)[:, 0].all())
    assert not bool(np.asarray(grid.ocean_mask)[:, 0].any())
    assert bool(np.asarray(grid.ocean_mask)[:, -1].all())
    assert not bool(np.asarray(grid.land_mask)[:, -1].any())


def test_from_coords_defaults_to_all_ocean(t31_coords):
    """A grid built with no mask is all ocean, not all land."""
    grid = SlabGrid.from_coords(t31_coords.horizontal)
    assert not bool(jnp.any(grid.land_mask))
    assert bool(jnp.all(grid.ocean_mask))


def test_the_land_and_ocean_masks_overlap_on_coast_cells():
    """Any land puts a cell in the land mask, any ocean in the ocean mask:
    a partly-land cell is in both, and every cell is in at least one.
    """
    fraction = np.array([[0.0, 0.3, 1.0]] * 4)
    grid = make_grid(fractional_mask=fraction)
    land = np.asarray(grid.land_mask)
    ocean = np.asarray(grid.ocean_mask)

    assert land.dtype == bool and ocean.dtype == bool
    np.testing.assert_array_equal(land, fraction > 0.0)
    np.testing.assert_array_equal(ocean, fraction < 1.0)
    np.testing.assert_array_equal(land & ocean, (fraction > 0.0) & (fraction < 1.0))
    assert bool((land | ocean).all())


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

    The values come from ``jcm.utils.data_to_xarray``, and the time axis is
    JCM's own labelling -- each record's interval midpoint, in whole
    milliseconds (``bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) // 2``,
    ``jcm.predictions.ModelPredictions.time_labels``) -- so this is the
    coordinate layout a real JCM run writes. The integration pass merges
    against genuine JCM output; this keeps the contract under test here.
    """
    dataset = data_to_xarray(
        {
            "surface_pressure": np.zeros(
                (n_records,) + coords.horizontal.nodal_shape, dtype=np.float32
            )
        },
        coords=coords,
        times=1.0 * np.arange(n_records),
        serialize_coords_to_attrs=False,
    )
    start_ms = round(start_days * MILLISECONDS_PER_DAY)
    lower = start_ms + np.arange(n_records, dtype=np.int64) * MILLISECONDS_PER_DAY
    midpoints = lower + MILLISECONDS_PER_DAY // 2
    dataset["time"] = midpoints.astype("datetime64[ms]")
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
    assert dataset["time"].dtype == np.dtype("datetime64[ms]")
    # Record k covers [start + k*dt, start + (k+1)*dt) and is stamped at its
    # MIDPOINT, JCM's convention (jcm.predictions.output_time_labels).
    assert str(dataset["time"].values[0]) == "2001-01-01T12:00:00.000"

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
    assert dataset["latitude"].dims == dataset["longitude"].dims == ("x", "y")
    # Named differently from the atmosphere's 1-D ``lon``/``lat`` dimension
    # coordinates, so a merge cannot lose them.
    assert "lon" not in dataset.variables and "lat" not in dataset.variables
    assert (
        dataset["sea_surface_temperature"].attrs["coordinates"]
        == "longitude latitude"
    )
    assert dataset["longitude"].attrs["standard_name"] == "longitude"
    assert dataset["latitude"].attrs["units"] == "degrees_north"
    # CF reserves ``axis`` for true coordinate variables.
    assert "axis" not in dataset["latitude"].attrs
    np.testing.assert_array_equal(
        dataset["latitude"].values, to_degrees(grid.latitude_radian))


def _displaced_pole_datasets(n_records=2):
    """Return the displaced-pole ocean's and sea ice's datasets, built by ``to_xarray``."""
    grid = SlabGrid.from_scrip(str(DISPLACED_POLE_SCRIP))
    datasets = {}
    for model in (SlabOceanModel(grid), SlabSeaiceModel(grid, name="seaice")):
        _, diagnostics = run_steps(model, model.initialize(), n_records)
        datasets[model.name] = model.to_xarray(diagnostics, time_axis(n_records))
    return datasets


def test_a_merged_curvilinear_run_keeps_every_variables_2d_positions(t31_coords):
    """``xr.merge`` of atmosphere, ocean and sea ice keeps the 2-D positions.

    Named ``lon``/``lat`` the curvilinear components' coordinates collided with
    the atmosphere's 1-D dimension coordinates of the same names: the merge
    "succeeded", kept the atmosphere's and silently dropped the others', so
    ``sea_surface_temperature`` came back with ``time`` as its only coordinate.
    """
    datasets = _displaced_pole_datasets()
    # 2001-01-01 is 11323 days after the epoch (see the time-axis test above).
    atmosphere = _jcm_like_dataset(t31_coords, 11323.0, 2)
    for model_name, dataset in datasets.items():
        dataset["time"] = atmosphere["time"]
        assert "longitude" in dataset.coords, model_name

    merged = xr.merge([atmosphere, *datasets.values()], join="exact", compat="no_conflicts")

    # The atmosphere's axes are the 1-D ``lon``/``lat``, untouched.
    assert merged["surface_pressure"].dims == ("time", "lon", "lat")
    np.testing.assert_array_equal(merged["lon"].values, atmosphere["lon"].values)
    for name in ("sea_surface_temperature", "ice_fraction"):
        variable = merged[name]
        assert variable.dims == ("time", "x", "y"), name
        assert {"longitude", "latitude"} <= set(variable.coords), name
        assert variable["longitude"].dims == ("x", "y"), name
        np.testing.assert_array_equal(
            variable["latitude"].values,
            to_degrees(SlabGrid.from_scrip(str(DISPLACED_POLE_SCRIP)).latitude_radian),
        )
        assert variable.attrs["coordinates"] == "longitude latitude"


def test_two_components_on_one_curvilinear_grid_merge_exactly():
    """The ocean and the sea ice share ``x``/``y`` and equal 2-D coordinates."""
    datasets = _displaced_pole_datasets()

    merged = xr.merge(datasets.values(), join="exact", compat="no_conflicts")

    assert merged.sizes["x"] == datasets["ocn"].sizes["x"]
    np.testing.assert_array_equal(
        merged["longitude"].values, datasets["ocn"]["longitude"].values)
    assert {"sea_surface_temperature", "ice_fraction"} <= set(merged.data_vars)


def test_curvilinear_components_on_grids_of_different_shapes_do_not_merge():
    """The convention's limit: ``x``/``y`` carry no values, only a size.

    Two curvilinear grids of different shapes cannot share index dimensions, and
    xarray refuses the merge with an error naming the conflicting sizes rather
    than producing a union. (No shipped configuration puts two components on
    different curvilinear grids; see ``docs/source/design/output.md``.)
    """
    ocean = _displaced_pole_datasets()["ocn"]
    longitude = np.deg2rad(
        np.array([0.0, 90.0, 180.0, 270.0])[:, None] + np.arange(3)[None, :])
    latitude = np.deg2rad(
        np.array([-60.0, 0.0, 60.0])[None, :] + np.arange(4)[:, None])
    small = SlabGrid(
        fractional_mask=jnp.zeros((4, 3)),
        latitude_radian=jnp.asarray(latitude),
        longitude_radian=jnp.asarray(longitude),
    )
    assert not small.is_separable
    model = SlabSeaiceModel(small, name="small_ice")
    _, diagnostics = run_steps(model, model.initialize(), 2)
    other = model.to_xarray(diagnostics, time_axis(2))

    with pytest.raises(ValueError, match="size"):
        xr.merge([ocean, other], join="exact", compat="no_conflicts")


def test_to_xarray_rejects_a_mismatched_time_axis():
    """Stacked diagnostics and the time axis must describe the same records."""
    model = SlabOceanModel(make_grid())
    _, diagnostics = run_steps(model, model.initialize(), 3)

    with pytest.raises(ValueError, match="time records"):
        model.to_xarray(diagnostics, time_axis(2))


@pytest.mark.parametrize("variable", ["grid_center_lat", "grid_center_lon"])
def test_scrip_non_finite_centre_coordinates_are_rejected(tmp_path, variable):
    """A NaN centre coordinate passes the range checks, so it is caught first.

    ``NaN < -90`` and ``NaN > 90`` are both False, so without this the cell
    would be accepted and the NaN would surface much later, in the coordinate
    axis of the run's output.
    """
    ds = xr.open_dataset(T31_SCRIP)
    values = np.asarray(ds[variable].values, dtype=np.float64)
    values[0] = np.nan
    ds[variable] = (ds[variable].dims, values, ds[variable].attrs)
    broken = tmp_path / "broken.SCRIP.nc"
    ds.to_netcdf(broken)

    with pytest.raises(ValueError, match="non-finite"):
        SlabGrid.from_scrip(str(broken))


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
