"""Tests for :mod:`jem.tools.ocean_bathymetry`.

None of these needs Veros or the real bathymetry: the source is a synthetic
1-degree lat-lon field laid out like Veros' ``global_1deg`` asset (``xt`` from
90.5E, ``yt`` from 79.5S, ``bathymetry`` in metres negative down).
"""

from importlib import resources

import numpy as np
import pytest
import xarray as xr

from jem.components.veros.setups._layers import LAYER_THICKNESSES
from jem.tools.ocean_bathymetry import (
    MAXIMUM_DEPTH,
    MINIMUM_DEPTH,
    ocean_bathymetry,
    read_depth,
    read_wet_mask,
)

DATA = resources.files("jem.data")
SCRIP_FILE = str(DATA / "RotatedGaussianLatLon.SCRIP.nc")
MASK_FILE = str(DATA / "landsea_mask_fraction_RotatedGaussianLatLon.nc")
NX, NY = 96, 48


def _source_file(path, depth_of):
    """Write a synthetic source: ``depth_of(lat, lon)`` metres, negated, 0 = land."""
    lon = np.arange(90.5, 450.0, 1.0)
    lat = np.arange(-79.5, 80.0, 1.0)
    lat2, lon2 = np.meshgrid(lat, lon % 360, indexing="ij")
    ds = xr.Dataset(
        {"bathymetry": (("yt", "xt"), (-depth_of(lat2, lon2)).astype(np.float32))},
        coords={"xt": lon.astype(np.float32), "yt": lat.astype(np.float32)},
    )
    ds.to_netcdf(path)
    return str(path)


def _generate(tmp_path, depth_of, **kwargs):
    source = _source_file(tmp_path / "source.nc", depth_of)
    out = ocean_bathymetry(source, SCRIP_FILE, MASK_FILE, tmp_path / "out.nc", **kwargs)
    with xr.open_dataset(out) as ds:
        return ds.load()


def test_the_default_maximum_depth_is_the_default_layers_total():
    assert MAXIMUM_DEPTH == sum(LAYER_THICKNESSES)
    assert MINIMUM_DEPTH == sum(LAYER_THICKNESSES[:2])


def test_depth_is_defined_exactly_on_the_wet_columns(tmp_path):
    """The bathymetry never changes which columns are wet: NaN is exactly the mask's land."""
    ds = _generate(tmp_path, lambda lat, lon: 3000.0 + 0 * lat)
    wet = read_wet_mask(MASK_FILE, 0.5, NY, NX)
    np.testing.assert_array_equal(~np.isnan(ds["depth"].to_numpy()), wet)
    assert ds.attrs["wet_cells"] == wet.sum()


def test_a_uniform_ocean_gives_the_same_depth_wherever_a_source_cell_exists(tmp_path):
    ds = _generate(tmp_path, lambda lat, lon: 3000.0 + 0 * lat)
    depth = ds["depth"].to_numpy()
    lat = ds["lat"].to_numpy()
    covered = np.abs(lat) < 75.0
    wet = ~np.isnan(depth)
    np.testing.assert_allclose(depth[wet & covered], 3000.0)
    # Cells with no source (beyond 79.5 degrees) are filled, not left NaN,
    # and they are counted.
    assert ds.attrs["cells_filled_from_neighbours"] > 0
    np.testing.assert_allclose(depth[wet], 3000.0)


def test_depth_follows_true_longitude_and_latitude_not_grid_indices(tmp_path):
    """A field that varies with true position comes out with the same pattern.

    The grid's pole is rotated, so a depth that is deep in one hemisphere and
    shallow in the other, and likewise east and west, can only come out right
    if the source cells are matched to the cells' true centres.
    """
    ds = _generate(tmp_path, lambda lat, lon: 1000.0 + 1500.0 * (lat > 0) + 1500.0 * (lon < 180))
    depth = ds["depth"].to_numpy()
    lat, lon = ds["lat"].to_numpy(), ds["lon"].to_numpy() % 360
    wet = ~np.isnan(depth)
    interior = wet & (np.abs(lat) > 10) & (np.abs(lat) < 70) & (np.abs(lon - 180) > 10) & (lon > 10) & (lon < 350)
    expected = 1000.0 + 1500.0 * (lat > 0) + 1500.0 * (lon < 180)
    np.testing.assert_allclose(depth[interior], expected[interior])


def test_depths_are_clipped_to_the_model_bottom_and_a_two_layer_minimum(tmp_path):
    deep = _generate(tmp_path, lambda lat, lon: 9000.0 + 0 * lat)
    assert np.nanmax(deep["depth"].to_numpy()) == MAXIMUM_DEPTH
    assert deep.attrs["cells_clipped_to_maximum_depth"] == deep.attrs["wet_cells"]
    shallow = _generate(tmp_path, lambda lat, lon: 10.0 + 0 * lat)
    assert np.nanmin(shallow["depth"].to_numpy()) == MINIMUM_DEPTH
    assert shallow.attrs["cells_raised_to_minimum_depth"] == shallow.attrs["wet_cells"]


def test_land_source_cells_do_not_dilute_a_coastal_cell(tmp_path):
    """Only ocean source cells are averaged: half land source gives the ocean half's depth."""
    ds = _generate(tmp_path, lambda lat, lon: np.where(lon < 180, 0.0, 2000.0))
    depth = ds["depth"].to_numpy()
    lat, lon = ds["lat"].to_numpy(), ds["lon"].to_numpy() % 360
    near_edge = ~np.isnan(depth) & (np.abs(lon - 180) < 4) & (np.abs(lat) < 60)
    assert near_edge.any()
    np.testing.assert_allclose(depth[near_edge], 2000.0)


def test_the_file_reads_back_in_the_setups_layout(tmp_path):
    ds = _generate(tmp_path, lambda lat, lon: 2500.0 + 0 * lat)
    depth = read_depth(tmp_path / "out.nc", NY, NX)
    assert depth.shape == (NX, NY)
    np.testing.assert_array_equal(depth.T, ds["depth"].to_numpy())
    with pytest.raises(ValueError, match="grid is"):
        read_depth(tmp_path / "out.nc", NY, NX + 1)


def test_a_source_with_the_wrong_dimensions_is_refused(tmp_path):
    source = tmp_path / "bad.nc"
    xr.Dataset({"bathymetry": (("a", "b"), np.ones((3, 3)))}).to_netcdf(source)
    with pytest.raises(ValueError, match="dims"):
        ocean_bathymetry(source, SCRIP_FILE, MASK_FILE, tmp_path / "out.nc")
