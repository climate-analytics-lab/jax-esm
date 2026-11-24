"""Tests for :mod:`jem.tools.ocean_bathymetry`.

None of these needs Veros or the real bathymetry: the source is a synthetic
1-degree lat-lon field laid out like Veros' ``global_1deg`` asset (``xt`` from
90.5E, ``yt`` from 79.5S, ``bathymetry`` in metres negative down).
"""

from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from jem.components.veros.setups._layers import LAYER_THICKNESSES
from jem.tools.ocean_bathymetry import (
    MAXIMUM_DEPTH,
    MINIMUM_DEPTH,
    etopo_to_coarse_bathymetry,
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


def test_provenance_attributes_and_the_ocean_fraction_weight_pass_through(tmp_path):
    """A source's ``source_*`` attributes reach the output; ``ocean_fraction`` weights the mean."""
    lon = np.arange(90.5, 450.0, 1.0)
    lat = np.arange(-79.5, 80.0, 1.0)
    lat2, lon2 = np.meshgrid(lat, lon % 360, indexing="ij")
    depth = np.where(lon2 < 180, 1000.0, 3000.0)

    def write(path, fraction):
        data = {"bathymetry": (("yt", "xt"), (-depth).astype(np.float32))}
        if fraction is not None:
            data["ocean_fraction"] = (("yt", "xt"), fraction.astype(np.float32))
        xr.Dataset(data, coords={"xt": lon, "yt": lat}, attrs={"source_doi": "10.0/test"}).to_netcdf(path)
        return ocean_bathymetry(path, SCRIP_FILE, MASK_FILE, path.with_name("out_" + path.name))

    with xr.open_dataset(write(tmp_path / "plain.nc", None)) as plain:
        plain = plain.load()
    # The 3000 m side is three times as much ocean per source cell as the 1000 m side.
    with xr.open_dataset(write(tmp_path / "weighted.nc", np.where(lon2 < 180, 0.25, 0.75))) as weighted:
        weighted = weighted.load()
    assert weighted.attrs["source_doi"] == "10.0/test"
    d_plain, d_weighted = plain["depth"].to_numpy(), weighted["depth"].to_numpy()
    lat_t, lon_t = plain["lat"].to_numpy(), plain["lon"].to_numpy() % 360
    mixed = ~np.isnan(d_plain) & (np.abs(lat_t) < 60) & (np.abs(lon_t - 180) < 1.5)
    assert mixed.any()
    assert np.all(d_weighted[mixed] > d_plain[mixed])
    pure = ~np.isnan(d_plain) & (np.abs(lat_t) < 60) & (np.abs(lon_t - 180) > 10) & (lon_t > 10) & (lon_t < 350)
    np.testing.assert_allclose(d_weighted[pure], d_plain[pure])


def test_etopo_block_average_takes_ocean_points_only(tmp_path):
    """Per block: mean of -elevation over the points below zero, and their share of the block."""
    n_lat, n_lon, block = 8, 16, 4
    lat = -90 + (np.arange(n_lat) + 0.5) * (180 / n_lat)
    lon = -180 + (np.arange(n_lon) + 0.5) * (360 / n_lon)
    z = np.full((n_lat, n_lon), 100.0, dtype=np.float32)  # land
    z[:4, :4] = -1000.0  # block (0, 0): all ocean
    z[:4, 4:8] = np.where(np.arange(16).reshape(4, 4) % 2 == 0, -500.0, 300.0)  # half ocean
    z[4:, 8:12] = np.nan  # a fill value read as missing: not ocean
    source = tmp_path / "etopo.nc"
    xr.Dataset({"z": (("lat", "lon"), z)}, coords={"lat": lat, "lon": lon}).to_netcdf(source)

    out = etopo_to_coarse_bathymetry(source, tmp_path / "coarse.nc", block=block, band_rows=4)
    with xr.open_dataset(out) as ds:
        bathymetry, fraction = ds["bathymetry"].to_numpy(), ds["ocean_fraction"].to_numpy()
        np.testing.assert_allclose(ds["yt"], lat.reshape(-1, block).mean(axis=1))
        np.testing.assert_allclose(ds["xt"], lon.reshape(-1, block).mean(axis=1))
        assert ds.attrs["source_doi"] == "10.25921/fd45-gt74"
    assert bathymetry.shape == fraction.shape == (2, 4)
    np.testing.assert_allclose(bathymetry[0, 0], -1000.0)
    np.testing.assert_allclose(fraction[0, 0], 1.0)
    np.testing.assert_allclose(bathymetry[0, 1], -500.0)
    np.testing.assert_allclose(fraction[0, 1], 0.5)
    np.testing.assert_allclose(bathymetry[1, 2], 0.0)
    np.testing.assert_allclose(fraction[1, 2], 0.0)
    np.testing.assert_allclose(fraction[0, 2], 0.0)


def test_etopo_block_average_refuses_a_grid_the_block_does_not_divide(tmp_path):
    source = tmp_path / "etopo.nc"
    xr.Dataset(
        {"z": (("lat", "lon"), np.zeros((6, 8)))}, coords={"lat": np.arange(6.0), "lon": np.arange(8.0)}
    ).to_netcdf(source)
    with pytest.raises(ValueError, match="multiple of block"):
        etopo_to_coarse_bathymetry(source, tmp_path / "out.nc", block=4, band_rows=4)
    with pytest.raises(ValueError, match="band_rows"):
        etopo_to_coarse_bathymetry(source, tmp_path / "out.nc", block=2, band_rows=3)


# ---------------------------------------------------------------------------
# The packaged ETOPO 2022 bathymetry.
# ---------------------------------------------------------------------------

PACKAGED_FILE = DATA / "bathymetry_RotatedGaussianLatLon.nc"
#: The 0.25-degree intermediate the packaged file is made from (the 478 MB
#: ETOPO source is not shipped; see the module docstring for the full recipe).
INTERMEDIATE_FILE = Path(__file__).parent / "data" / "etopo2022_0p25deg_ocean_depth.nc"


@pytest.fixture(scope="module")
def packaged():
    with xr.open_dataset(str(PACKAGED_FILE)) as ds:
        return ds.load()


def _box_mean(ds, lat0, lat1, lon0, lon1):
    """Mean depth of the cells whose TRUE centres lie in the box (lon in -180..180)."""
    depth = ds["depth"].to_numpy()
    lat = ds["lat"].to_numpy()
    lon = (ds["lon"].to_numpy() + 180.0) % 360.0 - 180.0
    inside = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1) & ~np.isnan(depth)
    assert inside.any(), "empty box"
    return float(depth[inside].mean())


def test_packaged_bathymetry_is_consistent_with_the_grid_and_mask(packaged):
    wet = read_wet_mask(MASK_FILE, 0.5, NY, NX)
    depth = packaged["depth"].to_numpy()
    assert depth.shape == (NY, NX)
    np.testing.assert_array_equal(~np.isnan(depth), wet)
    assert packaged.attrs["wet_cells"] == wet.sum()
    assert depth[wet].min() >= MINIMUM_DEPTH
    assert depth[wet].max() <= MAXIMUM_DEPTH
    with xr.open_dataset(SCRIP_FILE) as grid:
        np.testing.assert_array_equal(packaged["lat"].to_numpy(), grid["grid_center_lat"].to_numpy().reshape(NY, NX))
        np.testing.assert_array_equal(packaged["lon"].to_numpy(), grid["grid_center_lon"].to_numpy().reshape(NY, NX))
    # ETOPO reaches both poles: no wet cell relied on its neighbours' depths.
    assert packaged.attrs["cells_filled_from_neighbours"] == 0
    assert packaged.attrs["cells_isolated_set_to_minimum_depth"] == 0


def test_packaged_bathymetry_records_its_source_and_terms(packaged):
    assert packaged.attrs["source_doi"] == "10.25921/fd45-gt74"
    assert "ETOPO 2022" in packaged.attrs["source_citation"]
    assert "CC0" in packaged.attrs["source_terms"]
    assert "ETOPO_2022_v1_60s_N90W180_surface.nc" in packaged.attrs["source_url"]
    assert packaged.attrs["generator"].endswith("ocean_bathymetry")


def test_packaged_bathymetry_has_recognisable_geography_in_true_coordinates(packaged):
    """The rotated grid's indices are not geography: check by true longitude and latitude."""
    depth = packaged["depth"].to_numpy()
    lat = packaged["lat"].to_numpy()
    wet = ~np.isnan(depth)

    # The deepest tenth of cells is open ocean, away from the polar seas.
    deepest = wet & (depth >= np.percentile(depth[wet], 90))
    assert deepest.sum() >= 250
    assert -65.0 < lat[deepest].min() and lat[deepest].max() < 60.0
    assert _box_mean(packaged, 10, 40, -170, -130) > 4500.0  # central North Pacific
    assert _box_mean(packaged, -40, -10, -30, 0) > 3500.0  # South Atlantic
    assert _box_mean(packaged, -35, -5, 60, 100) > 3500.0  # Indian Ocean

    # Shelf and marginal seas are shallow.
    for name, box in {
        "North Sea": (52, 60, -2, 8),
        "Baltic": (54, 65, 10, 28),
        "East China Sea": (25, 40, 118, 126),
        "Hudson Bay": (52, 62, -92, -80),
        "Barents Sea": (70, 80, 20, 50),
    }.items():
        assert _box_mean(packaged, *box) < 400.0, name
    assert 1000.0 < _box_mean(packaged, 33, 42, 0, 30) < 2500.0  # Mediterranean

    # The Arctic has its deep basin and its shelves, not one shelf depth.
    arctic = wet & (lat > 75)
    assert arctic.sum() > 100
    assert depth[arctic].mean() > 1000.0
    assert depth[arctic].std() > 1000.0
    assert _box_mean(packaged, 82, 90, -180, 180) > 2000.0
    # The Antarctic margin has deep ocean offshore of its shelf.
    assert depth[wet & (lat < -60)].max() > 4000.0


def test_regenerating_from_the_intermediate_reproduces_the_packaged_file(tmp_path, packaged):
    regenerated = ocean_bathymetry(INTERMEDIATE_FILE, SCRIP_FILE, MASK_FILE, tmp_path / "again.nc")
    with xr.open_dataset(regenerated) as again:
        np.testing.assert_allclose(again["depth"].to_numpy(), packaged["depth"].to_numpy(), rtol=1e-6, equal_nan=True)
        for key in (
            "wet_cells",
            "cells_filled_from_neighbours",
            "cells_isolated_set_to_minimum_depth",
            "cells_raised_to_minimum_depth",
            "cells_clipped_to_maximum_depth",
            "source_doi",
            "source_terms",
        ):
            assert again.attrs[key] == packaged.attrs[key]


def test_the_packaged_file_is_small():
    assert Path(str(PACKAGED_FILE)).stat().st_size < 200_000
