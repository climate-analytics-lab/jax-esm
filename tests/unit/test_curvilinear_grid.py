"""
Tests for jem/mapping/curvilinear_grid.py and the "Curvilinear" grid_universe
in jem/mapping/builtin_grid_generator.py.

Covers:
  - read_curvilinear_grid_file: shapes, degrees->radians conversion, and
    robustness to on-disk (i, j) vs (j, i) dimension order.
  - fmask/bmask validation (out-of-range fraction, non-binary mask).
  - generate_grids_from_grid_specification("Curvilinear::...", mask_file=...)
    end-to-end wiring, including the missing-mask_file error.

Synthetic netCDF files are used instead of real grid data files, since *.nc
files are gitignored and not guaranteed to be present in a fresh checkout.
"""

import numpy as np
import pytest
import xarray as xr

from jem.mapping.curvilinear_grid import (
    CurvilinearGridData,
    read_curvilinear_grid_file,
)
from jem.mapping.builtin_grid_generator import (
    generate_grids_from_grid_specification,
    get_curvilinear_grids,
)

NI, NJ = 4, 3

LON_DEG = np.array([10.0, 20.0, 30.0, 40.0])
LAT_DEG = np.array([-30.0, 0.0, 30.0])

# "truth" arrays in (nj, ni) order -- deliberately not a broadcast of
# LON_DEG/LAT_DEG, to emulate a distorted (curvilinear) geographic location.
TRUE_LON_DEG_JI = np.array(
    [[11.0, 21.0, 31.0, 41.0],
     [12.0, 22.0, 32.0, 42.0],
     [13.0, 23.0, 33.0, 43.0]]
)
TRUE_LAT_DEG_JI = np.array(
    [[-29.0, -28.0, -27.0, -26.0],
     [1.0, 2.0, 3.0, 4.0],
     [31.0, 32.0, 33.0, 34.0]]
)
LAND_FRACTION_JI = np.array(
    [[0.0, 0.3, 1.0, 0.0],
     [1.0, 0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0, 0.0]]
)
LAND_SEA_MASK_JI = (LAND_FRACTION_JI >= 0.5).astype(np.int8)


def _make_dataset(dim_order="ij"):
    assert dim_order in ("ij", "ji")

    # .copy() so mutating one test's dataset never touches the shared
    # module-level truth arrays (.T alone would return a view).
    if dim_order == "ij":
        dims2d = ("i", "j")
        true_lon = TRUE_LON_DEG_JI.T.copy()
        true_lat = TRUE_LAT_DEG_JI.T.copy()
        land_fraction = LAND_FRACTION_JI.T.copy()
        land_sea_mask = LAND_SEA_MASK_JI.T.copy()
    else:
        dims2d = ("j", "i")
        true_lon = TRUE_LON_DEG_JI.copy()
        true_lat = TRUE_LAT_DEG_JI.copy()
        land_fraction = LAND_FRACTION_JI.copy()
        land_sea_mask = LAND_SEA_MASK_JI.copy()

    return xr.Dataset(
        data_vars=dict(
            true_lon=(dims2d, true_lon),
            true_lat=(dims2d, true_lat),
            land_fraction=(dims2d, land_fraction),
            land_sea_mask=(dims2d, land_sea_mask),
        ),
        coords=dict(i=("i", LON_DEG), j=("j", LAT_DEG)),
    )


@pytest.mark.parametrize("dim_order", ["ij", "ji"])
def test_read_curvilinear_grid_file(tmp_path, dim_order):
    path = tmp_path / f"grid_{dim_order}.nc"
    _make_dataset(dim_order).to_netcdf(path)

    data = read_curvilinear_grid_file(str(path))

    assert isinstance(data, CurvilinearGridData)
    assert data.longitude.shape == (NI,)
    assert data.latitude.shape == (NJ,)
    assert data.true_longitude.shape == (NI, NJ)
    assert data.true_latitude.shape == (NI, NJ)
    assert data.fmask.shape == (NI, NJ)
    assert data.bmask.shape == (NI, NJ)

    np.testing.assert_allclose(np.asarray(data.longitude), np.deg2rad(LON_DEG))
    np.testing.assert_allclose(np.asarray(data.latitude), np.deg2rad(LAT_DEG))

    # Reader always returns (ni, nj); transpose back to (nj, ni) to compare
    # against the canonical truth arrays regardless of on-disk dim order.
    np.testing.assert_allclose(np.asarray(data.true_longitude).T, np.deg2rad(TRUE_LON_DEG_JI))
    np.testing.assert_allclose(np.asarray(data.true_latitude).T, np.deg2rad(TRUE_LAT_DEG_JI))
    np.testing.assert_allclose(np.asarray(data.fmask).T, LAND_FRACTION_JI)
    np.testing.assert_allclose(np.asarray(data.bmask).T, LAND_SEA_MASK_JI.astype(float))


def test_read_curvilinear_grid_file_agrees_across_dim_order(tmp_path):
    path_ij = tmp_path / "grid_ij.nc"
    path_ji = tmp_path / "grid_ji.nc"
    _make_dataset("ij").to_netcdf(path_ij)
    _make_dataset("ji").to_netcdf(path_ji)

    data_ij = read_curvilinear_grid_file(str(path_ij))
    data_ji = read_curvilinear_grid_file(str(path_ji))

    np.testing.assert_allclose(np.asarray(data_ij.true_longitude), np.asarray(data_ji.true_longitude))
    np.testing.assert_allclose(np.asarray(data_ij.true_latitude), np.asarray(data_ji.true_latitude))
    np.testing.assert_allclose(np.asarray(data_ij.fmask), np.asarray(data_ji.fmask))
    np.testing.assert_allclose(np.asarray(data_ij.bmask), np.asarray(data_ji.bmask))


def test_read_curvilinear_grid_file_rejects_out_of_range_fraction(tmp_path):
    ds = _make_dataset("ij")
    ds["land_fraction"][0, 0] = 1.5
    path = tmp_path / "bad_fmask.nc"
    ds.to_netcdf(path)

    with pytest.raises(AssertionError):
        read_curvilinear_grid_file(str(path))


def test_read_curvilinear_grid_file_rejects_non_binary_mask(tmp_path):
    ds = _make_dataset("ij")
    ds["land_sea_mask"][0, 0] = 2
    path = tmp_path / "bad_bmask.nc"
    ds.to_netcdf(path)

    with pytest.raises(AssertionError):
        read_curvilinear_grid_file(str(path))


def test_get_curvilinear_grids(tmp_path):
    path = tmp_path / "grid.nc"
    _make_dataset("ij").to_netcdf(path)

    grids = get_curvilinear_grids("test_family", mask_file=str(path))

    assert set(grids.keys()) == {"T"}
    grid = grids["T"]
    assert grid.grid_specification.grid_universe == "Curvilinear"
    assert grid.grid_specification.grid_family == "test_family"
    assert grid.shape == (NI, NJ)
    assert grid.bmask.shape == (NI, NJ)
    assert grid.fmask.shape == (NI, NJ)
    assert grid.true_latitude.shape == (NI, NJ)
    assert grid.true_longitude.shape == (NI, NJ)


def test_get_curvilinear_grids_requires_mask_file():
    with pytest.raises(ValueError):
        get_curvilinear_grids("test_family", mask_file=None)


def test_generate_grids_from_grid_specification_curvilinear(tmp_path):
    path = tmp_path / "grid.nc"
    _make_dataset("ij").to_netcdf(path)

    grids = generate_grids_from_grid_specification(
        "Curvilinear::test_family", mask_file=str(path)
    )

    grid = grids["T"]
    assert str(grid.grid_specification) == "Curvilinear::test_family"
    assert grid.shape == (NI, NJ)
