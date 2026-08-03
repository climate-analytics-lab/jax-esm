"""
Integration tests for jem/mapping/curvilinear_grid.py against the real bundled
grid/mask files (as opposed to test_curvilinear_grid.py, which uses synthetic
netCDF files to test reader mechanics in isolation).

*.nc files are gitignored, so these tests skip (rather than fail) when the
data files aren't present in the local checkout.
"""

from pathlib import Path

import numpy as np
import pytest

import jem
from jem.mapping.curvilinear_grid import read_curvilinear_grid_file
from jem.mapping.builtin_grid_generator import generate_grids_from_grid_specification

DATA_DIR = Path(jem.__file__).parent / "data" / "grid"
ROTATED_GRID_FILE = DATA_DIR / "landsea_mask_RG_4.00deg.nc"
JCM_T31_FILE = DATA_DIR / "landsea_mask_JCM_T31.nc"

requires_rotated_grid_file = pytest.mark.skipif(
    not ROTATED_GRID_FILE.exists(), reason=f"{ROTATED_GRID_FILE} not present locally"
)
requires_jcm_t31_file = pytest.mark.skipif(
    not JCM_T31_FILE.exists(), reason=f"{JCM_T31_FILE} not present locally"
)


@requires_rotated_grid_file
def test_read_curvilinear_grid_file_landsea_mask_rg_4_00deg():
    data = read_curvilinear_grid_file(str(ROTATED_GRID_FILE))

    ni, nj = 90, 45
    assert data.longitude.shape == (ni,)
    assert data.latitude.shape == (nj,)
    assert data.true_longitude.shape == (ni, nj)
    assert data.true_latitude.shape == (ni, nj)
    assert data.fmask.shape == (ni, nj)
    assert data.bmask.shape == (ni, nj)

    # Logical axes span the full sphere, in radians.
    assert np.asarray(data.longitude).min() >= 0.0
    assert np.asarray(data.longitude).max() <= 2 * np.pi
    assert np.asarray(data.latitude).min() >= -0.5 * np.pi
    assert np.asarray(data.latitude).max() <= 0.5 * np.pi

    assert np.all((np.asarray(data.fmask) >= 0.0) & (np.asarray(data.fmask) <= 1.0))
    assert set(np.unique(np.asarray(data.bmask))) <= {0.0, 1.0}
    assert int(np.asarray(data.bmask).sum()) == 1423

    # This is a rotated grid: the true geographic location differs from a
    # plain broadcast of the logical i/j axes.
    broadcast_longitude = np.broadcast_to(
        np.asarray(data.longitude)[:, None], (ni, nj)
    )
    assert not np.allclose(np.asarray(data.true_longitude), broadcast_longitude)


@requires_rotated_grid_file
def test_get_curvilinear_grids_landsea_mask_rg_4_00deg():
    grids = generate_grids_from_grid_specification(
        "Curvilinear::RG_4.00deg", mask_file=str(ROTATED_GRID_FILE)
    )
    grid = grids["T"]

    assert str(grid.grid_specification) == "Curvilinear::RG_4.00deg"
    assert grid.shape == (90, 45)
    assert grid.bmask.shape == grid.shape
    assert grid.fmask.shape == grid.shape
    assert grid.true_latitude.shape == grid.shape
    assert grid.true_longitude.shape == grid.shape


@requires_jcm_t31_file
def test_read_curvilinear_grid_file_rejects_landsea_mask_jcm_t31():
    # landsea_mask_JCM_T31.nc is a plain (lon, lat) grid, not a curvilinear
    # one -- it has no i/j/true_lon/true_lat variables, so the curvilinear
    # reader (which defaults to those names) must fail loudly rather than
    # silently misreading it.
    with pytest.raises(KeyError):
        read_curvilinear_grid_file(str(JCM_T31_FILE))
