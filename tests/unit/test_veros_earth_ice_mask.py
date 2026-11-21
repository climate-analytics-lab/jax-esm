"""The packaged mask a sea ice on the ``veros-earth`` ocean grid is built on.

No Veros needed: the wet mask is read by ``jem.tools.veros_earth_terrain.wet_mask``,
which the Veros-side test (``test_veros_seaice.py``) holds to ``GridInfo``'s own.
"""

from importlib import resources

import numpy as np
import xarray as xr

from jem.components.slab.grid import SlabGrid
from jem.tools.veros_earth_ice_mask import PACKAGED_NAME, veros_earth_ice_mask
from jem.tools.veros_earth_terrain import wet_mask

DATA = resources.files("jem.data")
PACKAGED = str(DATA / PACKAGED_NAME)
SCRIP = str(DATA / "RotatedGaussianLatLon.SCRIP.nc")
FRACTION = str(DATA / "landsea_mask_fraction_RotatedGaussianLatLon.nc")


def test_regenerating_the_mask_reproduces_the_packaged_file(tmp_path):
    regenerated = xr.open_dataset(veros_earth_ice_mask(tmp_path / "mask.nc"))
    packaged = xr.open_dataset(PACKAGED)
    np.testing.assert_array_equal(regenerated["lsm"].to_numpy(), packaged["lsm"].to_numpy())
    assert regenerated["lsm"].dtype == packaged["lsm"].dtype
    for name in ("title", "comment", "source_ocean_mask", "source_grid",
                 "ocean_land_fraction_threshold", "generated_by"):
        assert packaged.attrs[name] == regenerated.attrs[name]


def test_the_packaged_mask_is_binary_and_is_the_complement_of_veros_wet_mask():
    lsm = xr.open_dataset(PACKAGED)["lsm"].to_numpy()
    assert lsm.shape == (1, 48, 96)
    assert set(np.unique(lsm)) == {0.0, 1.0}
    wet = wet_mask(FRACTION, 96, 48)
    np.testing.assert_array_equal(1.0 - lsm[0].transpose(), wet)
    # Not the un-thresholded file: that one has fractional cells, which is why
    # it cannot be a sea ice's land fraction over Veros.
    fraction = xr.open_dataset(FRACTION)["lsm"].to_numpy()
    assert np.any((fraction > 0.0) & (fraction < 1.0))


def test_a_seaice_grid_built_from_it_has_the_wet_mask_as_its_sea():
    """The runner's reading of ``land_fraction_file`` gives ``sea_fraction == wet`` exactly."""
    from jem.runners import _land_fraction

    grid = SlabGrid.from_scrip(SCRIP, fractional_mask=_land_fraction(PACKAGED))
    np.testing.assert_array_equal(
        1.0 - np.asarray(grid.fractional_mask), wet_mask(FRACTION, 96, 48))
