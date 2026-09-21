"""Tests for :mod:`jem.tools.idealised_terrain`."""

from importlib import resources

import numpy as np
import pytest
import xarray as xr

from jem.tools.idealised_terrain import CAP_LATITUDE, idealised_terrain

DATA = resources.files("jem.data")
REFERENCE_FILE = str(DATA / "terrain_JCM_T31.nc")
PACKAGED_DOUBLE_DRAKE_FILE = str(DATA / "terrain_double_drake_T31.nc")


def test_double_drake_matches_the_packaged_file(tmp_path):
    """Regenerating from the reference file reproduces the packaged one exactly.

    This is what keeps the tool and the shipped data from drifting apart: if
    either changes without the other, this test catches it. Dtype is checked
    too, not only values -- `assert_array_equal` alone would miss a fresh
    file drifting to a different precision than the one actually shipped.
    """
    regenerated = idealised_terrain(
        REFERENCE_FILE, "double_drake", tmp_path / "terrain_double_drake_T31.nc"
    )
    packaged = xr.open_dataset(PACKAGED_DOUBLE_DRAKE_FILE)
    fresh = xr.open_dataset(regenerated)
    np.testing.assert_array_equal(fresh["lsm"].to_numpy(), packaged["lsm"].to_numpy())
    np.testing.assert_array_equal(fresh["orog"].to_numpy(), packaged["orog"].to_numpy())
    assert fresh["lsm"].dtype == packaged["lsm"].dtype
    assert fresh["orog"].dtype == packaged["orog"].dtype


def test_aquaplanet_is_land_only_at_the_caps(tmp_path):
    """`|lat| >= CAP_LATITUDE` is land, and nothing else is."""
    output_file = idealised_terrain(
        REFERENCE_FILE, "aquaplanet", tmp_path / "terrain_aquaplanet.nc"
    )
    ds = xr.open_dataset(output_file)
    lat = ds["lat"].to_numpy()
    # `lsm` is `(lon, lat)`; broadcasting `(lat,)` against it aligns on the
    # trailing (lat) axis, giving every longitude the same land/ocean value.
    expected = np.broadcast_to(
        (np.abs(lat) >= CAP_LATITUDE).astype(ds["lsm"].dtype), ds["lsm"].shape
    )
    np.testing.assert_array_equal(ds["lsm"].to_numpy(), expected)


def test_unknown_planet_type_names_the_valid_ones(tmp_path):
    """A bad `planet_type` raises `ValueError` listing the valid ones."""
    with pytest.raises(ValueError, match="double_drake"):
        idealised_terrain(REFERENCE_FILE, "gas_giant", tmp_path / "out.nc")


def test_orography_is_zero(tmp_path):
    """An idealised planet has no real orography."""
    output_file = idealised_terrain(
        REFERENCE_FILE, "toy_earth", tmp_path / "terrain_toy_earth.nc"
    )
    ds = xr.open_dataset(output_file)
    np.testing.assert_array_equal(ds["orog"].to_numpy(), 0.0)
