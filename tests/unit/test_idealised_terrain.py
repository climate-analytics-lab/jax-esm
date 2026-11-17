"""Tests for :mod:`jem.tools.idealised_terrain`."""

from importlib import resources

import numpy as np
import pytest
import xarray as xr

from jem.tools.idealised_terrain import (
    CAP_LATITUDE,
    earth_ice_free_land_means,
    grid_axes,
    idealised_forcing,
    idealised_terrain,
)

DATA = resources.files("jem.data")
REFERENCE_FILE = str(DATA / "terrain_JCM_T31.nc")
PACKAGED_DOUBLE_DRAKE_FILE = str(DATA / "terrain_double_drake_T31.nc")
PACKAGED_FORCING_FILE = str(DATA / "forcing_idealised_T31.nc")


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


def _jcm_t31_axes():
    """JCM's own T31 axes, the way ``jcm.utils.data_to_xarray`` writes them."""
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    from jem.components.slab.grid import to_degrees

    horizontal = get_speedy_coords().horizontal
    return to_degrees(horizontal.longitudes), to_degrees(horizontal.latitudes)


def test_axes_are_jcms_own_grid_not_the_reference_files_labels(tmp_path):
    """The written ``lon``/``lat`` are the grid's, exactly, whatever the reference says.

    The packaged ``terrain_JCM_T31.nc`` labels its first row -87.66, where
    JCM's T31 grid has -87.159, and JCM interpolates a terrain file onto its
    grid by coordinate, so the output must not inherit those labels.
    """
    lon, lat = _jcm_t31_axes()
    reference = xr.open_dataset(REFERENCE_FILE)
    assert abs(float(reference["lat"][0]) - lat[0]) > 0.4  # the premise of the test
    for planet_type in ("double_drake", "aquaplanet"):
        ds = xr.open_dataset(
            idealised_terrain(REFERENCE_FILE, planet_type, tmp_path / f"{planet_type}.nc")
        )
        np.testing.assert_array_equal(ds["lon"].to_numpy(), lon)
        np.testing.assert_array_equal(ds["lat"].to_numpy(), lat)
    packaged = xr.open_dataset(PACKAGED_DOUBLE_DRAKE_FILE)
    np.testing.assert_array_equal(packaged["lon"].to_numpy(), lon)
    np.testing.assert_array_equal(packaged["lat"].to_numpy(), lat)


def test_grid_axes_match_the_packaged_scrip_grid():
    """The SCRIP file ``veros-double-drake`` reads its ocean grid from is the same grid."""
    lon, lat = grid_axes(96, 48)
    scrip = xr.open_dataset(str(DATA / "JCM_T31.SCRIP.nc"))
    # SCRIP cells run with the longitude index varying fastest.
    centre_lat = scrip["grid_center_lat"].to_numpy().reshape(48, 96)
    centre_lon = scrip["grid_center_lon"].to_numpy().reshape(48, 96)
    np.testing.assert_allclose(centre_lat[:, 0], lat, atol=1e-6)
    np.testing.assert_allclose(centre_lon[0], lon, atol=1e-6)


def test_grid_axes_rejects_a_shape_jcm_has_no_grid_for():
    with pytest.raises(ValueError, match="no grid"):
        grid_axes(97, 48)


def test_jcm_reads_the_double_drake_mask_unchanged():
    """JCM's terrain loader returns the file's own land-sea mask on its grid.

    The loader interpolates the file onto the grid by coordinate, so this is
    what shows that the labels and the mask's rows agree.
    """
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.terrain import get_terrain

    horizontal = get_speedy_coords().horizontal
    _, fmask = get_terrain(terrain_file=PACKAGED_DOUBLE_DRAKE_FILE, grid=horizontal)
    np.testing.assert_array_equal(
        np.asarray(fmask), xr.open_dataset(PACKAGED_DOUBLE_DRAKE_FILE)["lsm"].to_numpy()
    )


def test_double_drake_mask_is_the_documented_geometry():
    """Two meridional barriers (two cells wide) from 45S to the cap, and the caps, and nothing else."""
    ds = xr.open_dataset(PACKAGED_DOUBLE_DRAKE_FILE)
    lon, lat = ds["lon"].to_numpy(), ds["lat"].to_numpy()
    land = ds["lsm"].to_numpy() == 1
    cap = np.abs(lat) >= CAP_LATITUDE
    assert land[:, cap].all()
    barrier_columns = np.flatnonzero(land[:, (lat > -45) & (lat < CAP_LATITUDE)].all(axis=1))
    np.testing.assert_allclose(lon[barrier_columns], [0.0, 3.75, 67.5, 71.25])
    assert int(land.sum()) == 896


def test_forcing_matches_the_packaged_file(tmp_path):
    """Regenerating the forcing reproduces the packaged one, values and dtypes."""
    regenerated = idealised_forcing(
        PACKAGED_DOUBLE_DRAKE_FILE, tmp_path / "forcing_idealised_T31.nc"
    )
    packaged = xr.open_dataset(PACKAGED_FORCING_FILE)
    fresh = xr.open_dataset(regenerated)
    assert set(fresh.data_vars) == set(packaged.data_vars)
    for name in packaged.data_vars:
        np.testing.assert_array_equal(fresh[name].to_numpy(), packaged[name].to_numpy())
        assert fresh[name].dtype == packaged[name].dtype
        assert fresh[name].dims == packaged[name].dims
    np.testing.assert_array_equal(fresh["time"].to_numpy(), packaged["time"].to_numpy())
    np.testing.assert_array_equal(fresh["lat"].to_numpy(), packaged["lat"].to_numpy())


def test_packaged_forcing_is_small():
    """It ships in the wheel: a constant field must be stored compressed."""
    from pathlib import Path

    assert Path(PACKAGED_FORCING_FILE).stat().st_size < 200_000


def test_forcing_defaults_are_rounded_from_earths_ice_free_land():
    """The generator's defaults sit next to the numbers its docstring derives them from."""
    import inspect

    earth = earth_ice_free_land_means()
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(idealised_forcing).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    assert defaults["bare_land_albedo"] == pytest.approx(earth["bare_land_albedo"], abs=0.02)
    assert defaults["soil_water_availability"] == pytest.approx(
        earth["soil_water_availability"], abs=0.02
    )
    assert defaults["land_skin_temperature"] == pytest.approx(
        earth["land_skin_temperature"], abs=1.0
    )


def test_forcing_loads_as_a_climatology_with_a_lit_land():
    """JCM reads the file as a monthly climatology, and the land is not black.

    ``ForcingData.zeros`` has a bare-land albedo of 0; the file's land has a
    non-zero one, no snow, and its time-varying fields can be selected on any
    date of any year (``wrap_year`` replays the twelve records).
    """
    import datetime

    import jax_datetime as jdt
    from jcm.date import DateData
    from jcm.forcing import ForcingData
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    coords = get_speedy_coords()
    forcing = ForcingData.from_file(PACKAGED_FORCING_FILE, coords, align_mode="wrap_year")
    shape = coords.horizontal.nodal_shape
    np.testing.assert_allclose(np.asarray(forcing.alb0), 0.2, rtol=1e-6)
    for year in (2000, 2031):
        date = DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(datetime.datetime(year, 7, 15))
        )
        selected = forcing.select(date)
        for name, value in (
            ("snowc_am", 0.0),
            ("sice_am", 0.0),
            ("soilw_am", 0.65),
            ("stl_am", 290.0),
            ("sea_surface_temperature", 288.15),
        ):
            field = np.asarray(getattr(selected, name))
            assert field.shape == shape, name
            np.testing.assert_allclose(field, value, rtol=1e-6, err_msg=name)


def test_forcing_with_auto_align_is_refused():
    """``auto`` resolves only jax-gcm's own products, so a configuration must name the alignment."""
    from jcm.forcing import ForcingData
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    with pytest.raises(ValueError, match="auto"):
        ForcingData.from_file(PACKAGED_FORCING_FILE, get_speedy_coords(), align_mode="auto")
