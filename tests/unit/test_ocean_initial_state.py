"""Tests for :mod:`jem.tools.ocean_initial_state`.

None of these needs Veros or the real World Ocean Atlas: the generator is run
on small synthetic 1-degree files laid out like ``woa18_decav_[ts]00_01.nc``
(``t_an``/``s_an`` with dims ``(time, depth, lat, lon)``, lon from -179.5), and
the packaged file is checked for consistency with the grid, the mask and the
bathymetry, and for recognisable geography in true longitude and latitude.
"""

from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from jem.components.veros.setups._layers import LAYER_THICKNESSES
from jem.tools import ocean_initial_state as ois
from jem.tools.ocean_bathymetry import read_depth, read_wet_mask
from jem.tools.ocean_initial_state import (
    absolute_salinity_from_practical,
    adiabatic_lapse_rate,
    conservative_temperature_from_in_situ,
    conservative_temperature_from_potential,
    depth_from_pressure,
    layer_centres,
    ocean_initial_state,
    potential_temperature,
    pressure_from_depth,
    read_initial_state,
    static_stability,
    wet_layer_mask,
)

DATA = resources.files("jem.data")
SCRIP_FILE = str(DATA / "RotatedGaussianLatLon.SCRIP.nc")
MASK_FILE = str(DATA / "landsea_mask_fraction_RotatedGaussianLatLon.nc")
BATHYMETRY_FILE = str(DATA / "bathymetry_RotatedGaussianLatLon.nc")
PACKAGED_FILE = DATA / "ocean_initial_state_RotatedGaussianLatLon.nc"
NX, NY, NZ = 96, 48, len(LAYER_THICKNESSES)

#: A reduced set of WOA's standard depths, enough to resolve the model's layers.
SYNTHETIC_DEPTHS = np.array(
    [0.0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 5500]
)


# ---------------------------------------------------------------------------
# The conversions, against published check values.
# ---------------------------------------------------------------------------

# The check values below are those printed with the algorithms: Fofonoff and
# Millard (1983, UNESCO Technical Papers in Marine Science 44) for the
# EOS-80 functions, and the TEOS-10 GSW Oceanographic Toolbox's documented
# check set for the Conservative Temperature polynomial (six points from 4 to
# 29 degrees C).
GSW_SA = np.array([34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324])
GSW_PT = np.array([28.7832, 28.4209, 22.7850, 10.2305, 6.8292, 4.3245])
GSW_CT = np.array([28.8099, 28.4392, 22.7862, 10.2262, 6.8272, 4.3236])
GSW_T = np.array([28.7856, 28.4329, 22.8103, 10.2600, 6.8863, 4.4036])
GSW_P = np.array([10.0, 50.0, 125.0, 250.0, 600.0, 1000.0])


def test_the_adiabatic_lapse_rate_and_potential_temperature_match_the_unesco_check_values():
    np.testing.assert_allclose(adiabatic_lapse_rate(40.0, 40.0, 10000.0), 3.255976e-4, rtol=1e-6)
    np.testing.assert_allclose(potential_temperature(40.0, 40.0, 10000.0), 36.89073, atol=1e-5)


def test_depth_and_pressure_match_the_unesco_check_value_and_invert_one_another():
    np.testing.assert_allclose(depth_from_pressure(10000.0, 30.0), 9712.653, atol=1e-3)
    depth = np.array([0.0, 5.0, 100.0, 1500.0, 5500.0])
    for lat in (0.0, 45.0, -80.0):
        np.testing.assert_allclose(depth_from_pressure(pressure_from_depth(depth, lat), lat), depth, atol=1e-6)
    # About one dbar per metre, a little more with depth.
    assert 1.0 < float(pressure_from_depth(5000.0, 0.0)) / 5000.0 < 1.03


def test_conservative_temperature_matches_the_gsw_check_values():
    np.testing.assert_allclose(conservative_temperature_from_potential(GSW_SA, GSW_PT), GSW_CT, atol=1e-4)


def test_in_situ_temperature_converts_to_the_published_potential_temperature():
    """EOS-80 potential temperature against the TEOS-10 check values: a few tenths of a millikelvin."""
    practical = GSW_SA / ois.REFERENCE_SALINITY_FACTOR
    np.testing.assert_allclose(potential_temperature(practical, GSW_T, GSW_P), GSW_PT, atol=5e-4)
    np.testing.assert_allclose(
        conservative_temperature_from_in_situ(practical, GSW_T, GSW_P), GSW_CT, atol=5e-4
    )


def test_the_in_situ_to_potential_difference_is_not_negligible_at_depth():
    """The reason the conversion exists: cold deep water is warmer in situ than potentially."""
    practical = np.array([34.7, 34.7])
    pressure = pressure_from_depth(np.array([1000.0, 5000.0]), 0.0)
    potential = potential_temperature(practical, np.array([3.0, 1.5]), pressure)
    difference = np.array([3.0, 1.5]) - potential
    assert 0.06 < difference[0] < 0.08
    assert 0.4 < difference[1] < 0.5


def test_absolute_salinity_is_reference_salinity():
    np.testing.assert_allclose(absolute_salinity_from_practical(35.0), 35.16504, rtol=1e-12)
    np.testing.assert_allclose(absolute_salinity_from_practical(np.array([0.0, 10.0])), [0.0, 10.0 * 35.16504 / 35])


def test_the_linear_equation_of_state_constants_are_the_veros_ones():
    """The restated constants equal Veros' own, read in a separate process.

    Veros switches the process-global ``jax_enable_x64`` on the first time
    its operators are imported with the JAX backend selected, which an
    earlier test of the same session may have done. Importing
    ``veros.core.density`` here would then leave every later test in double
    precision, so the values are read in a subprocess, which cannot.
    """
    import importlib.util
    import json
    import subprocess
    import sys

    if importlib.util.find_spec("veros") is None:
        pytest.skip("veros is not installed")
    code = (
        "import json; from veros.core.density import linear_eq as eq; "
        "print(json.dumps([eq.rho0, eq.theta0, eq.S0, eq.betaT, eq.betaS]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    rho0, theta0, s0, beta_t, beta_s = json.loads(result.stdout.strip().splitlines()[-1])

    assert ois.LINEAR_EOS_RHO0 == rho0
    assert ois.LINEAR_EOS_THETA0 == pytest.approx(theta0, abs=1e-12)
    assert ois.LINEAR_EOS_S0 == s0
    assert ois.LINEAR_EOS_BETA_T == beta_t
    assert ois.LINEAR_EOS_BETA_S == beta_s


# ---------------------------------------------------------------------------
# The wet cells and the stability count.
# ---------------------------------------------------------------------------


def test_importing_the_tool_does_not_import_veros():
    import subprocess
    import sys

    code = "import sys; import jem.tools.ocean_initial_state; sys.exit('veros' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_the_wet_layer_mask_follows_the_layers_whose_centre_is_above_the_floor_with_a_two_layer_minimum():
    thickness = np.array([50.0, 70.0, 100.0, 140.0])  # centres 25, 85, 170, 290
    depth = np.array([[10.0, 100.0, 171.0, 5000.0, 400.0]])
    wet = np.array([[True, True, True, True, False]])
    mask = wet_layer_mask(depth, wet, thickness)
    np.testing.assert_array_equal(mask[0, 0], [True, True, False, False])  # raised to two layers
    np.testing.assert_array_equal(mask[0, 1], [True, True, False, False])  # centre 170 below 100 m floor
    np.testing.assert_array_equal(mask[0, 2], [True, True, True, False])
    np.testing.assert_array_equal(mask[0, 3], [True, True, True, True])
    assert not mask[0, 4].any()


def test_static_stability_counts_only_wet_pairs_and_measures_the_inversion():
    # Temperature falls and salinity is constant: stable (colder is denser) when it falls downward.
    temperature = np.array([[[20.0, 20.0, np.nan]], [[10.0, 25.0, np.nan]], [[5.0, 5.0, np.nan]]])
    salinity = np.full_like(temperature, 35.0)
    result = static_stability(temperature, salinity)
    assert result["pairs"] == 4  # the NaN column has none
    assert result["inverted"] == 1  # the 20 -> 25 C pair: warmer water below colder
    expected = ois.LINEAR_EOS_BETA_T * 5.0 * ois.LINEAR_EOS_RHO0
    assert result["largest"] == pytest.approx(expected)
    assert np.isnan(result["inversion"][:, 0, 2]).all()


# ---------------------------------------------------------------------------
# The generator, on synthetic WOA-like files.
# ---------------------------------------------------------------------------


def _woa_files(directory, temperature_of, salinity_of, depths=SYNTHETIC_DEPTHS):
    """Write synthetic WOA-layout files; ``f(depth, lat, lon)`` gives each field (NaN = no ocean)."""
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    z, la, lo = np.meshgrid(depths, lat, lon, indexing="ij")
    paths = []
    for name, variable, function in (("t", "t_an", temperature_of), ("s", "s_an", salinity_of)):
        field = function(z, la, lo).astype(np.float32)[None]
        path = directory / f"woa18_decav_{name}00_01.nc"
        xr.Dataset(
            {variable: (("time", "depth", "lat", "lon"), field)},
            coords={"time": [0.0], "depth": depths.astype(np.float32), "lat": lat.astype(np.float32), "lon": lon.astype(np.float32)},
        ).to_netcdf(path)
        paths.append(str(path))
    return paths


def _generate(directory, temperature_of, salinity_of, **kwargs):
    t_file, s_file = _woa_files(directory, temperature_of, salinity_of, **kwargs)
    out = ocean_initial_state(t_file, s_file, SCRIP_FILE, MASK_FILE, BATHYMETRY_FILE, directory / "state.nc")
    with xr.open_dataset(out) as ds:
        return ds.load()


def _model_wet():
    wet2d = read_wet_mask(MASK_FILE, 0.5, NY, NX)
    depth = read_depth(BATHYMETRY_FILE, NY, NX).T  # (ny, nx)
    return np.moveaxis(wet_layer_mask(depth, wet2d, LAYER_THICKNESSES), -1, 0)  # (nz, ny, nx)


def test_the_output_is_defined_on_exactly_the_models_wet_cells(tmp_path):
    ds = _generate(tmp_path, lambda z, la, lo: 10.0 + 0 * z, lambda z, la, lo: 35.0 + 0 * z)
    wet = _model_wet()
    for name in ("temperature", "salinity"):
        assert ds[name].dims == ("depth", "latitude", "longitude")
        assert ds[name].dtype == np.float32
        np.testing.assert_array_equal(np.isfinite(ds[name].to_numpy()), wet)
    np.testing.assert_allclose(ds["depth"], layer_centres(LAYER_THICKNESSES))
    assert ds.attrs["wet_cells"] == wet.sum()
    # Every wet cell is accounted for by exactly one rule.
    assert (
        ds.attrs["cells_from_source"]
        + ds.attrs["cells_filled_horizontally"]
        + ds.attrs["cells_filled_from_layer_above"]
        + ds.attrs["cells_filled_at_surface_from_nearest_cell"]
        == wet.sum()
    )


def test_a_depth_only_profile_gives_each_layers_mean_not_its_centre_value(tmp_path):
    """The layer takes the mean of the profile over its thickness, with the conversion applied at each depth."""
    knots_z, knots_t = [0.0, 100.0, 1000.0, 5500.0], [20.0, 4.0, 3.0, 1.0]

    def temperature(z, la, lo):
        return np.interp(z, knots_z, knots_t) + 0 * la

    ds = _generate(tmp_path, temperature, lambda z, la, lo: 35.0 + 0 * z)
    # The independent expectation: Conservative Temperature at a fixed latitude
    # at each source depth, linear between them, averaged over each layer.
    lat = 45.0
    ct = conservative_temperature_from_in_situ(
        np.full(SYNTHETIC_DEPTHS.size, 35.0),
        np.interp(SYNTHETIC_DEPTHS, knots_z, knots_t),
        pressure_from_depth(SYNTHETIC_DEPTHS, lat),
    )
    bottoms = np.cumsum(LAYER_THICKNESSES)
    tops = bottoms - np.asarray(LAYER_THICKNESSES)
    expected = np.array([
        np.interp(np.linspace(t, b, 4001), SYNTHETIC_DEPTHS, ct).mean() for t, b in zip(tops, bottoms)
    ])
    centre_value = np.interp(layer_centres(LAYER_THICKNESSES), SYNTHETIC_DEPTHS, ct)
    # The second layer straddles the thermocline: the two definitions differ by several tenths of a degree.
    assert np.max(np.abs(expected - centre_value)) > 0.4

    wet = _model_wet()
    field = ds["temperature"].to_numpy()
    for k in range(NZ):
        cells = field[k][wet[k]]
        # Cells differ only by the weak latitude dependence of pressure.
        np.testing.assert_allclose(cells, expected[k], atol=2e-3)
    np.testing.assert_allclose(ds["salinity"].to_numpy()[wet], 35.0 * ois.REFERENCE_SALINITY_FACTOR, rtol=1e-6)


def test_the_regrid_follows_true_longitude_and_latitude_not_grid_indices(tmp_path):
    """A salinity that is fresh in one hemisphere and salty in the other, and likewise east and west."""
    ds = _generate(
        tmp_path,
        lambda z, la, lo: 10.0 + 0 * z,
        lambda z, la, lo: 34.0 + 2.0 * (la > 0) + 1.0 * (lo > 0),
    )
    lat, lon = ds["lat"].to_numpy(), ds["lon"].to_numpy()
    lon = (lon + 180.0) % 360.0 - 180.0
    interior = (np.abs(lat) > 15) & (np.abs(lat) < 70) & (np.abs(lon) > 15) & (np.abs(lon) < 165)
    salinity = ds["salinity"].to_numpy()[0] / ois.REFERENCE_SALINITY_FACTOR
    cells = interior & np.isfinite(salinity)
    assert cells.sum() > 500
    np.testing.assert_allclose(salinity[cells], (34.0 + 2.0 * (lat > 0) + 1.0 * (lon > 0))[cells], atol=1e-6)


def test_cells_the_source_does_not_cover_are_filled_horizontally_and_counted(tmp_path):
    """No ocean in the source north of 80 N: those wet cells take their wet neighbours' mean."""

    def temperature(z, la, lo):
        return np.where(la > 80.0, np.nan, 10.0 + 0 * z)

    ds = _generate(tmp_path, temperature, lambda z, la, lo: 35.0 + 0 * z)
    wet = _model_wet()
    assert ds.attrs["cells_filled_horizontally"] > 0
    np.testing.assert_array_equal(np.isfinite(ds["temperature"].to_numpy()), wet)
    # A uniform source: the filled cells equal their neighbours, so the whole ocean is uniform.
    surface = ds["temperature"].to_numpy()[0]
    assert (np.isfinite(surface) & (ds["lat"].to_numpy() > 82)).any()
    assert np.ptp(surface[np.isfinite(surface)]) < 0.05


def test_layers_below_the_sources_deepest_level_are_filled_from_the_layer_above_and_counted(tmp_path):
    """With no source value below 1000 m anywhere, no wet neighbour has one either: copy downward."""

    def temperature(z, la, lo):
        return np.where(z > 1000.0, np.nan, 10.0 - z / 200.0 + 0 * la)

    ds = _generate(tmp_path, temperature, lambda z, la, lo: 35.0 + 0 * z)
    wet = _model_wet()
    centres = layer_centres(LAYER_THICKNESSES)
    below = centres > 1000.0
    assert below.any()
    assert ds.attrs["cells_filled_from_layer_above"] == wet[below].sum()
    field = ds["temperature"].to_numpy()
    last_covered = np.nonzero(~below)[0][-1]
    for k in np.nonzero(below)[0]:
        np.testing.assert_array_equal(field[k][wet[k]], field[last_covered][wet[k]])


def test_an_isolated_wet_cell_the_source_does_not_reach_takes_the_nearest_filled_cells_value(tmp_path):
    """The model keeps a Great Lakes cell wet; WOA has no lake, and no wet neighbour has a value to lend."""
    lat, lon = _grid_centres()
    wet = _model_wet()[0]
    vectors = ois.unit_vectors(lat, lon)
    lake = np.unravel_index(np.argmax(vectors @ ois.unit_vectors(45.3, -81.9)), lat.shape)
    assert wet[lake]
    others = wet.copy()
    others[lake] = False
    nearest = np.unravel_index(np.argmax(np.where(others, vectors @ vectors[lake], -2.0)), lat.shape)

    def temperature(z, la, lo):
        in_lakes = (np.abs(la - 45.3) < 4.0) & (np.abs(lo + 81.9) < 5.0)
        return np.where(in_lakes, np.nan, 30.0 - np.abs(la) / 3.0 + 0 * z)

    ds = _generate(tmp_path, temperature, lambda z, la, lo: 35.0 + 0 * z)
    assert ds.attrs["cells_filled_at_surface_from_nearest_cell"] == 1
    field = ds["temperature"].to_numpy()[0]
    assert field[lake] == field[nearest]


def _grid_centres():
    with xr.open_dataset(SCRIP_FILE) as grid:
        return (
            grid["grid_center_lat"].to_numpy().reshape(NY, NX),
            grid["grid_center_lon"].to_numpy().reshape(NY, NX),
        )


def test_a_source_with_no_surface_values_is_refused(tmp_path):
    t_file, s_file = _woa_files(tmp_path, lambda z, la, lo: np.where(z < 60, np.nan, 10.0 + 0 * z), lambda z, la, lo: 35.0 + 0 * z)
    with pytest.raises(ValueError, match="surface"):
        ocean_initial_state(t_file, s_file, SCRIP_FILE, MASK_FILE, BATHYMETRY_FILE, tmp_path / "out.nc")


def test_temperature_and_salinity_on_different_grids_are_refused(tmp_path):
    t_file, _ = _woa_files(tmp_path, lambda z, la, lo: 10.0 + 0 * z, lambda z, la, lo: 35.0 + 0 * z)
    other = tmp_path / "other"
    other.mkdir()
    _, s_file = _woa_files(other, lambda z, la, lo: 10.0 + 0 * z, lambda z, la, lo: 35.0 + 0 * z, depths=SYNTHETIC_DEPTHS[:-1])
    with pytest.raises(ValueError, match="same grid"):
        ocean_initial_state(t_file, s_file, SCRIP_FILE, MASK_FILE, BATHYMETRY_FILE, tmp_path / "out.nc")


def test_the_file_reads_back_in_the_setups_layout_and_refuses_another_shape(tmp_path):
    ds = _generate(tmp_path, lambda z, la, lo: 10.0 + 0 * z, lambda z, la, lo: 35.0 + 0 * z)
    temperature, salinity, depth, centres = read_initial_state(tmp_path / "state.nc", NZ, NY, NX)
    assert temperature.shape == salinity.shape == (NX, NY, NZ)
    np.testing.assert_array_equal(np.transpose(temperature, (2, 1, 0)), ds["temperature"].to_numpy().astype(float))
    assert depth.shape == (NZ,)
    assert centres.shape == (2, NY, NX)
    with pytest.raises(ValueError, match="regenerate"):
        read_initial_state(tmp_path / "state.nc", NZ - 1, NY, NX)


# ---------------------------------------------------------------------------
# The packaged WOA18 file.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def packaged():
    with xr.open_dataset(str(PACKAGED_FILE)) as ds:
        return ds.load()


def _box_mean(ds, name, lat0, lat1, lon0, lon1, layer=0):
    """Mean over the cells whose TRUE centres lie in the box (lon in -180..180)."""
    field = ds[name].to_numpy()[layer].astype(float)
    lat = ds["lat"].to_numpy()
    lon = (ds["lon"].to_numpy() + 180.0) % 360.0 - 180.0
    inside = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1) & np.isfinite(field)
    assert inside.any(), "empty box"
    return float(field[inside].mean())


def test_packaged_state_is_consistent_with_the_grid_mask_and_bathymetry(packaged):
    wet = _model_wet()
    for name in ("temperature", "salinity"):
        field = packaged[name].to_numpy()
        assert field.shape == (NZ, NY, NX)
        assert field.dtype == np.float32
        np.testing.assert_array_equal(np.isfinite(field), wet)
    np.testing.assert_allclose(packaged["depth"], layer_centres(LAYER_THICKNESSES))
    lat, lon = _grid_centres()
    np.testing.assert_array_equal(packaged["lat"].to_numpy(), lat)
    np.testing.assert_array_equal(packaged["lon"].to_numpy(), lon)
    attrs = packaged.attrs
    assert attrs["wet_cells"] == wet.sum()
    assert (
        attrs["cells_from_source"]
        + attrs["cells_filled_horizontally"]
        + attrs["cells_filled_from_layer_above"]
        + attrs["cells_filled_at_surface_from_nearest_cell"]
        == wet.sum()
    )
    assert attrs["bathymetry_file"] == "bathymetry_RotatedGaussianLatLon.nc"
    # Almost every wet cell comes from the data.
    assert attrs["cells_from_source"] > 0.99 * wet.sum()


def test_packaged_state_records_its_source_terms_and_attribution(packaged):
    attrs = packaged.attrs
    assert "World Ocean Atlas 2018" in attrs["source_citation"]
    assert "NCEI-WOA18" in attrs["source_citation"]
    assert "10.25923/e5rn-9711" in attrs["source_citation"]  # the temperature volume
    assert "10.25923/9pgv-1224" in attrs["source_citation"]  # the salinity volume
    assert "openly available to the public" in attrs["source_terms"]
    assert "warranty" in attrs["source_terms"]
    assert "woa18_decav_t00_01.nc" in attrs["source_url"]
    assert attrs["generator"].endswith("ocean_initial_state")
    assert packaged["temperature"].attrs["standard_name"] == "sea_water_conservative_temperature"
    assert packaged["salinity"].attrs["standard_name"] == "sea_water_absolute_salinity"


def test_packaged_state_has_recognisable_geography_in_true_coordinates(packaged):
    """Indices of a rotated grid are not geography: check by true longitude and latitude."""
    assert _box_mean(packaged, "temperature", -5, 10, 120, 160) > 27.5  # the warm pool
    assert _box_mean(packaged, "temperature", -5, 5, -130, -90) < _box_mean(packaged, "temperature", -5, 10, 120, 160)
    polar_south = _box_mean(packaged, "temperature", -80, -65, -180, 180)
    assert -2.0 < polar_south < -0.8  # near the freezing point
    assert -2.0 < _box_mean(packaged, "temperature", 78, 90, -180, 180) < 0.0
    # Salinity (Absolute: 0.47 % above practical): salty subtropics and Mediterranean, fresh Arctic and Baltic.
    atlantic = _box_mean(packaged, "salinity", 20, 30, -60, -30)
    pacific = _box_mean(packaged, "salinity", 20, 30, -170, -130)
    assert atlantic > 37.0 > pacific + 1.0
    assert _box_mean(packaged, "salinity", 33, 42, 0, 30) > 38.0  # Mediterranean
    assert _box_mean(packaged, "salinity", 75, 90, -180, 180) < 33.5  # Arctic surface
    assert _box_mean(packaged, "salinity", 54, 65, 10, 28) < 10.0  # Baltic
    # Warm at the surface, cold at depth, in every latitude band that has deep water.
    surface = packaged["temperature"].to_numpy()[0]
    deep = packaged["temperature"].to_numpy()[-1]
    both = np.isfinite(surface) & np.isfinite(deep)
    assert both.sum() > 500
    assert np.all(surface[both] >= deep[both] - 1.0)
    assert -1.0 < np.nanmin(deep) and np.nanmax(deep) < 3.0


def test_packaged_state_is_statically_stable_enough_to_start_from(packaged):
    """The inversions are a few hundred pairs, nearly all weak and polar, and the recorded counts are right."""
    temperature = packaged["temperature"].to_numpy().astype(float)
    salinity = packaged["salinity"].to_numpy().astype(float)
    result = static_stability(temperature, salinity)
    assert result["pairs"] == packaged.attrs["adjacent_wet_pairs"]
    assert result["inverted"] == packaged.attrs["adjacent_pairs_statically_inverted"]
    assert result["largest"] == pytest.approx(packaged.attrs["largest_inversion_kg_m3"], rel=1e-6)
    assert result["inverted"] < 0.02 * result["pairs"]
    inversion = result["inversion"]
    assert np.sum(np.nan_to_num(inversion) > 0.5) <= 2
    # Not a mixed-up column: the typical inversion is a few hundredths of a kg m-3.
    assert np.median(inversion[np.nan_to_num(inversion) > 0]) < 0.1


def test_the_packaged_file_is_small():
    assert Path(str(PACKAGED_FILE)).stat().st_size < 600_000
