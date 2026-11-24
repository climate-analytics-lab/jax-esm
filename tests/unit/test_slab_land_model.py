"""Tests for `jem.components.slab.slab_land_model`."""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import jcm.constants as jcm_constants
import numpy as np
import pytest
import xarray as xr
from jax.test_util import check_grads

from jem.base.coupler import Coupler
from jem.components.slab.base import (
    MASKED_SURFACE_TEMPERATURE,
    load_monthly_climatology,
    load_static_field,
)
from jem.components.slab.bucket_land_model import BucketLandModel, BucketLandParameters
from jem.components.slab.slab_land_model import SlabLandModel, SlabLandParameters
from tests.unit.slab_test_utils import (
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    TIMESTEP,
    coupling_time,
    make_grid,
    run_steps,
    time_axis,
    tree_signature,
    write_climatology,
)

#: The length of 2001, the year `coupling_time`'s start date falls in.
DAYS_PER_YEAR = 365.0

#: SPEEDY's names for the land climatologies, as jax-gcm packages them.
SPEEDY_VARNAMES = {
    "land_surface_temperature_varname_in_clim_file": "stl",
    "snow_depth_varname_in_clim_file": "snowc",
    "soil_water_availability_varname_in_clim_file": "soilw_am",
}

# A coast column of 0.3 land and two all-land columns, ocean to the west:
# partly-land cells are integrated, all-ocean cells are masked.
LAND_FRACTION = np.array([
    [0.0, 0.0, 0.0],
    [0.3, 0.3, 0.3],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
])
N_LON, N_LAT = LAND_FRACTION.shape


@pytest.fixture
def grid():
    return make_grid(LAND_FRACTION)


@pytest.fixture
def half_land_grid():
    """Return a 4x3 grid whose eastern half is land, so both mask branches run."""
    shape = (len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))
    fraction = np.zeros(shape)
    fraction[shape[0] // 2:] = 1.0
    return make_grid(fractional_mask=fraction)


@pytest.fixture
def x64():
    """Run a test in double precision and put the process-global flag back."""
    before = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", before)


def _land(grid):
    return np.asarray(grid.land_mask)


def _water_mass(state, params):
    """Soil water per unit area (kg m-2), summed over both layers."""
    return jcm_constants.rhow * jnp.sum(
        state.soil_water_content * params.soil_bucket_depth, axis=-1)


def _forced(carry, grid, precipitation=0.0, evaporation=0.0, total_heat_flux=0.0):
    def field(value):
        return jnp.broadcast_to(jnp.asarray(value, dtype=float), grid.shape)

    forcing = carry["forcing"].replace(
        precipitation=field(precipitation),
        evaporation=field(evaporation),
        total_heat_flux=field(total_heat_flux),
    )
    return dict(carry, forcing=forcing)


def _with_soil_water(carry, grid, surface, deep):
    water = jnp.where(
        jnp.asarray(_land(grid))[..., None], jnp.asarray([surface, deep], dtype=float), 0.0
    )
    return dict(carry, state=carry["state"].replace(soil_water_content=water))


def _write_fields(path, **fields):
    """Write ``{name: (time, lat, lon) | (lat, lon) array}`` on the test grid."""
    data_vars = {
        name: ((("time", "lat", "lon") if values.ndim == 3 else ("lat", "lon")), values)
        for name, values in fields.items()
    }
    xr.Dataset(
        data_vars=data_vars,
        coords={"time": np.arange(12), "lat": LATITUDE_DEGREES, "lon": LONGITUDE_DEGREES},
    ).to_netcdf(path)
    return str(path)


def _write_land_climatology(path, names=("stl", "snowc", "soilw_am")):
    """Write a 12-month land climatology on the test grid, by name."""
    n_lat, n_lon = len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)
    months = np.arange(12)
    temperature = 280.0 + 5.0 * np.cos(2 * np.pi * months / 12.0)
    _write_fields(
        path,
        **{
            names[0]: np.broadcast_to(temperature[:, None, None], (12, n_lat, n_lon)).copy(),
            names[1]: np.full((12, n_lat, n_lon), 30.0),
            names[2]: np.full((12, n_lat, n_lon), 0.25),
        },
    )
    return str(path), temperature


def _climatology_at(temperature, year_fraction):
    """Linear interpolation of 12 equally spaced monthly records, periodic."""
    position = (year_fraction % 1.0) * 12.0
    left = int(np.floor(position)) % 12
    weight = position - np.floor(position)
    return (1.0 - weight) * temperature[left] + weight * temperature[(left + 1) % 12]


# ---------------------------------------------------------------------------
# Surface temperature
# ---------------------------------------------------------------------------


def test_idealized_climatology_varies_with_latitude_only(half_land_grid):
    """The idealised climatology must vary with latitude, not longitude."""
    model = SlabLandModel(half_land_grid)

    temperature = model._idealized_land_temperature()

    n_lon, n_lat = half_land_grid.shape
    assert temperature.shape == (n_lon, n_lat, 12)
    profile = np.asarray(temperature[:, :, 0], dtype=np.float64)
    assert np.all(profile.std(axis=0) == 0.0)
    assert np.all(profile.std(axis=1) > 1.0)


def test_land_step_runs_and_is_finite(half_land_grid):
    """Three unforced steps stay finite and physically plausible."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()

    for step in range(3):
        carry, _ = model.step(carry, coupling_time(step))

    temperature = carry["state"].land_surface_temperature
    assert bool(jnp.all(jnp.isfinite(temperature)))
    assert 150.0 < float(jnp.min(temperature)) and float(jnp.max(temperature)) < 350.0


def test_a_partly_land_cell_is_integrated_and_an_all_ocean_cell_is_masked(grid):
    """``land_mask`` is ``fractional_mask > 0``: the 0.3 column is land."""
    model = SlabLandModel(grid)
    stepped, _ = model.step(
        _forced(model.initialize(), grid, total_heat_flux=-100.0, precipitation=1e-5),
        coupling_time(0),
    )
    temperature = np.asarray(stepped["state"].land_surface_temperature)
    water = np.asarray(stepped["state"].soil_water_content)
    vegetation = np.asarray(stepped["state"].vegetation_fraction)
    albedo = np.asarray(stepped["derived"].surface_albedo)

    np.testing.assert_allclose(temperature[0], MASKED_SURFACE_TEMPERATURE)
    assert np.all(temperature[1] != MASKED_SURFACE_TEMPERATURE)
    assert np.all(temperature[1] > MASKED_SURFACE_TEMPERATURE)
    for field in (water[0], vegetation[0], albedo[0],
                  np.asarray(stepped["derived"].soil_water_availability)[0]):
        assert np.all(field == 0.0)
    assert np.all(water[1] > 0.0) and np.all(vegetation[1] > 0.0) and np.all(albedo[1] > 0.0)


def test_a_one_metre_slab_warms_at_its_heat_capacity(half_land_grid):
    """100 W m-2 into 1 m of soil for one hour warms it by 100 * 3600 / 2.5e6 K."""
    model = SlabLandModel(half_land_grid)
    carry = _forced(model.initialize(), half_land_grid, total_heat_flux=-100.0)
    before = np.asarray(carry["state"].land_surface_temperature)
    stepped, _ = model.step(carry, coupling_time(0, dt=3600.0))

    land = _land(half_land_grid)
    warming = np.asarray(stepped["state"].land_surface_temperature)[land] - before[land]
    np.testing.assert_allclose(warming, 100.0 * 3600.0 / 2.5e6, rtol=1e-4)


def test_the_free_running_slab_conserves_energy(grid):
    """``C dT / dt = -total_heat_flux`` on land, with the soil/glacier blend in ``C``."""
    glacier = np.array([[0.0, 0.5, 1.0]] * N_LON)
    params = SlabLandParameters(glacier_fraction=jnp.asarray(glacier))
    model = SlabLandModel(grid, params)
    flux = jnp.asarray(np.linspace(-150.0, 90.0, N_LON * N_LAT).reshape(N_LON, N_LAT))
    carry = dict(model.initialize(), forcing=model.initialize()["forcing"].replace(
        total_heat_flux=flux))
    stepped, _ = model.step(carry, coupling_time(0))

    heat_capacity = ((1.0 - glacier) * params.soil_thermal_depth
                     * params.soil_volumetric_heat_capacity
                     + glacier * params.glacier_depth * params.glacier_volumetric_heat_capacity)
    change = (np.asarray(stepped["state"].land_surface_temperature)
              - np.asarray(carry["state"].land_surface_temperature))
    land = _land(grid)
    np.testing.assert_allclose(
        (heat_capacity * change / TIMESTEP)[land], -np.asarray(flux)[land], rtol=2e-4)


def test_climatology_is_loaded_under_the_names_it_is_given(tmp_path, half_land_grid):
    """SPEEDY's file names reach the model through the *_varname arguments."""
    path, temperature = _write_land_climatology(tmp_path / "land.nc")
    model = SlabLandModel(
        half_land_grid,
        land_clim_file=path,
        snow_depth_from_clim_file=True,
        soil_water_availability_from_clim_file=True,
        **SPEEDY_VARNAMES,
    )

    assert model.land_surface_temperature_climatology.shape == half_land_grid.shape + (12,)
    np.testing.assert_allclose(
        np.asarray(model.land_surface_temperature_climatology)[0, 0, :], temperature, rtol=1e-5
    )

    stepped, _ = model.step(model.initialize(), coupling_time(0))
    land = _land(half_land_grid)
    # Snow depth 30 mm against the 60 mm saturation scale: half cover.
    np.testing.assert_allclose(np.asarray(stepped["state"].snow_cover)[land], 0.5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(stepped["state"].snow_depth)[land], 30.0, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(stepped["derived"].soil_water_availability)[land], 0.25, rtol=1e-6
    )
    assert float(np.abs(np.asarray(stepped["derived"].soil_water_availability)[~land]).max()) == 0.0
    assert float(np.abs(np.asarray(stepped["state"].snow_depth)[~land]).max()) == 0.0


def test_climatology_on_the_wrong_grid_is_rejected(tmp_path, half_land_grid):
    """A file on a shifted latitude axis is rejected, naming the file."""
    path = write_climatology(
        tmp_path / "land_shifted.nc",
        "stl",
        np.full((12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)), 280.0),
        latitude_degrees=LATITUDE_DEGREES + 1.0,
    )

    with pytest.raises(ValueError, match="latitude") as excinfo:
        SlabLandModel(half_land_grid, land_clim_file=path, **SPEEDY_VARNAMES)

    assert path in str(excinfo.value)


def test_a_missing_climatology_file_is_refused(tmp_path, half_land_grid):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SlabLandModel(half_land_grid, land_clim_file=str(tmp_path / "absent.nc"))


def test_a_variable_the_file_lacks_falls_back_with_a_warning(tmp_path, half_land_grid, caplog):
    """Only the temperature is in the file: the rest is idealized, and each
    field the model was told to read from the file says so loudly.
    """
    path = write_climatology(
        tmp_path / "temperature_only.nc", "stl",
        np.full((12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)), 281.0))
    with caplog.at_level(logging.WARNING, logger="jem.components.slab.slab_land_model"):
        SlabLandModel(half_land_grid, land_clim_file=path, **SPEEDY_VARNAMES)
    # A field nothing reads from the file is not worth a warning.
    assert not caplog.records

    with caplog.at_level(logging.WARNING, logger="jem.components.slab.slab_land_model"):
        model = SlabLandModel(
            half_land_grid, land_clim_file=path, **SPEEDY_VARNAMES,
            snow_depth_from_clim_file=True,
            soil_water_availability_from_clim_file=True,
            surface_albedo_from_clim_file=True,
            vegetation_fraction_from_clim_file=True,
        )

    shape = half_land_grid.shape + (12,)
    np.testing.assert_allclose(np.asarray(model.land_surface_temperature_climatology), 281.0)
    np.testing.assert_array_equal(np.asarray(model.snow_depth_climatology), np.zeros(shape))
    np.testing.assert_allclose(np.asarray(model.soil_water_availability_climatology), 0.5)
    np.testing.assert_allclose(np.asarray(model.surface_albedo_climatology), 0.2)
    np.testing.assert_allclose(np.asarray(model.vegetation_fraction_climatology), 0.8)
    for variable in ("snowc", "soilw_am", "surface_albedo", "vegetation_fraction"):
        assert any(variable in record.getMessage() for record in caplog.records), variable


def test_without_a_file_the_climatologies_are_idealized(half_land_grid):
    model = SlabLandModel(half_land_grid)
    shape = half_land_grid.shape + (12,)
    assert model.land_surface_temperature_climatology.shape == shape
    np.testing.assert_array_equal(np.asarray(model.snow_depth_climatology), np.zeros(shape))
    np.testing.assert_allclose(np.asarray(model.surface_albedo_climatology), 0.2)


def test_a_climatology_without_a_time_dimension_is_the_same_map_every_month(
        tmp_path, half_land_grid):
    """A static map used wherever a climatology is expected."""
    fraction = np.broadcast_to(np.array([0.1, 0.5, 0.9])[:, None], (3, 4)).copy()
    path = _write_fields(tmp_path / "vegetation.nc", vegetation_fraction=fraction)
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, vegetation_fraction_from_clim_file=True)

    climatology = np.asarray(model.vegetation_fraction_climatology)
    assert climatology.shape == half_land_grid.shape + (12,)
    np.testing.assert_allclose(climatology, np.broadcast_to(fraction.T[..., None], climatology.shape))
    stepped, _ = model.step(model.initialize(), coupling_time(200))
    land = _land(half_land_grid)
    np.testing.assert_allclose(
        np.asarray(stepped["state"].vegetation_fraction)[land], fraction.T[land], rtol=1e-6)


def test_static_fields_are_read_on_the_grid(tmp_path, half_land_grid):
    path = _write_fields(
        tmp_path / "static.nc",
        glacier_fraction=np.full((3, 4), 0.25),
        monthly=np.full((12, 3, 4), 0.25),
    )
    field = load_static_field(path, "glacier_fraction", half_land_grid)
    assert field.shape == half_land_grid.shape
    np.testing.assert_allclose(field, 0.25)
    with pytest.raises(ValueError, match="constant in time"):
        load_static_field(path, "monthly", half_land_grid)
    # A static field is a monthly climatology that never changes.
    assert load_monthly_climatology(path, "glacier_fraction", half_land_grid).shape == (4, 3, 12)


def test_the_glacier_map_comes_from_the_file_and_sets_the_heat_capacity(
        tmp_path, half_land_grid):
    """NaN over the ocean is allowed; the map is zero there and a pytree leaf."""
    glacier = np.full((3, 4), np.nan)
    glacier[:, 2:] = np.array([0.0, 0.5, 1.0])[:, None]
    path = _write_fields(tmp_path / "glacier.nc", glacier_fraction=glacier)
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, glacier_fraction_from_clim_file=True)

    fraction = np.asarray(model.params.glacier_fraction)
    assert fraction.shape == half_land_grid.shape
    np.testing.assert_allclose(fraction[2:], glacier[:, 2:].T)
    assert np.all(fraction[:2] == 0.0)

    carry = _forced(model.initialize(), half_land_grid, total_heat_flux=-100.0)
    stepped, _ = model.step(carry, coupling_time(0))
    warming = (np.asarray(stepped["state"].land_surface_temperature)
               - np.asarray(carry["state"].land_surface_temperature))[2:]
    soil, ice = 1.0 * 2.5e6, 5.0 * 1.93e6
    capacity = (1.0 - fraction[2:]) * soil + fraction[2:] * ice
    np.testing.assert_allclose(warming, 100.0 * TIMESTEP / capacity, rtol=1e-4)


def test_the_glacier_map_can_be_derived_from_the_background_albedo(
        tmp_path, half_land_grid):
    """SPEEDY's test for land ice: a land cell whose albedo is at or above
    the threshold is all glacier, the rest all soil, and the ocean none. The
    map is the parameter the heat capacity reads, so bright land warms more
    slowly under the same flux.
    """
    albedo = np.full((3, 4), np.nan)
    albedo[:, 2:] = np.array([0.15, 0.4, 0.8])[:, None]
    path = _write_fields(tmp_path / "boundary.nc", alb=albedo)
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, glacier_fraction_from_albedo=True,
        surface_albedo_varname_in_clim_file="alb")

    fraction = np.asarray(model.params.glacier_fraction)
    assert fraction.shape == half_land_grid.shape
    np.testing.assert_array_equal(
        fraction[2:], np.broadcast_to(np.array([0.0, 1.0, 1.0]), fraction[2:].shape))
    assert np.all(fraction[:2] == 0.0)

    carry = _forced(model.initialize(), half_land_grid, total_heat_flux=-100.0)
    stepped, _ = model.step(carry, coupling_time(0))
    warming = (np.asarray(stepped["state"].land_surface_temperature)
               - np.asarray(carry["state"].land_surface_temperature))[2:]
    capacity = np.where(fraction[2:] == 1.0, 5.0 * 1.93e6, 1.0 * 2.5e6)
    np.testing.assert_allclose(warming, 100.0 * TIMESTEP / capacity, rtol=1e-4)

    # The threshold is the parameter's, read when the model is built.
    stricter = SlabLandModel(
        half_land_grid, SlabLandParameters(glacier_albedo_threshold=0.5),
        land_clim_file=path, glacier_fraction_from_albedo=True,
        surface_albedo_varname_in_clim_file="alb")
    np.testing.assert_array_equal(
        np.asarray(stricter.params.glacier_fraction)[2:],
        np.broadcast_to(np.array([0.0, 0.0, 1.0]), fraction[2:].shape))


def test_a_monthly_albedo_is_averaged_over_the_year_for_the_glacier_map(
        tmp_path, half_land_grid):
    """A cell bright for half the year averages 0.35: below the threshold."""
    albedo = np.full((12, 3, 4), 0.2)
    albedo[:6, 0, :] = 0.5
    albedo[:, 1, :] = 0.6
    path = _write_fields(tmp_path / "boundary.nc", alb=albedo)
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, glacier_fraction_from_albedo=True,
        surface_albedo_varname_in_clim_file="alb")
    np.testing.assert_array_equal(
        np.asarray(model.params.glacier_fraction)[2:],
        np.broadcast_to(np.array([0.0, 1.0, 0.0]), (2, 3)))


def test_a_glacier_map_from_albedo_needs_the_albedo(tmp_path, half_land_grid):
    with pytest.raises(ValueError, match="needs a land_clim_file"):
        SlabLandModel(half_land_grid, glacier_fraction_from_albedo=True)
    path = _write_fields(tmp_path / "boundary.nc", glacier_fraction=np.full((3, 4), 0.2))
    with pytest.raises(ValueError, match="does not have"):
        SlabLandModel(half_land_grid, land_clim_file=path, glacier_fraction_from_albedo=True)
    with pytest.raises(ValueError, match="choose one"):
        SlabLandModel(
            half_land_grid, land_clim_file=path, glacier_fraction_from_albedo=True,
            glacier_fraction_from_clim_file=True)
    bad = np.full((3, 4), 0.2)
    bad[0, 3] = np.nan
    path = _write_fields(tmp_path / "bad.nc", alb=bad)
    with pytest.raises(ValueError, match="non-finite"):
        SlabLandModel(
            half_land_grid, land_clim_file=path, glacier_fraction_from_albedo=True,
            surface_albedo_varname_in_clim_file="alb")


def test_a_glacier_map_with_nan_over_land_is_refused(tmp_path, half_land_grid):
    glacier = np.full((3, 4), 0.2)
    glacier[0, 3] = np.nan
    path = _write_fields(tmp_path / "glacier.nc", glacier_fraction=glacier)
    with pytest.raises(ValueError, match="non-finite"):
        SlabLandModel(half_land_grid, land_clim_file=path, glacier_fraction_from_clim_file=True)


def test_a_used_climatology_with_nan_over_land_is_refused(tmp_path, half_land_grid):
    values = np.full((12, 3, 4), 0.4)
    values[:, 0, 3] = np.nan
    path = _write_fields(tmp_path / "vegetation.nc", vegetation_fraction=values)
    with pytest.raises(ValueError, match="NaNs over land"):
        SlabLandModel(half_land_grid, land_clim_file=path, vegetation_fraction_from_clim_file=True)
    # Not read by this model, so not its business.
    SlabLandModel(half_land_grid, land_clim_file=path)


def test_relaxation_follows_the_climatology(tmp_path, half_land_grid):
    """Unforced, a relaxed land started on its climatology damps the drift of it."""
    path, temperature = _write_land_climatology(tmp_path / "land.nc")
    model = SlabLandModel(
        half_land_grid,
        land_clim_file=path,
        land_surface_temperature_from_clim_file=True,
        **SPEEDY_VARNAMES,
    )
    carry = model.initialize(initialize_from_clim_file=True)
    assert "sim_time" not in carry["state"].asdict()

    land = _land(half_land_grid)
    timescale_steps = 40.0 * 86400.0 / TIMESTEP
    dissipation = timescale_steps / (1.0 + timescale_steps)
    for step in (0, 182):
        stepped, _ = model.step(carry, coupling_time(step))
        year_fraction = step * TIMESTEP / (86400.0 * DAYS_PER_YEAR)
        anomaly = _climatology_at(temperature, 0.0) - _climatology_at(temperature, year_fraction)
        expected = (dissipation * anomaly
                    + _climatology_at(temperature, year_fraction + 1.0 / DAYS_PER_YEAR))
        np.testing.assert_allclose(
            np.asarray(stepped["state"].land_surface_temperature)[land], expected, rtol=1e-5
        )


def test_relaxation_with_forcing_reproduces_the_closed_form_over_two_steps(
        tmp_path, half_land_grid):
    """``T'_new = tau / (tau + dt) (T' + dt F / C)``, the climatology added back at the end."""
    path, temperature = _write_land_climatology(tmp_path / "land.nc")
    model = SlabLandModel(
        half_land_grid, land_clim_file=path,
        land_surface_temperature_from_clim_file=True, **SPEEDY_VARNAMES)
    flux = 60.0                       # W m-2 upward: cools the land
    carry = _forced(model.initialize(), half_land_grid, total_heat_flux=flux)
    # Start off the climatology so the anomaly is not zero.
    start = jnp.where(jnp.asarray(_land(half_land_grid)), 285.0, MASKED_SURFACE_TEMPERATURE)
    carry = dict(carry, state=carry["state"].replace(land_surface_temperature=start))

    heat_capacity = 1.0 * 2.5e6
    tau = 40.0 * 86400.0
    damping = tau / (tau + TIMESTEP)
    expected = 285.0
    land = _land(half_land_grid)
    # Two steps, 180 days in: a climatology that moves between the steps.
    first_step = 180
    for index in range(2):
        step = first_step + index
        year_begin = step * TIMESTEP / (86400.0 * DAYS_PER_YEAR)
        year_end = (step + 1) * TIMESTEP / (86400.0 * DAYS_PER_YEAR)
        carry, _ = model.step(carry, coupling_time(step))
        anomaly = expected - _climatology_at(temperature, year_begin)
        expected = (damping * (anomaly - TIMESTEP * flux / heat_capacity)
                    + _climatology_at(temperature, year_end))
        np.testing.assert_allclose(
            np.asarray(carry["state"].land_surface_temperature)[land], expected, rtol=1e-5)


def test_bind_sets_the_initial_climatology_month(half_land_grid):
    """The start date reaches `initialize()` through the coupler, not a parameter."""

    def initial_land_temperature(start_date):
        model = SlabLandModel(half_land_grid)
        Coupler(
            {"lnd": model},
            coupling_timestep=jdt.to_timedelta(1, "day"),
            start_date=jdt.to_datetime(start_date),
        )
        assert model.start_year_fraction == pytest.approx(
            0.0 if start_date.endswith("01-01") else 181.0 / DAYS_PER_YEAR
        )
        temperature = np.asarray(
            model.initialize(initialize_from_clim_file=True)["state"].land_surface_temperature
        )
        return temperature[_land(half_land_grid)]

    january = initial_land_temperature("2001-01-01")
    july = initial_land_temperature("2001-07-01")
    assert np.max(np.abs(july - january)) > 1.0


def test_rebinding_to_a_different_clock_is_rejected(half_land_grid):
    """One instance belongs to one coupled model; a conflicting second bind raises."""
    model = SlabLandModel(half_land_grid)
    clock = {"coupling_timestep": jdt.to_timedelta(1, "day")}
    Coupler({"lnd": model}, start_date=jdt.to_datetime("2001-01-01"), **clock)
    Coupler({"lnd": model}, start_date=jdt.to_datetime("2001-01-01"), **clock)
    with pytest.raises(ValueError, match="already bound"):
        Coupler({"lnd": model}, start_date=jdt.to_datetime("2001-07-01"), **clock)
    assert model.start_year_fraction == 0.0


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------


def test_the_constructor_and_initialize_both_choose_the_climatology_start(
        tmp_path, half_land_grid):
    path, temperature = _write_land_climatology(tmp_path / "land.nc")
    land = _land(half_land_grid)

    def start(model, **kwargs):
        return np.asarray(model.initialize(**kwargs)["state"].land_surface_temperature)[land]

    plain = SlabLandModel(half_land_grid, land_clim_file=path, **SPEEDY_VARNAMES)
    from_file = SlabLandModel(
        half_land_grid, land_clim_file=path, initialize_from_clim_file=True, **SPEEDY_VARNAMES)

    np.testing.assert_allclose(start(plain), 288.15)
    np.testing.assert_allclose(start(from_file), temperature[0], rtol=1e-6)
    np.testing.assert_allclose(start(plain, initialize_from_clim_file=True), temperature[0], rtol=1e-6)
    np.testing.assert_allclose(start(from_file, initialize_from_clim_file=False), 288.15)


def test_soil_water_starts_at_the_fill_between_wilting_point_and_capacity(half_land_grid):
    model = SlabLandModel(half_land_grid)
    params = model.params
    water = np.asarray(model.initialize()["state"].soil_water_content)
    land = _land(half_land_grid)

    wilting = np.asarray(params.soil_water_wilting_point)
    capacity = np.asarray(params.soil_water_field_capacity)
    np.testing.assert_allclose(
        water[land], np.broadcast_to(wilting + 0.5 * (capacity - wilting), water[land].shape),
        rtol=1e-6)
    assert float(np.abs(water[~land]).max()) == 0.0

    drier = model.initialize(params.replace(initial_soil_water_fill=0.0))
    np.testing.assert_allclose(
        np.asarray(drier["state"].soil_water_content)[land],
        np.broadcast_to(wilting, water[land].shape))


def test_the_soil_starts_from_the_climatologys_availability_like_the_bucket(
        tmp_path, half_land_grid):
    availability = np.full((12, 3, 4), 0.3)
    path = _write_fields(tmp_path / "soil.nc", soilw_am=availability)
    land = _land(half_land_grid)

    slab = SlabLandModel(
        half_land_grid, land_clim_file=path, initialize_from_clim_file=True,
        soil_water_availability_varname_in_clim_file="soilw_am")
    carry = slab.initialize()
    np.testing.assert_allclose(
        np.asarray(carry["derived"].soil_water_availability)[land], 0.3, rtol=1e-5)

    bucket = BucketLandModel(half_land_grid, soil_water_clim_file=path).initialize()
    np.testing.assert_array_equal(
        np.asarray(carry["state"].soil_water_content),
        np.asarray(bucket["state"].soil_water_content))

    # Without the flag the file is not read for the initial condition.
    plain = SlabLandModel(
        half_land_grid, land_clim_file=path,
        soil_water_availability_varname_in_clim_file="soilw_am").initialize()
    assert not np.allclose(np.asarray(plain["derived"].soil_water_availability)[land], 0.3)


def test_a_file_without_the_availability_starts_the_soil_at_the_fill(tmp_path, half_land_grid):
    path = write_climatology(
        tmp_path / "temperature.nc", "stl",
        np.full((12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)), 281.0))
    from_file = SlabLandModel(
        half_land_grid, land_clim_file=path, initialize_from_clim_file=True, **SPEEDY_VARNAMES)
    plain = SlabLandModel(half_land_grid)
    np.testing.assert_array_equal(
        np.asarray(from_file.initialize()["state"].soil_water_content),
        np.asarray(plain.initialize()["state"].soil_water_content))


# ---------------------------------------------------------------------------
# Snow
# ---------------------------------------------------------------------------


def test_snow_is_held_without_a_climatology(half_land_grid):
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    carry = dict(carry, state=carry["state"].replace(snow_depth=jnp.full(half_land_grid.shape, 90.0)))
    stepped, _ = model.step(carry, coupling_time(0))
    np.testing.assert_allclose(np.asarray(stepped["state"].snow_depth), 90.0)
    np.testing.assert_allclose(np.asarray(stepped["state"].snow_cover), 1.0)


def test_snow_cover_units_in_the_file_are_depth_over_the_scale(tmp_path, half_land_grid):
    """A file value v is a depth ``v * scale`` mm and a cover ``min(1, v)``."""
    snow = np.broadcast_to(np.array([0.25, 2.0, 0.5])[None, :, None], (12, 3, 4)).copy()
    path = _write_fields(tmp_path / "snow.nc", snowc=snow)
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, snow_depth_from_clim_file=True,
        initialize_from_clim_file=True, snow_depth_varname_in_clim_file="snowc",
        snow_depth_units_in_clim_file="cover")
    land = _land(half_land_grid)
    values = np.broadcast_to(np.array([0.25, 2.0, 0.5])[None, :], half_land_grid.shape)

    def check(state, scale):
        np.testing.assert_allclose(
            np.asarray(state.snow_depth)[land], (values * scale)[land], rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(state.snow_cover)[land], np.minimum(1.0, values)[land], rtol=1e-6)

    initial = model.initialize()
    check(initial["state"], 60.0)
    stepped, _ = model.step(initial, coupling_time(0))
    check(stepped["state"], 60.0)

    # The scale is a parameter: the conversion follows the carried value.
    rescaled = dict(initial, params=initial["params"].replace(snow_depth_to_cover_scale=30.0))
    stepped, _ = model.step(rescaled, coupling_time(0))
    check(stepped["state"], 30.0)
    check(model.initialize(initial["params"].replace(snow_depth_to_cover_scale=30.0))["state"], 30.0)


def test_snow_depth_in_mm_is_not_rescaled(tmp_path, half_land_grid):
    path = _write_fields(tmp_path / "snow.nc", snowc=np.full((12, 3, 4), 45.0))
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, snow_depth_from_clim_file=True,
        snow_depth_varname_in_clim_file="snowc")
    stepped, _ = model.step(model.initialize(), coupling_time(0))
    land = _land(half_land_grid)
    np.testing.assert_allclose(np.asarray(stepped["state"].snow_depth)[land], 45.0)
    np.testing.assert_allclose(np.asarray(stepped["state"].snow_cover)[land], 0.75)


def test_other_snow_depth_units_are_refused(half_land_grid):
    with pytest.raises(ValueError, match="snow_depth_units_in_clim_file"):
        SlabLandModel(half_land_grid, snow_depth_units_in_clim_file="m")


# ---------------------------------------------------------------------------
# Soil water: the bucket's, in a land with its own vegetation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vegetation", [0.8, 0.3])
def test_the_soil_water_of_a_step_is_the_buckets(grid, vegetation):
    """Same water, forcing, parameters and vegetation: the same soil water, bit for bit."""
    bucket_params = BucketLandParameters(vegetation_fraction=vegetation)
    slab_params = SlabLandParameters(vegetation_fraction=vegetation)
    bucket = BucketLandModel(grid, bucket_params)
    slab = SlabLandModel(grid, slab_params)
    water = jnp.where(
        jnp.asarray(_land(grid))[..., None], jnp.asarray([0.21, 0.24], dtype=float), 0.0)

    def stepped(model, carry):
        carry = dict(carry, state=carry["state"].replace(soil_water_content=water))
        shape = grid.shape
        carry = dict(carry, forcing=carry["forcing"].replace(
            precipitation=jnp.full(shape, 3e-5), evaporation=jnp.full(shape, 2e-5)))
        return model.step(carry, coupling_time(0))[0]

    from_bucket = stepped(bucket, bucket.initialize())
    from_slab = stepped(slab, slab.initialize())
    np.testing.assert_array_equal(
        np.asarray(from_slab["state"].soil_water_content),
        np.asarray(from_bucket["state"].soil_water_content))
    for name in ("soil_water_availability", "deep_drainage", "soil_water_overflow",
                 "runoff", "unmet_evaporation", "runoff_mass_rate"):
        np.testing.assert_array_equal(
            np.asarray(getattr(from_slab["derived"], name)),
            np.asarray(getattr(from_bucket["derived"], name)), err_msg=name)


def test_the_derived_fields_include_the_buckets(grid):
    from jem.components.slab.bucket_land_model import BucketLandDerived

    carry = SlabLandModel(grid).initialize()
    bucket_fields = set(BucketLandDerived.zeros(grid.shape).asdict())
    assert bucket_fields <= set(carry["derived"].asdict())
    assert set(carry["derived"].asdict()) - bucket_fields == {"surface_albedo"}


def test_the_vegetation_of_the_state_weights_the_deep_layer(half_land_grid):
    """A state's vegetation fraction, not a parameter, is what the bucket sees."""
    model = SlabLandModel(half_land_grid)
    carry = _with_soil_water(model.initialize(), half_land_grid, 0.20, 0.25)
    bare = dict(carry, state=carry["state"].replace(
        vegetation_fraction=jnp.zeros(half_land_grid.shape)))
    land = _land(half_land_grid)
    leafy, _ = model.step(carry, coupling_time(0))
    barren, _ = model.step(bare, coupling_time(0))
    assert np.all(np.asarray(leafy["derived"].soil_water_availability)[land]
                  > np.asarray(barren["derived"].soil_water_availability)[land])


@pytest.mark.parametrize(
    "surface, deep, precipitation, evaporation",
    [
        (0.25, 0.28, 0.0, 3e-5),    # both layers wet
        (0.00, 0.28, 0.0, 3e-5),    # dry surface, wet deep layer
        (0.01, 0.18, 0.0, 1e-4),    # demand exceeds supply
        (0.28, 0.33, 5e-4, 1e-5),   # heavy rain over a full bucket
        (0.10, 0.20, 1e-5, -1e-5),  # dew
    ],
)
def test_the_water_budget_closes_per_unit_land_area(
        grid, surface, deep, precipitation, evaporation):
    """P - (E - unmet) - drainage - overflow is exactly the change in soil water."""
    model = SlabLandModel(grid)
    carry = _forced(
        _with_soil_water(model.initialize(), grid, surface, deep),
        grid, precipitation=precipitation, evaporation=evaporation,
    )
    stepped, _ = model.step(carry, coupling_time(0))
    derived = stepped["derived"]

    params = model.params
    change = (_water_mass(stepped["state"], params) - _water_mass(carry["state"], params)) / TIMESTEP
    net = (
        precipitation
        - (evaporation - derived.unmet_evaporation)
        - derived.deep_drainage
        - derived.soil_water_overflow
    )
    land = _land(grid)
    np.testing.assert_allclose(np.asarray(change)[land], np.asarray(net)[land], atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(derived.runoff), np.asarray(derived.deep_drainage + derived.soil_water_overflow))


def test_no_layer_leaves_the_range_between_wilting_point_and_field_capacity(half_land_grid):
    """Heavy demand stops at the wilting point; heavy rain stops at field capacity."""
    model = SlabLandModel(half_land_grid)
    params = model.params
    land = _land(half_land_grid)

    dry, _ = model.step(
        _forced(_with_soil_water(model.initialize(), half_land_grid, 0.01, 0.18),
                half_land_grid, evaporation=1e-4),
        coupling_time(0),
    )
    water = np.asarray(dry["state"].soil_water_content)[land]
    assert np.all(water >= np.asarray(params.soil_water_wilting_point) - 1e-7)
    assert float(np.asarray(dry["derived"].unmet_evaporation)[land].min()) > 0.0

    wet, _ = model.step(
        _forced(model.initialize(), half_land_grid, precipitation=1e-3), coupling_time(0)
    )
    water = np.asarray(wet["state"].soil_water_content)[land]
    assert np.all(water <= np.asarray(params.soil_water_field_capacity) + 1e-7)
    assert float(np.asarray(wet["derived"].soil_water_overflow)[land].min()) > 0.0


def test_the_runoff_mass_rate_sums_runoff_over_the_land_area(grid):
    model = SlabLandModel(grid)
    carry = _forced(model.initialize(), grid, precipitation=1e-3)
    carry, _ = run_steps(model, carry, 2)
    derived = carry["derived"]
    expected = np.sum(np.asarray(grid.cell_solid_angle) * jcm_constants.rearth ** 2
                      * LAND_FRACTION * np.asarray(derived.runoff))
    assert float(derived.runoff_mass_rate) > 0.0
    np.testing.assert_allclose(float(derived.runoff_mass_rate), expected, rtol=1e-5)


def test_soil_water_availability_is_speedys(half_land_grid):
    """The availability is the vegetation-weighted water above the wilting point."""
    params = SlabLandParameters(drainage_timescale=jnp.asarray((1e15, 1e15)))
    model = SlabLandModel(half_land_grid, params)
    stepped, _ = model.step(
        _with_soil_water(model.initialize(), half_land_grid, 0.20, 0.25), coupling_time(0)
    )

    expected = (0.07 * 0.20 + 0.8 * 0.21 * (0.25 - 0.17)) / (0.07 * 0.30 + 0.21 * (0.30 - 0.17))
    land = _land(half_land_grid)
    availability = np.asarray(stepped["derived"].soil_water_availability)
    np.testing.assert_allclose(availability[land], expected, rtol=1e-5)
    assert float(np.abs(availability[~land]).max()) == 0.0


def test_a_prescribed_availability_does_not_feed_back_on_the_soil(tmp_path, half_land_grid):
    path = _write_fields(tmp_path / "soil.nc", soilw_am=np.full((12, 3, 4), 0.9))
    kwargs = dict(soil_water_availability_varname_in_clim_file="soilw_am")
    prescribed = SlabLandModel(
        half_land_grid, land_clim_file=path, soil_water_availability_from_clim_file=True, **kwargs)
    diagnosed = SlabLandModel(half_land_grid)
    land = _land(half_land_grid)

    def run(model):
        carry = _forced(model.initialize(), half_land_grid, precipitation=1e-5, evaporation=2e-5)
        return model.step(carry, coupling_time(0))[0]

    left, right = run(prescribed), run(diagnosed)
    np.testing.assert_array_equal(
        np.asarray(left["state"].soil_water_content), np.asarray(right["state"].soil_water_content))
    np.testing.assert_allclose(np.asarray(left["derived"].soil_water_availability)[land], 0.9)
    assert not np.allclose(np.asarray(right["derived"].soil_water_availability)[land], 0.9)


# ---------------------------------------------------------------------------
# Vegetation and albedo
# ---------------------------------------------------------------------------


def test_vegetation_fraction_starts_from_the_parameter(half_land_grid):
    model = SlabLandModel(half_land_grid, SlabLandParameters(vegetation_fraction=0.35))
    land = _land(half_land_grid)
    vegetation = np.asarray(model.initialize()["state"].vegetation_fraction)
    np.testing.assert_allclose(vegetation[land], 0.35, rtol=1e-6)
    assert float(np.abs(vegetation[~land]).max()) == 0.0


def test_vegetation_fraction_is_read_from_the_climatology(tmp_path, half_land_grid):
    """A prescribed vegetation fraction reaches the albedo and the availability."""
    n_lat, n_lon = len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)
    values = np.full((12, n_lat, n_lon), 0.25, dtype=np.float32)
    path = write_climatology(tmp_path / "vegetation.nc", "vegetation_fraction", values)
    model = SlabLandModel(half_land_grid, land_clim_file=path, vegetation_fraction_from_clim_file=True)

    initial = model.initialize()
    land = _land(half_land_grid)
    np.testing.assert_allclose(np.asarray(initial["state"].vegetation_fraction)[land], 0.25)
    stepped, _ = model.step(initial, coupling_time(0))
    np.testing.assert_allclose(np.asarray(stepped["state"].vegetation_fraction)[land], 0.25)
    np.testing.assert_allclose(
        np.asarray(stepped["derived"].surface_albedo)[land], 0.2 * 0.75 + 0.1 * 0.25, rtol=1e-6
    )


def test_albedo_follows_the_carried_params(half_land_grid):
    """The blend reads carry["params"], so the albedos stay differentiable."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    land = _land(half_land_grid)

    default, _ = model.step(carry, coupling_time(0))
    np.testing.assert_allclose(
        np.asarray(default["derived"].surface_albedo)[land], 0.2 * 0.2 + 0.1 * 0.8, rtol=1e-6
    )
    brighter, _ = model.step(
        dict(carry, params=carry["params"].replace(baresoil_albedo=0.4)), coupling_time(0)
    )
    np.testing.assert_allclose(
        np.asarray(brighter["derived"].surface_albedo)[land], 0.4 * 0.2 + 0.1 * 0.8, rtol=1e-6
    )
    assert float(np.abs(np.asarray(brighter["derived"].surface_albedo)[~land]).max()) == 0.0


def test_the_albedo_is_read_from_the_climatology(tmp_path, half_land_grid):
    path = _write_fields(tmp_path / "albedo.nc", surface_albedo=np.full((12, 3, 4), 0.45))
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, surface_albedo_from_clim_file=True)
    land = _land(half_land_grid)
    stepped, _ = model.step(model.initialize(), coupling_time(0))
    np.testing.assert_allclose(np.asarray(stepped["derived"].surface_albedo)[land], 0.45)
    np.testing.assert_allclose(
        np.asarray(model.initialize()["derived"].surface_albedo)[land], 0.45)


# ---------------------------------------------------------------------------
# Structure, parameters and gradients
# ---------------------------------------------------------------------------


def test_the_land_parameters_are_the_buckets_plus_the_thermal_ones():
    assert issubclass(SlabLandParameters, BucketLandParameters)
    bucket, slab = BucketLandParameters(), SlabLandParameters()
    for field in dataclasses.fields(BucketLandParameters):
        np.testing.assert_array_equal(
            np.asarray(getattr(slab, field.name)), np.asarray(getattr(bucket, field.name)))
    assert {f.name for f in dataclasses.fields(SlabLandParameters)} - {
        f.name for f in dataclasses.fields(BucketLandParameters)} == {
        "soil_thermal_depth", "glacier_depth", "soil_volumetric_heat_capacity",
        "glacier_volumetric_heat_capacity", "temperature_relaxation_timescale",
        "snow_depth_to_cover_scale", "baresoil_albedo", "vegetation_albedo",
        "glacier_fraction", "glacier_albedo_threshold"}
    # The inherited list-to-array conversion (Hydra hands a pair over as a list).
    assert isinstance(SlabLandParameters(drainage_timescale=[1.0, 2.0]).drainage_timescale, jax.Array)


def test_params_default_equivalence(half_land_grid):
    """Constructing with no params is constructing with the defaults."""
    implicit = SlabLandModel(half_land_grid).initialize()
    explicit = SlabLandModel(half_land_grid, SlabLandParameters.default()).initialize()

    assert jax.tree_util.tree_structure(implicit) == jax.tree_util.tree_structure(explicit)
    for left, right in zip(jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_initialize_takes_parameters_and_defaults_to_the_models_own(half_land_grid):
    """``initialize(params)`` uses them, and they shape the initial soil water."""
    model = SlabLandModel(half_land_grid)
    default = model.initialize()
    explicit = model.initialize(model.params)
    for left, right in zip(jax.tree_util.tree_leaves(default), jax.tree_util.tree_leaves(explicit)):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    drier = model.params.replace(initial_soil_water_fill=0.0, vegetation_fraction=0.6)
    carry = model.initialize(drier)
    land = _land(half_land_grid)
    np.testing.assert_allclose(
        np.asarray(carry["state"].soil_water_content)[land], [[0.0, 0.17]] * land.sum())
    np.testing.assert_allclose(np.asarray(carry["state"].vegetation_fraction)[land], 0.6)
    assert carry["params"] is drier


def test_step_shapes_and_dtypes_stable(half_land_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_every_climatology_mode_keeps_the_carry_structure(tmp_path, half_land_grid):
    """All the ``*_from_clim_file`` branches at once, from the file's start."""
    n_lat, n_lon = len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)
    path = _write_fields(
        tmp_path / "all.nc",
        land_surface_temperature=np.full((12, n_lat, n_lon), 281.0, dtype=np.float32),
        snow_depth=np.full((12, n_lat, n_lon), 10.0, dtype=np.float32),
        soil_water_availability=np.full((12, n_lat, n_lon), 0.4, dtype=np.float32),
        surface_albedo=np.full((12, n_lat, n_lon), 0.3, dtype=np.float32),
        vegetation_fraction=np.full((12, n_lat, n_lon), 0.6, dtype=np.float32),
        glacier_fraction=np.full((n_lat, n_lon), 0.1, dtype=np.float32),
    )
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, initialize_from_clim_file=True,
        soil_water_availability_from_clim_file=True, snow_depth_from_clim_file=True,
        land_surface_temperature_from_clim_file=True, surface_albedo_from_clim_file=True,
        glacier_fraction_from_clim_file=True, vegetation_fraction_from_clim_file=True)
    coupler = Coupler(
        {"lnd": model}, coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=jdt.to_datetime("2001-01-01"))
    carry = coupler.initialize()
    final, _ = jax.jit(coupler.generate_trajectory_function(2))(carry)
    assert tree_signature(final.components["lnd"]) == tree_signature(carry.components["lnd"])


def _two_coupled_steps(model):
    coupler = Coupler(
        {"lnd": model},
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=jdt.to_datetime("2001-01-01"),
    )
    carry = coupler.initialize()
    final, trajectory = jax.jit(coupler.generate_trajectory_function(2))(carry)
    return carry, final, trajectory


def test_two_coupled_steps_keep_the_carry(half_land_grid):
    """The structure check that only lax.scan makes, in single precision."""
    carry, final, _ = _two_coupled_steps(SlabLandModel(half_land_grid))
    assert int(final.step) == 2
    assert tree_signature(final.components["lnd"]) == tree_signature(carry.components["lnd"])
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(final.components["lnd"])} == {
        np.dtype("float32")}


def test_two_coupled_steps_keep_the_carry_under_x64(x64, half_land_grid):
    """And in double precision: the carry is float64 throughout, so scan agrees."""
    carry, final, _ = _two_coupled_steps(SlabLandModel(half_land_grid))
    assert tree_signature(final.components["lnd"]) == tree_signature(carry.components["lnd"])
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(final.components["lnd"])} == {
        np.dtype("float64")}


def test_a_climatology_file_in_float32_gives_a_float64_carry_under_x64(
        x64, tmp_path, half_land_grid):
    n_lat, n_lon = len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)
    path = _write_fields(
        tmp_path / "land.nc",
        stl=np.full((12, n_lat, n_lon), 281.0, dtype=np.float32),
        snowc=np.full((12, n_lat, n_lon), 10.0, dtype=np.float32),
        soilw_am=np.full((12, n_lat, n_lon), 0.4, dtype=np.float32))
    model = SlabLandModel(
        half_land_grid, land_clim_file=path, initialize_from_clim_file=True,
        land_surface_temperature_from_clim_file=True, snow_depth_from_clim_file=True,
        soil_water_availability_from_clim_file=True, **SPEEDY_VARNAMES)
    carry, final, _ = _two_coupled_steps(model)
    assert tree_signature(final.components["lnd"]) == tree_signature(carry.components["lnd"])
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(final.components["lnd"])} == {
        np.dtype("float64")}


def test_step_is_differentiable(half_land_grid):
    """Reverse-mode gradients of one step agree with finite differences."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    land = jnp.asarray(_land(half_land_grid))

    def mean_land_temperature(heat_flux_scale):
        stepped, _ = model.step(
            _forced(carry, half_land_grid, total_heat_flux=1.0e3 * heat_flux_scale), coupling_time(0)
        )
        return jnp.mean(jnp.where(land, stepped["state"].land_surface_temperature, 0.0))

    check_grads(mean_land_temperature, (0.0,), order=1, modes=["rev"], eps=1e-2, atol=1e-2, rtol=1e-2)


def test_the_bucket_is_differentiable(half_land_grid):
    """Availability and runoff have finite, non-zero gradients in the forcing and the depths."""
    model = SlabLandModel(half_land_grid)
    start = _with_soil_water(model.initialize(), half_land_grid, 0.2, 0.25)

    def availability(precipitation, bucket_depth):
        carry = _forced(start, half_land_grid, precipitation=precipitation, evaporation=1e-5)
        carry = dict(carry, params=carry["params"].replace(soil_bucket_depth=bucket_depth))
        stepped, _ = model.step(carry, coupling_time(0))
        return jnp.sum(stepped["derived"].soil_water_availability)

    gradients = jax.grad(availability, argnums=(0, 1))(
        jnp.asarray(2e-5), model.params.soil_bucket_depth
    )
    for gradient in jax.tree_util.tree_leaves(gradients):
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_a_two_step_result_is_differentiable_in_thermal_bucket_and_glacier_parameters(grid):
    """The thermal depth, a bucket parameter and a glacier-fraction *map* all have
    finite gradients, non-zero on land and exactly zero where the model is masked.
    """
    model = SlabLandModel(grid)
    land = jnp.asarray(_land(grid))
    start = _with_soil_water(model.initialize(), grid, 0.2, 0.25)

    def two_steps(params):
        carry = _forced(dict(start, params=params), grid,
                        precipitation=2e-5, evaporation=1e-5, total_heat_flux=-40.0)
        for step in range(2):
            carry, _ = model.step(carry, coupling_time(step))
        return carry

    def mean_temperature(thermal_depth, glacier_fraction):
        params = start["params"].replace(
            soil_thermal_depth=thermal_depth, glacier_fraction=glacier_fraction)
        temperature = two_steps(params)["state"].land_surface_temperature
        return jnp.sum(jnp.where(land, temperature, 0.0))

    def total_availability(drainage_timescale):
        params = start["params"].replace(drainage_timescale=drainage_timescale)
        return jnp.sum(two_steps(params)["derived"].soil_water_availability)

    depth_grad, glacier_grad = jax.grad(mean_temperature, argnums=(0, 1))(
        jnp.asarray(1.0), jnp.zeros(grid.shape))
    assert bool(jnp.isfinite(depth_grad)) and float(depth_grad) < 0.0   # a deeper slab warms less
    glacier_grad = np.asarray(glacier_grad)
    assert glacier_grad.shape == grid.shape
    assert np.all(np.isfinite(glacier_grad))
    assert np.all(glacier_grad[np.asarray(land)] < 0.0)   # glacier ice holds more heat per area
    assert np.all(glacier_grad[~np.asarray(land)] == 0.0)

    bucket_grad = jax.grad(total_availability)(start["params"].drainage_timescale)
    assert bool(jnp.all(jnp.isfinite(bucket_grad)))
    assert float(jnp.max(jnp.abs(bucket_grad))) > 0.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("baresoil_albedo", np.nan),
        ("baresoil_albedo", 1.5),
        ("vegetation_albedo", -0.1),
        ("glacier_fraction", 2.0),
        ("soil_volumetric_heat_capacity", 0.0),
        ("glacier_volumetric_heat_capacity", -1.0),
        ("soil_volumetric_heat_capacity", np.inf),
        ("soil_thermal_depth", 0.0),
        ("glacier_depth", -5.0),
        # The bucket's own parameters are checked by the bucket's validation.
        ("soil_bucket_depth", jnp.asarray((0.0, 0.21))),
        ("drainage_timescale", jnp.asarray((-1.0, 86400.0))),
        ("soil_water_field_capacity", jnp.asarray((0.30, 0.10))),
        ("vegetation_fraction", 1.5),
        ("initial_soil_water_fill", -0.1),
        # A timescale or a length is read as a ratio, so an infinite one is
        # silent -- no relaxation, or no snow cover however deep the snow --
        # where a zero one is not. Both signs of nonsense are refused.
        ("temperature_relaxation_timescale", 0.0),
        ("temperature_relaxation_timescale", np.inf),
        ("temperature_relaxation_timescale", np.nan),
        ("snow_depth_to_cover_scale", 0.0),
        ("snow_depth_to_cover_scale", np.inf),
        ("snow_depth_to_cover_scale", np.nan),
    ],
)
def test_invalid_parameters_are_rejected(half_land_grid, field, value):
    """Fractions, heat capacities, depths and scales are checked at construction."""
    params = SlabLandParameters(**{field: value})
    with pytest.raises(ValueError, match=field):
        SlabLandModel(half_land_grid, params=params)


@pytest.mark.parametrize("field", ["baresoil_albedo", "vegetation_albedo",
                                   "glacier_fraction", "vegetation_fraction"])
def test_a_map_off_the_grid_is_rejected(half_land_grid, field):
    params = SlabLandParameters(**{field: jnp.full((3, 4), 0.2)})
    with pytest.raises(ValueError, match=field):
        SlabLandModel(half_land_grid, params=params)


def test_a_grid_without_cell_areas_is_refused():
    grid = dataclasses.replace(make_grid(LAND_FRACTION), cell_solid_angle=None)
    with pytest.raises(ValueError, match="cell areas"):
        SlabLandModel(grid)


def test_the_water_density_is_the_atmospheres(grid):
    """The soil water reads ``jcm.constants.rhow``, as the bucket does."""
    model = SlabLandModel(grid)

    def rain_effect():
        def after(precipitation):
            carry = _forced(model.initialize(), grid, precipitation=precipitation)
            stepped, _ = model.step(carry, coupling_time(0))
            return np.asarray(stepped["state"].soil_water_content)[..., 0]
        return after(1e-7) - after(0.0)

    default = rain_effect()
    original = jcm_constants.physical_constants
    try:
        jcm_constants.set_constants(rhow=2.0 * original.rhow)
        doubled = rain_effect()
    finally:
        jcm_constants.set_constants(original)
    land = LAND_FRACTION > 0.0
    np.testing.assert_allclose(doubled[land], 0.5 * default[land], rtol=2e-2)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output_is_labelled(half_land_grid):
    model = SlabLandModel(half_land_grid)
    _, stacked = run_steps(model, model.initialize(), 2)
    dataset = model.to_xarray(stacked, time_axis(2))

    assert dataset["soil_water_content"].dims == ("time", "lon", "lat", "soil_layer")
    assert dataset["land_surface_temperature"].dims == ("time", "lon", "lat")
    assert dataset["runoff_mass_rate"].dims == ("time",)
    assert dataset["land_surface_temperature"].attrs["jem_role"] == "state"
    assert dataset["surface_albedo"].attrs["jem_role"] == "derived"
    forcing = {str(name) for name in dataset.filter_by_attrs(jem_role="forcing")}
    assert forcing == {
        "forcing_land_total_heat_flux", "forcing_precipitation", "forcing_land_evaporation"}
    for name, variable in dataset.data_vars.items():
        assert "units" in variable.attrs and "long_name" in variable.attrs, name
    assert dataset["forcing_land_total_heat_flux"].attrs["positive"] == "up"
    assert dataset["glacier_fraction"].dims == ("lon", "lat")
    assert "time" not in dataset["glacier_fraction"].dims
