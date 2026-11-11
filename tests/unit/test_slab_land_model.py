"""Tests for `jem.components.slab.slab_land_model`."""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr
from jax.test_util import check_grads

from jem.base.coupler import Coupler
from jem.components.slab.base import MASKED_SURFACE_TEMPERATURE
from jem.components.slab.slab_land_model import SlabLandModel, SlabLandParameters
from tests.unit.slab_test_utils import (
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    TIMESTEP,
    coupling_time,
    make_grid,
    tree_signature,
    write_climatology,
)

#: The length of 2001, the year `coupling_time`'s start date falls in.
DAYS_PER_YEAR = 365.0

#: Density of liquid water (kg m-3), the bucket's conversion from a
#: volumetric water content times a depth to a mass per area.
WATER_DENSITY = 1000.0

#: SPEEDY's names for the land climatologies, as jax-gcm packages them.
SPEEDY_VARNAMES = {
    "land_surface_temperature_varname_in_clim_file": "stl",
    "snow_depth_varname_in_clim_file": "snowc",
    "soil_water_availability_varname_in_clim_file": "soilw_am",
}


@pytest.fixture
def half_land_grid():
    """Return a 4x3 grid whose eastern half is land, so both mask branches run."""
    shape = (len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))
    land_fraction = jnp.where(
        jnp.arange(shape[0])[:, None] >= shape[0] // 2,
        jnp.ones(shape),
        jnp.zeros(shape),
    )
    return make_grid(land_fraction=land_fraction)


def _land(grid):
    return np.asarray(grid.binary_land_mask) == 1


def _water_mass(state, params):
    """Soil water per unit area (kg m-2), summed over both layers."""
    return WATER_DENSITY * jnp.sum(state.soil_water_content * params.soil_bucket_depth, axis=-1)


def _forced(carry, grid, precipitation=0.0, evaporation=0.0, total_heat_flux=0.0):
    forcing = carry["forcing"].replace(
        precipitation=jnp.broadcast_to(jnp.asarray(precipitation, jnp.float32), grid.shape),
        evaporation=jnp.broadcast_to(jnp.asarray(evaporation, jnp.float32), grid.shape),
        total_heat_flux=jnp.broadcast_to(jnp.asarray(total_heat_flux, jnp.float32), grid.shape),
    )
    return dict(carry, forcing=forcing)


def _with_soil_water(carry, grid, surface, deep):
    water = jnp.where(
        jnp.asarray(_land(grid))[..., None], jnp.asarray([surface, deep], jnp.float32), 0.0
    )
    return dict(carry, state=carry["state"].replace(soil_water_content=water))


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


def test_ocean_cells_report_the_masked_temperature(half_land_grid):
    """Cells with no land are not integrated."""
    model = SlabLandModel(half_land_grid)
    stepped, _ = model.step(
        _forced(model.initialize(), half_land_grid, total_heat_flux=-100.0), coupling_time(0)
    )

    land = _land(half_land_grid)
    temperature = np.asarray(stepped["state"].land_surface_temperature)
    np.testing.assert_allclose(temperature[~land], MASKED_SURFACE_TEMPERATURE)
    assert np.all(temperature[land] != MASKED_SURFACE_TEMPERATURE)


def test_a_one_metre_slab_warms_at_its_heat_capacity(half_land_grid):
    """100 W m-2 into 1 m of soil for one hour warms it by 100 * 3600 / 2.5e6 K."""
    model = SlabLandModel(half_land_grid)
    carry = _forced(model.initialize(), half_land_grid, total_heat_flux=-100.0)
    before = np.asarray(carry["state"].land_surface_temperature)
    stepped, _ = model.step(carry, coupling_time(0, dt=3600.0))

    land = _land(half_land_grid)
    warming = np.asarray(stepped["state"].land_surface_temperature)[land] - before[land]
    np.testing.assert_allclose(warming, 100.0 * 3600.0 / 2.5e6, rtol=1e-4)


def _write_land_climatology(path, names=("stl", "snowc", "soilw_am")):
    """Write a 12-month land climatology on the test grid, by name."""
    n_lat, n_lon = len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)
    months = np.arange(12)
    temperature = 280.0 + 5.0 * np.cos(2 * np.pi * months / 12.0)
    dataset = xr.Dataset(
        data_vars={
            names[0]: (
                ("time", "lat", "lon"),
                np.broadcast_to(temperature[:, None, None], (12, n_lat, n_lon)).copy(),
            ),
            names[1]: (("time", "lat", "lon"), np.full((12, n_lat, n_lon), 30.0)),
            names[2]: (("time", "lat", "lon"), np.full((12, n_lat, n_lon), 0.25)),
        },
        coords={"time": months, "lat": LATITUDE_DEGREES, "lon": LONGITUDE_DEGREES},
    )
    dataset.to_netcdf(path)
    return str(path), temperature


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
    np.testing.assert_allclose(
        np.asarray(stepped["derived"].soil_water_availability)[land], 0.25, rtol=1e-6
    )
    assert float(np.abs(np.asarray(stepped["derived"].soil_water_availability)[~land]).max()) == 0.0


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

    def climatology_at(year_fraction):
        position = (year_fraction % 1.0) * 12.0
        left = int(np.floor(position)) % 12
        weight = position - np.floor(position)
        return (1.0 - weight) * temperature[left] + weight * temperature[(left + 1) % 12]

    land = _land(half_land_grid)
    timescale_steps = 40.0 * 86400.0 / TIMESTEP
    dissipation = timescale_steps / (1.0 + timescale_steps)
    for step in (0, 182):
        stepped, _ = model.step(carry, coupling_time(step))
        year_fraction = step * TIMESTEP / (86400.0 * DAYS_PER_YEAR)
        anomaly = climatology_at(0.0) - climatology_at(year_fraction)
        expected = dissipation * anomaly + climatology_at(year_fraction + 1.0 / DAYS_PER_YEAR)
        np.testing.assert_allclose(
            np.asarray(stepped["state"].land_surface_temperature)[land], expected, rtol=1e-5
        )


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
# Soil water bucket
# ---------------------------------------------------------------------------


def test_soil_water_starts_at_the_wilting_point(half_land_grid):
    """The driest state the bucket allows, and zero off land."""
    model = SlabLandModel(half_land_grid)
    water = np.asarray(model.initialize()["state"].soil_water_content)
    land = _land(half_land_grid)

    np.testing.assert_allclose(
        water[land], np.broadcast_to(np.asarray(model.params.soil_water_wilting_point), water[land].shape)
    )
    assert float(np.abs(water[~land]).max()) == 0.0


@pytest.mark.parametrize(
    "surface, deep, precipitation, evaporation",
    [
        (0.25, 0.28, 0.0, 3e-5),    # both layers wet
        (0.25, 0.17, 0.0, 3e-5),    # deep layer at its wilting point
        (0.00, 0.28, 0.0, 3e-5),    # dry surface, wet deep layer
        (0.01, 0.18, 0.0, 1e-4),    # demand exceeds supply
        (0.28, 0.33, 5e-4, 1e-5),   # heavy rain over a full bucket
        (0.10, 0.20, 1e-5, -1e-5),  # dew
    ],
)
def test_the_water_budget_closes(half_land_grid, surface, deep, precipitation, evaporation):
    """P - (E - unmet) - drainage - overflow is exactly the change in soil water."""
    model = SlabLandModel(half_land_grid)
    carry = _forced(
        _with_soil_water(model.initialize(), half_land_grid, surface, deep),
        half_land_grid, precipitation=precipitation, evaporation=evaporation,
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
    land = _land(half_land_grid)
    np.testing.assert_allclose(np.asarray(change)[land], np.asarray(net)[land], atol=1e-9)


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
    # 8.64 mm of demand against about 2.8 mm the two layers can give.
    assert float(np.asarray(dry["derived"].unmet_evaporation)[land].min()) > 0.0

    wet, _ = model.step(
        _forced(model.initialize(), half_land_grid, precipitation=1e-3), coupling_time(0)
    )
    water = np.asarray(wet["state"].soil_water_content)[land]
    assert np.all(water <= np.asarray(params.soil_water_field_capacity) + 1e-7)
    assert float(np.asarray(wet["derived"].soil_water_overflow)[land].min()) > 0.0


def test_evaporation_is_shared_by_the_water_each_layer_can_give(half_land_grid):
    """The deep layer gives water only through vegetation, and dew goes on top."""
    # A drainage timescale far longer than the step leaves evaporation alone.
    params = SlabLandParameters(drainage_timescale=jnp.asarray((1e15, 1e15)))
    model = SlabLandModel(half_land_grid, params)
    land = _land(half_land_grid)
    start = _with_soil_water(model.initialize(), half_land_grid, 0.10, 0.27)

    stepped, _ = model.step(_forced(start, half_land_grid, evaporation=1e-6), coupling_time(0))
    taken = (
        np.asarray(start["state"].soil_water_content - stepped["state"].soil_water_content)
        * np.asarray(params.soil_bucket_depth)
    )[land]
    vegetation = 0.8
    available = np.array([0.07 * 0.10, vegetation * 0.21 * (0.27 - 0.17)])
    np.testing.assert_allclose(taken[:, 0] / taken.sum(axis=-1), available[0] / available.sum(), rtol=1e-4)

    dew, _ = model.step(_forced(start, half_land_grid, evaporation=-1e-6), coupling_time(0))
    gained = np.asarray(dew["state"].soil_water_content - start["state"].soil_water_content)[land]
    assert np.all(gained[:, 0] > 0.0)
    np.testing.assert_allclose(gained[:, 1], 0.0, atol=1e-7)


def test_soil_water_availability_is_speedys(half_land_grid):
    """The availability is the vegetation-weighted water above the wilting point."""
    model = SlabLandModel(half_land_grid)
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


def test_runoff_to_ocean_is_drainage_plus_overflow_through_the_mapping(half_land_grid):
    """The routing is applied to the water that leaves the bucket, on the model grid."""
    def to_the_first_column(runoff):
        return jnp.zeros_like(runoff).at[0].set(jnp.sum(runoff, axis=0))

    model = SlabLandModel(half_land_grid, runoff_to_ocean_mapping=to_the_first_column)
    stepped, _ = model.step(
        _forced(_with_soil_water(model.initialize(), half_land_grid, 0.28, 0.33),
                half_land_grid, precipitation=5e-4),
        coupling_time(0),
    )
    derived = stepped["derived"]
    leaving = derived.deep_drainage + derived.soil_water_overflow
    np.testing.assert_allclose(
        np.asarray(derived.runoff_to_ocean), np.asarray(to_the_first_column(leaving)), rtol=1e-6
    )
    np.testing.assert_allclose(
        float(jnp.sum(derived.runoff_to_ocean)), float(jnp.sum(leaving)), rtol=1e-6
    )


def test_by_default_the_runoff_is_spread_evenly_over_the_cells_with_no_land(half_land_grid):
    """The area-weighted total is kept, as one uniform flux on all-ocean cells."""
    model = SlabLandModel(half_land_grid)
    stepped, _ = model.step(
        _forced(_with_soil_water(model.initialize(), half_land_grid, 0.28, 0.33),
                half_land_grid, precipitation=5e-4),
        coupling_time(0),
    )
    derived = stepped["derived"]
    area = np.cos(np.asarray(half_land_grid.latitude_radian))
    leaving = np.asarray(derived.deep_drainage + derived.soil_water_overflow)
    routed = np.asarray(derived.runoff_to_ocean)
    no_land = np.asarray(half_land_grid.land_fraction) == 0.0

    assert leaving.sum() > 0.0
    np.testing.assert_allclose((routed * area).sum(), (leaving * area).sum(), rtol=1e-5)
    np.testing.assert_allclose(routed[no_land], routed[no_land][0], rtol=1e-6)
    assert float(np.abs(routed[~no_land]).max()) == 0.0


def test_a_mapping_off_the_model_grid_is_rejected(half_land_grid):
    with pytest.raises(ValueError, match="runoff_to_ocean_mapping"):
        SlabLandModel(half_land_grid, runoff_to_ocean_mapping=lambda runoff: runoff[:1])


def test_vegetation_fraction_is_read_from_the_climatology(tmp_path, half_land_grid):
    """A prescribed vegetation fraction reaches the albedo and the availability."""
    n_lat, n_lon = len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)
    values = np.full((12, n_lat, n_lon), 0.25, dtype=np.float32)
    path = write_climatology(tmp_path / "vegetation.nc", "vegetation_fraction", values)
    model = SlabLandModel(half_land_grid, land_clim_file=path, vegetation_fraction_from_clim_file=True)

    stepped, _ = model.step(model.initialize(), coupling_time(0))
    land = _land(half_land_grid)
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


# ---------------------------------------------------------------------------
# Structure, parameters and gradients
# ---------------------------------------------------------------------------


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

    wetter = model.params.replace(soil_water_wilting_point=jnp.asarray((0.05, 0.2)))
    carry = model.initialize(wetter)
    np.testing.assert_allclose(
        np.asarray(carry["state"].soil_water_content)[_land(half_land_grid)], [[0.05, 0.2]] * 6
    )
    assert carry["params"] is wetter


def test_step_shapes_and_dtypes_stable(half_land_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_two_coupled_steps_keep_the_carry(half_land_grid):
    """The structure check that only lax.scan makes."""
    model = SlabLandModel(half_land_grid)
    coupler = Coupler(
        {"lnd": model},
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=jdt.to_datetime("2001-01-01"),
    )
    final, _ = jax.jit(coupler.generate_trajectory_function(2))(coupler.initialize())
    assert int(final.step) == 2


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
        jnp.float32(2e-5), model.params.soil_bucket_depth
    )
    for gradient in jax.tree_util.tree_leaves(gradients):
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_the_thermal_depth_is_differentiable(half_land_grid):
    """The soil thermal depth is a pytree leaf, so a run is differentiable in it."""
    model = SlabLandModel(half_land_grid)

    def mean_temperature(depth):
        carry = model.initialize()
        carry["params"] = carry["params"].replace(soil_thermal_depth=depth)
        carry = _forced(carry, half_land_grid, total_heat_flux=-50.0)
        for step in range(3):
            carry, _ = model.step(carry, coupling_time(step))
        return jnp.mean(carry["state"].land_surface_temperature)

    gradient = jax.grad(mean_temperature)(jnp.float32(1.0))
    assert bool(jnp.isfinite(gradient)) and abs(float(gradient)) > 0.0


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
        ("soil_bucket_depth", jnp.asarray((0.0, 0.21))),
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


def test_an_albedo_map_off_the_grid_is_rejected(half_land_grid):
    params = SlabLandParameters(baresoil_albedo=jnp.full((3, 4), 0.2))
    with pytest.raises(ValueError, match="baresoil_albedo"):
        SlabLandModel(half_land_grid, params=params)
