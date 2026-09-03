"""Tests for `jem.components.slab.slab_ocean_model`."""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from jem import constants
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_ocean_model import SlabOceanModel

LONGITUDE_DEGREES = np.array([0.0, 90.0, 180.0, 270.0])
LATITUDE_DEGREES = np.array([-60.0, 0.0, 60.0])
TIMESTEP = 86400.0


@pytest.fixture
def uniform_grid() -> SlabGrid:
    """Build a tiny all-ocean 4x3 lon-lat grid by hand so the tests own it."""
    longitude_2d, latitude_2d = np.meshgrid(
        np.deg2rad(LONGITUDE_DEGREES), np.deg2rad(LATITUDE_DEGREES), indexing="ij"
    )
    return SlabGrid(
        fractional_mask=jnp.zeros((len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))),
        latitude_radian=jnp.asarray(latitude_2d),
        longitude_radian=jnp.asarray(longitude_2d),
    )


def _write_climatology(
    path,
    var: str,
    values_time_lat_lon: np.ndarray,
    longitude_degrees: np.ndarray = LONGITUDE_DEGREES,
    latitude_degrees: np.ndarray = LATITUDE_DEGREES,
) -> str:
    """Write a 12-month climatology in (time, lat, lon) order and return its path."""
    dataset = xr.Dataset(
        data_vars={var: (("time", "lat", "lon"), values_time_lat_lon)},
        coords={
            "time": np.arange(12),
            "lat": latitude_degrees,
            "lon": longitude_degrees,
        },
    )
    dataset.to_netcdf(path)
    return str(path)


def _monthly_ramp() -> np.ndarray:
    """Return a (12, n_lat, n_lon) field whose every element is distinct."""
    return np.arange(
        12 * len(LATITUDE_DEGREES) * len(LONGITUDE_DEGREES), dtype=np.float32
    ).reshape(12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES))


def test_qflux_file_is_loaded(tmp_path, uniform_grid):
    """A Q-flux file is read into the forcing carry, transposed by name."""
    values = _monthly_ramp()
    q_flux_file = _write_climatology(tmp_path / "qflux.nc", "qflux", values)

    model = SlabOceanModel(
        grid=uniform_grid,
        timestep=TIMESTEP,
        forcing_method="Qflux",
        Q_flux_file=q_flux_file,
    )
    q_flux = model.initialize()["forcing"].q_flux

    # (time, lat, lon) in the file -> (lon, lat, time) in the model.
    expected = np.transpose(values, (2, 1, 0))
    assert q_flux.shape == expected.shape
    np.testing.assert_allclose(np.asarray(q_flux), expected)


def test_climatology_rejects_wrong_coords(tmp_path, uniform_grid):
    """A file on a shifted longitude axis is rejected, naming the file."""
    q_flux_file = _write_climatology(
        tmp_path / "qflux_shifted.nc",
        "qflux",
        _monthly_ramp(),
        longitude_degrees=LONGITUDE_DEGREES + 1.0,
    )

    model = SlabOceanModel(
        grid=uniform_grid,
        timestep=TIMESTEP,
        forcing_method="Qflux",
        Q_flux_file=q_flux_file,
    )
    with pytest.raises(ValueError, match="longitude") as excinfo:
        model.initialize()

    assert q_flux_file in str(excinfo.value)


def test_qflux_zero_without_file(uniform_grid):
    """Q-flux forcing without a file is still a valid (zero-forcing) setup."""
    model = SlabOceanModel(
        grid=uniform_grid,
        timestep=TIMESTEP,
        forcing_method="Qflux",
    )
    q_flux = model.initialize()["forcing"].q_flux

    assert q_flux.shape == uniform_grid.shape + (12,)
    assert bool(jnp.all(q_flux == 0.0))


def test_relaxation_matches_analytic(tmp_path, uniform_grid):
    """One relaxation step reproduces the closed-form Euler-backward update.

    The climatology is constant in time so the interpolated climatology drops
    out of the comparison and only the temperature update is under test. The
    initial SST is displaced from the climatology so that both terms of the
    update -- the damped anomaly and the heat-flux increment -- are exercised.
    """
    climatology_lat_lon = 290.0 + np.arange(
        len(LATITUDE_DEGREES) * len(LONGITUDE_DEGREES), dtype=np.float32
    ).reshape(len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES))
    values = np.broadcast_to(climatology_lat_lon, (12,) + climatology_lat_lon.shape)
    sst_file = _write_climatology(tmp_path / "sst.nc", "sst", np.array(values))

    relaxation_time = 5.0 * TIMESTEP
    initial_anomaly = 5.0
    heat_flux = -2000.0  # W m-2; upward-positive convention, so negative warms

    model = SlabOceanModel(
        grid=uniform_grid,
        timestep=TIMESTEP,
        forcing_method="relaxation",
        relaxation_time=relaxation_time,
        SST_clim_file=sst_file,
    )
    carry = model.initialize()

    climatology = np.transpose(climatology_lat_lon).astype(np.float64)
    carry["state"] = carry["state"].replace(
        sea_surface_temperature=carry["state"].sea_surface_temperature
        + initial_anomaly,
    )
    carry["forcing"] = carry["forcing"].replace(
        total_heat_flux=jnp.full(uniform_grid.shape, heat_flux),
    )
    new_carry, _ = model.generate_step_function()(carry, 0)

    mixed_layer_depth = np.asarray(carry["state"].mixed_layer_depth, dtype=np.float64)
    heat_capacity = (
        constants.ocean_density
        * constants.ocean_specific_heat_capacity
        * mixed_layer_depth
    )
    time_factor = 1.0 / (1.0 + TIMESTEP / relaxation_time)
    expected = climatology + time_factor * (
        initial_anomaly + TIMESTEP / heat_capacity * (-heat_flux)
    )

    np.testing.assert_allclose(
        np.asarray(new_carry["state"].sea_surface_temperature, dtype=np.float64),
        expected,
        rtol=1e-5,
        atol=1e-3,
    )


def test_freeze_melt_energy_sign(uniform_grid):
    """A large upward heat flux drives the mixed layer to the freezing point."""
    model = SlabOceanModel(grid=uniform_grid, timestep=TIMESTEP)
    carry = model.initialize()

    # Big enough to remove tens of kelvin from the mixed layer in one step.
    carry["forcing"] = carry["forcing"].replace(
        total_heat_flux=jnp.full(uniform_grid.shape, 1.0e5),
    )
    new_carry, _ = model.generate_step_function()(carry, 0)

    sea_surface_temperature = new_carry["state"].sea_surface_temperature
    ice_frazil_melt_energy = new_carry["derived"].ice_frazil_melt_energy

    assert bool(jnp.all(ice_frazil_melt_energy > 0.0))
    np.testing.assert_allclose(
        np.asarray(sea_surface_temperature),
        constants.seawater_freezing_point_K,
        rtol=1e-6,
    )


def test_qflux_positive_warms_ocean(uniform_grid, tmp_path):
    """A positive Q-flux is a heat source: with no atmospheric flux the SST must rise.

    Guards the sign convention: Q is defined as ``+Q/(rho cp h)`` in the SST
    equation, but the step folds it into the UPWARD-positive total heat flux,
    which is negated -- so Q has to enter with a minus sign there.
    """
    q_file = tmp_path / "qflux.nc"
    _write_climatology(q_file, "qflux", uniform_grid, fill_value=50.0)
    model = SlabOceanModel(
        grid=uniform_grid,
        forcing_method="Qflux",
        Q_flux_file=str(q_file),
        timestep=86400.0,
    )
    carry = model.initialize()
    step = model.generate_step_function()
    new_carry, _ = step(carry, 0)
    sst_before = carry["state"].sea_surface_temperature
    sst_after = new_carry["state"].sea_surface_temperature
    assert bool(jnp.all(sst_after > sst_before)), "positive Q-flux must warm the mixed layer"
