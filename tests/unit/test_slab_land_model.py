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
    DAYS_PER_YEAR,
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    TIMESTEP,
    coupling_time,
    make_grid,
    tree_signature,
    write_climatology,
)


@pytest.fixture
def half_land_grid():
    """Return a 4x3 grid whose eastern half is land, so both mask branches run."""
    shape = (len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))
    fractional_mask = jnp.where(
        jnp.arange(shape[0])[:, None] >= shape[0] // 2,
        jnp.ones(shape),
        jnp.zeros(shape),
    )
    return make_grid(fractional_mask=fractional_mask)


def test_idealized_climatology_varies_with_latitude_only(half_land_grid):
    """The idealised climatology must vary with latitude, not longitude.

    The regression this guards: the profile used to be laid out along the
    longitude axis because `grid.shape` was unpacked as `(n_lat, n_lon)`.
    """
    model = SlabLandModel(half_land_grid)

    temperature = model._idealized_land_temperature()

    n_lon, n_lat = half_land_grid.shape
    assert temperature.shape == (n_lon, n_lat, 12)

    # Standard deviations are taken in float64: a float32 mean of ~10^2
    # values is not bit-exact, so jnp.std of a genuinely constant float32
    # slice returns O(1e-5) rather than zero.
    profile = np.asarray(temperature[:, :, 0], dtype=np.float64)

    # Constant along longitude (axis 0) at every latitude.
    assert np.all(profile.std(axis=0) == 0.0)

    # Genuinely varying along latitude (axis 1) at every longitude.
    assert np.all(profile.std(axis=1) > 1.0)


def test_land_step_runs_and_is_finite(half_land_grid):
    """Three unforced steps stay finite and physically plausible."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()

    for step in range(3):
        carry, _ = model.step(carry, coupling_time(step))

    temperature = carry["state"].land_surface_temperature
    assert bool(jnp.all(jnp.isfinite(temperature)))
    assert float(jnp.min(temperature)) > 150.0
    assert float(jnp.max(temperature)) < 350.0


def test_ocean_cells_report_the_masked_temperature(half_land_grid):
    """Cells below the land threshold are not integrated."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        total_heat_flux=jnp.full(half_land_grid.shape, -100.0)
    )
    stepped, _ = model.step(carry, coupling_time(0))

    ocean = np.asarray(half_land_grid.fractional_mask) < 0.1
    temperature = np.asarray(stepped["state"].land_surface_temperature)
    np.testing.assert_allclose(temperature[ocean], MASKED_SURFACE_TEMPERATURE)
    assert np.all(temperature[~ocean] != MASKED_SURFACE_TEMPERATURE)


def _write_land_climatology(path, half_land_grid, names=("stl", "snowd", "soilw")):
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
            names[1]: (
                ("time", "lat", "lon"),
                np.full((12, n_lat, n_lon), 30.0),
            ),
            names[2]: (
                ("time", "lat", "lon"),
                np.full((12, n_lat, n_lon), 0.25),
            ),
        },
        coords={
            "time": months,
            "lat": LATITUDE_DEGREES,
            "lon": LONGITUDE_DEGREES,
        },
    )
    dataset.to_netcdf(path)
    return str(path), temperature


@pytest.mark.parametrize("names", [("stl", "snowd", "soilw"), ("stl", "snowc", "soilw_am")])
def test_climatology_is_loaded_by_name(tmp_path, half_land_grid, names):
    """Every climatology is resolved by variable name, in either spelling.

    The land model used to read `stl`/`snowd`/`soilw` positionally, which
    accepted any array whose shape happened to fit -- a file on another grid,
    or written in another axis order, loaded silently and wrongly.
    """
    path, temperature = _write_land_climatology(
        tmp_path / f"land_{names[1]}.nc", half_land_grid, names=names
    )
    model = SlabLandModel(half_land_grid, land_clim_file=path)

    assert model.surface_temperature_climatology.shape == half_land_grid.shape + (12,)
    np.testing.assert_allclose(
        np.asarray(model.surface_temperature_climatology)[0, 0, :],
        temperature,
        rtol=1e-5,
    )

    carry = model.initialize()
    # snow depth 30 mm against the 60 mm saturation scale -> half cover.
    np.testing.assert_allclose(np.asarray(carry["state"].snowc), 0.5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(carry["state"].soilw), 0.25, rtol=1e-6)


def test_climatology_on_the_wrong_grid_is_rejected(tmp_path, half_land_grid):
    """A file on a shifted latitude axis is rejected, naming the file."""
    path = write_climatology(
        tmp_path / "land_shifted.nc",
        "stl",
        np.full((12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)), 280.0),
        latitude_degrees=LATITUDE_DEGREES + 1.0,
    )

    with pytest.raises(ValueError, match="latitude") as excinfo:
        SlabLandModel(half_land_grid, land_clim_file=path)

    assert path in str(excinfo.value)


def test_params_default_equivalence(half_land_grid):
    """Constructing with no params is constructing with the defaults."""
    implicit = SlabLandModel(half_land_grid).initialize()
    explicit = SlabLandModel(
        half_land_grid, SlabLandParameters.default()
    ).initialize()

    assert jax.tree_util.tree_structure(implicit) == jax.tree_util.tree_structure(
        explicit
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_state_has_no_clock(tmp_path, half_land_grid):
    """State carries no time; the seasonal cycle comes from the coupler's clock."""
    model = SlabLandModel(half_land_grid)
    assert "sim_time" not in model.initialize()["state"].asdict()

    path, temperature = _write_land_climatology(tmp_path / "land.nc", half_land_grid)
    model = SlabLandModel(half_land_grid, land_clim_file=path)
    carry = model.initialize()

    def climatology_at(year_fraction):
        """Interpolate the monthly cycle exactly as the model does, in numpy."""
        position = (year_fraction % 1.0) * 12.0
        left = int(np.floor(position)) % 12
        weight = position - np.floor(position)
        return (1.0 - weight) * temperature[left] + weight * temperature[
            (left + 1) % 12
        ]

    land = np.asarray(half_land_grid.fractional_mask) >= 0.1
    for step in (0, 182):
        stepped, _ = model.step(carry, coupling_time(step))
        year_fraction = step * TIMESTEP / (86400.0 * DAYS_PER_YEAR)
        # The anomaly starts at zero (the state was initialized from the
        # climatology at year fraction 0) and is unforced, so the temperature
        # is exactly the end-of-step climatology... except at step 0, where the
        # state and the climatology coincide anyway.
        anomaly = climatology_at(0.0) - climatology_at(year_fraction)
        dissipation = (
            (40.0 * 86400.0 / TIMESTEP) / (1.0 + 40.0 * 86400.0 / TIMESTEP)
        )
        expected = dissipation * anomaly + climatology_at(
            year_fraction + 1.0 / DAYS_PER_YEAR
        )
        np.testing.assert_allclose(
            np.asarray(stepped["state"].land_surface_temperature)[land],
            expected,
            rtol=1e-5,
        )


def test_step_shapes_and_dtypes_stable(half_land_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_step_is_differentiable(half_land_grid):
    """Reverse-mode gradients of one step agree with finite differences."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    land = half_land_grid.fractional_mask >= 0.1

    def mean_land_temperature(heat_flux_scale):
        # Scaled to 1e3 W m-2 so a unit change moves the surface temperature by
        # tens of kelvin: a float32 finite difference of a ~290 K field cannot
        # resolve less.
        forced = carry["forcing"].replace(
            total_heat_flux=jnp.full(half_land_grid.shape, 1.0e3 * heat_flux_scale)
        )
        stepped, _ = model.step({**carry, "forcing": forced}, coupling_time(0))
        return jnp.mean(
            jnp.where(land, stepped["state"].land_surface_temperature, 0.0)
        )

    check_grads(
        mean_land_temperature,
        (0.0,),
        order=1,
        modes=["rev"],
        eps=1e-2,
        atol=1e-2,
        rtol=1e-2,
    )


def test_depths_are_differentiable(half_land_grid):
    """The soil depth is a pytree leaf, so a run is differentiable in it."""
    model = SlabLandModel(half_land_grid)

    def mean_temperature(depth_soil):
        carry = model.initialize()
        carry["params"] = carry["params"].replace(depth_soil=depth_soil)
        carry["forcing"] = carry["forcing"].replace(
            total_heat_flux=jnp.full(half_land_grid.shape, -50.0)
        )
        for step in range(3):
            carry, _ = model.step(carry, coupling_time(step))
        return jnp.mean(carry["state"].land_surface_temperature)

    gradient = jax.grad(mean_temperature)(jnp.float32(1.0))
    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 0.0


def test_bind_sets_the_initial_climatology_month(half_land_grid):
    """The start date reaches `initialize()` through the coupler, not a parameter.

    The idealised land climatology peaks in March, so a January start and a
    July start must produce different initial land temperatures over land.
    """

    def initial_land_temperature(start_date):
        model = SlabLandModel(half_land_grid)
        Coupler(
            {"lnd": model},
            coupling_timestep=jdt.to_timedelta(1, "day"),
            start_date=jdt.to_datetime(start_date),
            calendar="365_day",
        )
        assert model.start_year_fraction == pytest.approx(
            0.0 if start_date.endswith("01-01") else 181.0 / DAYS_PER_YEAR
        )
        land = np.asarray(half_land_grid.fractional_mask) > 0.0
        temperature = np.asarray(
            model.initialize()["state"].land_surface_temperature
        )
        return temperature[land]

    january = initial_land_temperature("2001-01-01")
    july = initial_land_temperature("2001-07-01")

    # Poleward cells carry the whole seasonal amplitude; the equator none, so
    # compare the largest difference rather than every cell.
    assert np.max(np.abs(july - january)) > 1.0


def test_rebinding_to_a_different_clock_is_rejected(half_land_grid):
    """One instance belongs to one coupled model; a conflicting second bind raises."""
    model = SlabLandModel(half_land_grid)
    clock = dict(coupling_timestep=jdt.to_timedelta(1, "day"), calendar="365_day")
    Coupler({"lnd": model}, start_date=jdt.to_datetime("2001-01-01"), **clock)
    # The same clock again is harmless (a second coupler over the same run).
    Coupler({"lnd": model}, start_date=jdt.to_datetime("2001-01-01"), **clock)
    with pytest.raises(ValueError, match="already bound"):
        Coupler({"lnd": model}, start_date=jdt.to_datetime("2001-07-01"), **clock)
    with pytest.raises(ValueError, match="already bound"):
        Coupler(
            {"lnd": model},
            start_date=jdt.to_datetime("2001-01-01"),
            coupling_timestep=jdt.to_timedelta(1, "day"),
            calendar="gregorian",
        )
    assert model.start_year_fraction == 0.0


def test_default_albedo_follows_the_carried_params(half_land_grid):
    """Without an explicit albedo field, the soil/ice choice reads carry["params"]."""
    model = SlabLandModel(half_land_grid)
    carry = model.initialize()
    land = np.asarray(half_land_grid.fractional_mask) > 0.0
    forcing = carry["forcing"].replace(
        total_heat_flux=jnp.full(half_land_grid.shape, 100.0)
    )
    carry = dict(carry, forcing=forcing)

    soil, _ = model.step(carry, coupling_time(0))
    # Raising the carried albedo above the ice threshold switches every land
    # cell to the (thicker, different heat capacity) ice-sheet slab, so the
    # response to the same flux changes; snapshotting the albedo at
    # construction would have left it unchanged.
    ice_params = carry["params"].replace(surface_albedo=0.8)
    ice, _ = model.step(dict(carry, params=ice_params), coupling_time(0))
    assert not np.allclose(
        np.asarray(soil["state"].land_surface_temperature)[land],
        np.asarray(ice["state"].land_surface_temperature)[land],
    )


@pytest.mark.parametrize("bad_value", [-0.1, 1.5, np.nan, np.inf], ids=["negative", "above_one", "nan", "inf"])
def test_explicit_albedo_is_validated(half_land_grid, bad_value):
    """A supplied albedo field must be finite and in [0, 1]."""
    albedo = np.full(half_land_grid.shape, 0.2)
    albedo[0, 0] = bad_value
    with pytest.raises(ValueError, match="surface_albedo"):
        SlabLandModel(half_land_grid, surface_albedo=albedo)


@pytest.mark.parametrize(
    "field, value",
    [
        ("surface_albedo", np.nan),
        ("surface_albedo", 1.5),
        ("soil_volumetric_heat_capacity", 0.0),
        ("land_ice_volumetric_heat_capacity", -1.0),
        ("soil_volumetric_heat_capacity", np.inf),
    ],
)
def test_invalid_scalar_parameters_are_rejected(half_land_grid, field, value):
    """The default albedo and the heat capacities are checked at construction."""
    params = SlabLandParameters(**{field: value})
    with pytest.raises(ValueError, match=field):
        SlabLandModel(half_land_grid, params=params)
