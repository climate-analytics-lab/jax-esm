"""Tests for `jem.components.slab.slab_ocean_model`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import jcm.constants as jcm_constants
from jem import constants
from jem.components.slab.base import load_monthly_climatology
from jem.components.slab.slab_ocean_model import SlabOceanModel, SlabOceanParameters
from tests.unit.slab_test_utils import (
    DAYS_PER_YEAR,
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    TIMESTEP,
    coupling_time,
    make_grid,
    monthly_ramp,
    run_steps,
    tree_signature,
    write_climatology,
)


@pytest.fixture
def uniform_grid():
    """Return a tiny all-ocean 4x3 lon-lat grid the tests own."""
    return make_grid()


def test_load_monthly_climatology_is_shared_and_transposes_by_name(tmp_path, uniform_grid):
    """The loader is public on the slab base module so every slab component can use it."""
    values = monthly_ramp()
    path = write_climatology(tmp_path / "field.nc", "field", values)

    loaded = load_monthly_climatology(path, "field", uniform_grid)

    # (time, lat, lon) in the file -> (lon, lat, time) on the model grid.
    np.testing.assert_allclose(np.asarray(loaded), np.transpose(values, (2, 1, 0)))


def test_qflux_file_is_loaded(tmp_path, uniform_grid):
    """A Q-flux file is read into the forcing carry, transposed by name."""
    values = monthly_ramp()
    q_flux_file = write_climatology(tmp_path / "qflux.nc", "qflux", values)

    model = SlabOceanModel(
        uniform_grid,
        SlabOceanParameters(forcing_method="qflux"),
        q_flux_file=q_flux_file,
    )
    q_flux = model.initialize()["forcing"].q_flux

    # (time, lat, lon) in the file -> (lon, lat, time) in the model.
    expected = np.transpose(values, (2, 1, 0))
    assert q_flux.shape == expected.shape
    np.testing.assert_allclose(np.asarray(q_flux), expected)


def test_climatology_rejects_wrong_coords(tmp_path, uniform_grid):
    """A file on a shifted longitude axis is rejected, naming the file.

    Boundary data is loaded by the constructor now, so the rejection happens
    where the caller can see which line configured the component.
    """
    q_flux_file = write_climatology(
        tmp_path / "qflux_shifted.nc",
        "qflux",
        monthly_ramp(),
        longitude_degrees=LONGITUDE_DEGREES + 1.0,
    )

    with pytest.raises(ValueError, match="longitude") as excinfo:
        SlabOceanModel(
            uniform_grid,
            SlabOceanParameters(forcing_method="qflux"),
            q_flux_file=q_flux_file,
        )

    assert q_flux_file in str(excinfo.value)


def test_qflux_zero_without_file(uniform_grid):
    """Q-flux forcing without a file is still a valid (zero-forcing) setup."""
    model = SlabOceanModel(uniform_grid, SlabOceanParameters(forcing_method="qflux"))
    q_flux = model.initialize()["forcing"].q_flux

    assert q_flux.shape == uniform_grid.shape + (12,)
    assert bool(jnp.all(q_flux == 0.0))


def test_qflux_file_with_wrong_forcing_method_is_rejected(tmp_path, uniform_grid):
    """A Q-flux file the run would silently ignore is a configuration error."""
    q_flux_file = write_climatology(tmp_path / "qflux.nc", "qflux", monthly_ramp())

    with pytest.raises(ValueError, match="forcing_method"):
        SlabOceanModel(uniform_grid, q_flux_file=q_flux_file)


def test_relaxation_without_climatology_is_rejected(uniform_grid):
    """Relaxation needs something to relax to."""
    with pytest.raises(ValueError, match="sst_clim_file"):
        SlabOceanModel(uniform_grid, SlabOceanParameters(forcing_method="relaxation"))


def _constant_climatology(tmp_path, value_lat_lon):
    """Write a 12-month SST climatology that does not vary through the year."""
    values = np.broadcast_to(value_lat_lon, (12,) + value_lat_lon.shape)
    return write_climatology(tmp_path / "sst.nc", "sst", np.array(values))


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
    sst_file = _constant_climatology(tmp_path, climatology_lat_lon)

    relaxation_time = 5.0 * TIMESTEP
    initial_anomaly = 5.0
    heat_flux = -2000.0  # W m-2; upward-positive convention, so negative warms

    model = SlabOceanModel(
        uniform_grid,
        SlabOceanParameters(
            forcing_method="relaxation", relaxation_time=relaxation_time
        ),
        sst_clim_file=sst_file,
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
    new_carry, _ = model.step(carry, coupling_time(0))

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
    model = SlabOceanModel(uniform_grid)
    carry = model.initialize()

    # Big enough to remove tens of kelvin from the mixed layer in one step.
    carry["forcing"] = carry["forcing"].replace(
        total_heat_flux=jnp.full(uniform_grid.shape, 1.0e5),
    )
    new_carry, _ = model.step(carry, coupling_time(0))

    assert bool(jnp.all(new_carry["derived"].ice_frazil_melt_energy > 0.0))
    np.testing.assert_allclose(
        np.asarray(new_carry["state"].sea_surface_temperature),
        constants.seawater_freezing_point_K,
        rtol=1e-6,
    )


def test_qflux_positive_warms_ocean(tmp_path, uniform_grid):
    """A positive Q-flux is a heat source: with no atmospheric flux the SST must rise.

    Guards the sign convention: Q is defined as ``+Q/(rho cp h)`` in the SST
    equation, but the step folds it into the UPWARD-positive total heat flux,
    which is negated -- so Q has to enter with a minus sign there.
    """
    q_file = write_climatology(
        tmp_path / "qflux.nc",
        "qflux",
        np.full((12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)), 50.0),
    )
    model = SlabOceanModel(
        uniform_grid,
        SlabOceanParameters(forcing_method="qflux"),
        q_flux_file=q_file,
    )
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert bool(
        jnp.all(
            new_carry["state"].sea_surface_temperature
            > carry["state"].sea_surface_temperature
        )
    ), "positive Q-flux must warm the mixed layer"


def test_params_default_equivalence(uniform_grid):
    """Constructing with no params is constructing with the defaults."""
    implicit = SlabOceanModel(uniform_grid).initialize()
    explicit = SlabOceanModel(
        uniform_grid, SlabOceanParameters.default()
    ).initialize()

    assert jax.tree_util.tree_structure(implicit) == jax.tree_util.tree_structure(
        explicit
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_params_travel_in_the_carry(uniform_grid):
    """The tunables are pytree leaves of the carry, not closure constants."""
    model = SlabOceanModel(uniform_grid)
    carry = model.initialize()

    assert carry["params"] == model.params
    assert any(
        leaf is not None and np.ndim(leaf) == 0
        for leaf in jax.tree_util.tree_leaves(carry["params"])
    )


def test_state_has_no_clock(tmp_path, uniform_grid):
    """State carries no time, and the seasonal cycle comes from the coupler's clock.

    The same carry stepped with two different `CouplingTime`s must give two
    different answers, and specifically the ones the Euler-backward relaxation
    to a month-varying climatology predicts. That is the whole point of moving
    the clock out of the component: nothing but `time` distinguishes these two
    calls.
    """
    assert "sim_time" not in SlabOceanModel(uniform_grid).initialize()["state"].asdict()

    monthly = 285.0 + 10.0 * np.cos(2 * np.pi * np.arange(12) / 12.0)
    seasonal = np.broadcast_to(
        monthly[:, None, None], (12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES))
    )
    sst_file = write_climatology(tmp_path / "sst.nc", "sst", np.array(seasonal))

    def climatology_at(year_fraction):
        """Interpolate the monthly cycle exactly as the model does, in numpy."""
        position = (year_fraction % 1.0) * 12.0
        left = int(np.floor(position)) % 12
        weight = position - np.floor(position)
        return (1.0 - weight) * monthly[left] + weight * monthly[(left + 1) % 12]

    relaxation_time = TIMESTEP  # so the damping factor is exactly 1/2
    model = SlabOceanModel(
        uniform_grid,
        SlabOceanParameters(
            forcing_method="relaxation", relaxation_time=relaxation_time
        ),
        sst_clim_file=sst_file,
    )
    carry = model.initialize()
    initial = float(np.asarray(carry["state"].sea_surface_temperature)[0, 0])

    for step in (0, 182):
        stepped, _ = model.step(carry, coupling_time(step))
        year_fraction = step * TIMESTEP / (86400.0 * DAYS_PER_YEAR)
        expected = 0.5 * (initial - climatology_at(year_fraction)) + climatology_at(
            year_fraction + 1.0 / DAYS_PER_YEAR
        )
        np.testing.assert_allclose(
            np.asarray(stepped["state"].sea_surface_temperature),
            expected,
            rtol=1e-5,
        )


def test_step_shapes_and_dtypes_stable(uniform_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabOceanModel(uniform_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_grad_wrt_relaxation_time_is_finite(tmp_path, uniform_grid):
    """The relaxation timescale is differentiable through a multi-step run."""
    climatology = np.full((12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES)), 290.0)
    sst_file = write_climatology(tmp_path / "sst.nc", "sst", climatology)

    def mean_final_sst(relaxation_time):
        model = SlabOceanModel(
            uniform_grid,
            SlabOceanParameters(forcing_method="relaxation"),
            sst_clim_file=sst_file,
        )
        carry = model.initialize()
        # A steady heat flux keeps the anomaly non-zero, so the relaxation
        # timescale actually does something over the five steps.
        carry["forcing"] = carry["forcing"].replace(
            total_heat_flux=jnp.full(uniform_grid.shape, -500.0)
        )
        carry["params"] = carry["params"].replace(relaxation_time=relaxation_time)
        carry, _ = run_steps(model, carry, 5)
        return jnp.mean(carry["state"].sea_surface_temperature)

    gradient = jax.grad(mean_final_sst)(jnp.float32(5.0 * TIMESTEP))

    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 0.0


def test_step_is_differentiable(uniform_grid):
    """Reverse-mode gradients of one step agree with finite differences."""
    model = SlabOceanModel(uniform_grid)
    carry = model.initialize()

    def mean_sst(heat_flux_scale):
        # Scaled to 1e4 W m-2 so a unit change moves SST by a few kelvin: a
        # float32 finite difference of a ~288 K field cannot resolve less.
        forced = carry["forcing"].replace(
            total_heat_flux=jnp.full(uniform_grid.shape, 1.0e4 * heat_flux_scale)
        )
        stepped, _ = model.step({**carry, "forcing": forced}, coupling_time(0))
        return jnp.mean(stepped["state"].sea_surface_temperature)

    check_grads(mean_sst, (0.0,), order=1, modes=["rev"], eps=1e-2, atol=1e-2, rtol=1e-2)


def test_freeze_melt_energy_uses_jcm_constants(uniform_grid):
    """Nothing in the ocean model shadows a constant jcm.constants already owns."""
    assert not hasattr(constants, "freezing_point_K")
    assert constants.seawater_freezing_point_K < jcm_constants.tmelt
