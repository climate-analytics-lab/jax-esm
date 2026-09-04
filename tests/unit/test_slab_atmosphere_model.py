"""Tests for `jem.components.slab.slab_atmosphere_model`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import jcm.constants as jcm_constants
from jem import constants
from jem.components.slab.slab_atmosphere_model import (
    SlabAtmosphereModel,
    SlabAtmosphereParameters,
)
from tests.unit.slab_test_utils import (
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    TIMESTEP,
    coupling_time,
    make_grid,
    tree_signature,
)


@pytest.fixture
def half_land_grid():
    """Return a 4x3 grid whose eastern half is land, so both surfaces are used."""
    shape = (len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))
    fractional_mask = jnp.where(
        jnp.arange(shape[0])[:, None] >= shape[0] // 2,
        jnp.ones(shape),
        jnp.zeros(shape),
    )
    return make_grid(fractional_mask=fractional_mask)


def test_each_cell_feels_the_surface_beneath_it(half_land_grid):
    """Land cells are heated by the land, ocean cells by the ocean.

    The two surface temperatures are set far apart so a cell that read the
    wrong one would be obvious.
    """
    model = SlabAtmosphereModel(half_land_grid)
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        land_surface_temperature=jnp.full(half_land_grid.shape, 400.0),
        sea_surface_temperature=jnp.full(half_land_grid.shape, 200.0),
    )
    stepped, _ = model.step(carry, coupling_time(0))

    land = np.asarray(half_land_grid.binary_mask) == 1.0
    heat_flux = np.asarray(stepped["derived"].internal_total_heat_flux)
    assert np.all(heat_flux[land] > 0.0), "warm land must heat the air above it"
    assert np.all(heat_flux[~land] < 0.0), "cold ocean must cool the air above it"

    warming = np.asarray(
        stepped["state"].mean_air_temperature - carry["state"].mean_air_temperature
    )
    assert np.all(warming[land] > 0.0)
    assert np.all(warming[~land] < 0.0)


def test_bulk_flux_matches_the_analytic_formula(half_land_grid):
    """One step reproduces the bulk formula written down in the class docstring."""
    surface_temperature = 300.0
    model = SlabAtmosphereModel(half_land_grid)
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        land_surface_temperature=jnp.full(half_land_grid.shape, surface_temperature),
        sea_surface_temperature=jnp.full(half_land_grid.shape, surface_temperature),
    )
    stepped, _ = model.step(carry, coupling_time(0))

    air = np.asarray(carry["state"].mean_air_temperature, dtype=np.float64)
    wind_speed = float(model.params.initial_zonal_wind)
    expected_flux = (
        constants.surface_air_density
        * constants.bulk_drag_coefficient
        * wind_speed
        * jcm_constants.cpd
        * (surface_temperature - air)
    )
    expected = air + TIMESTEP / (
        constants.atmosphere_column_mass * jcm_constants.cpd
    ) * expected_flux

    np.testing.assert_allclose(
        np.asarray(stepped["state"].mean_air_temperature, dtype=np.float64),
        expected,
        rtol=1e-5,
    )


def test_initial_state_follows_the_parameters(half_land_grid):
    """The initial condition is a parameter, not a hard-wired profile."""
    params = SlabAtmosphereParameters(
        initial_temperature_base=250.0,
        initial_temperature_amplitude=0.0,
        initial_zonal_wind=3.0,
        initial_meridional_wind=-4.0,
    )
    carry = SlabAtmosphereModel(half_land_grid, params).initialize()

    np.testing.assert_allclose(
        np.asarray(carry["state"].mean_air_temperature), 250.0, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(carry["state"].mean_zonal_wind_velocity), 3.0
    )
    np.testing.assert_allclose(
        np.asarray(carry["state"].mean_meridional_wind_velocity), -4.0
    )


def test_params_default_equivalence(half_land_grid):
    """Constructing with no params is constructing with the defaults."""
    implicit = SlabAtmosphereModel(half_land_grid).initialize()
    explicit = SlabAtmosphereModel(
        half_land_grid, SlabAtmosphereParameters.default()
    ).initialize()

    assert jax.tree_util.tree_structure(implicit) == jax.tree_util.tree_structure(
        explicit
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_state_has_no_clock(half_land_grid):
    """State carries no time; only the timestep on the clock reaches the physics."""
    model = SlabAtmosphereModel(half_land_grid)
    carry = model.initialize()
    assert "sim_time" not in carry["state"].asdict()

    carry["forcing"] = carry["forcing"].replace(
        land_surface_temperature=jnp.full(half_land_grid.shape, 300.0),
        sea_surface_temperature=jnp.full(half_land_grid.shape, 300.0),
    )
    # The model has no seasonal cycle, so the date must not matter...
    first, _ = model.step(carry, coupling_time(0))
    later, _ = model.step(carry, coupling_time(182))
    np.testing.assert_array_equal(
        np.asarray(first["state"].mean_air_temperature),
        np.asarray(later["state"].mean_air_temperature),
    )

    # ... but the length of the step must, and it comes from the clock too.
    half_step, _ = model.step(carry, coupling_time(0, dt=TIMESTEP / 2))
    assert float(
        jnp.mean(half_step["state"].mean_air_temperature)
    ) < float(jnp.mean(first["state"].mean_air_temperature))


def test_step_shapes_and_dtypes_stable(half_land_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabAtmosphereModel(half_land_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_step_is_differentiable(half_land_grid):
    """Reverse-mode gradients of one step agree with finite differences."""
    model = SlabAtmosphereModel(half_land_grid)
    carry = model.initialize()

    def mean_air_temperature(surface_temperature_scale):
        # One unit is 100 K of surface temperature anomaly, which moves the air
        # temperature by ~10 K: a float32 finite difference of a ~280 K field
        # cannot resolve much less.
        surface = 288.0 + 100.0 * surface_temperature_scale
        forced = carry["forcing"].replace(
            land_surface_temperature=jnp.full(half_land_grid.shape, surface),
            sea_surface_temperature=jnp.full(half_land_grid.shape, surface),
        )
        stepped, _ = model.step({**carry, "forcing": forced}, coupling_time(0))
        return jnp.mean(stepped["state"].mean_air_temperature)

    check_grads(
        mean_air_temperature,
        (0.0,),
        order=1,
        modes=["rev"],
        eps=1e-2,
        atol=1e-2,
        rtol=1e-2,
    )


def test_drag_coefficient_is_differentiable(half_land_grid):
    """The drag coefficient travels in the forcing, so an exchanger can tune it."""
    model = SlabAtmosphereModel(half_land_grid)

    def mean_air_temperature(drag):
        carry = model.initialize()
        carry["forcing"] = carry["forcing"].replace(
            bulk_drag_coefficient=drag,
            land_surface_temperature=jnp.full(half_land_grid.shape, 320.0),
            sea_surface_temperature=jnp.full(half_land_grid.shape, 320.0),
        )
        carry, _ = model.step(carry, coupling_time(0))
        return jnp.mean(carry["state"].mean_air_temperature)

    gradient = jax.grad(mean_air_temperature)(jnp.float32(1e-3))
    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 0.0
