"""Tests for `jem.components.slab.slab_land_model`."""

import jax.numpy as jnp
import numpy as np
import pytest

from jem.components.slab.grid import generate_slab_grid
from jem.components.slab.slab_land_model import SlabLandModel


@pytest.fixture
def t31_grid():
    """Build a T31 JCM grid whose eastern half is land, so both mask branches run."""
    shape = generate_slab_grid("JCM::T31").shape
    n_lon = shape[0]
    fractional_mask = jnp.where(
        jnp.arange(n_lon)[:, None] >= n_lon // 2,
        jnp.ones(shape),
        jnp.zeros(shape),
    )
    return generate_slab_grid("JCM::T31", fractional_mask=fractional_mask)


def test_idealized_climatology_varies_with_latitude_only(t31_grid):
    """The idealised climatology must vary with latitude, not longitude.

    The regression this guards: the profile used to be laid out along the
    longitude axis because `grid.shape` was unpacked as `(n_lat, n_lon)`.
    """
    model = SlabLandModel(grid=t31_grid)

    temperature = model._idealized_land_temperature()

    n_lon, n_lat = t31_grid.shape
    assert temperature.shape == (n_lon, n_lat, 12)

    # Standard deviations are taken in float64: a float32 mean of ~10^2
    # values is not bit-exact, so jnp.std of a genuinely constant float32
    # slice returns O(1e-5) rather than zero.
    profile = np.asarray(temperature[:, :, 0], dtype=np.float64)

    # Constant along longitude (axis 0) at every latitude.
    assert np.all(profile.std(axis=0) == 0.0)

    # Genuinely varying along latitude (axis 1) at every longitude.
    assert np.all(profile.std(axis=1) > 1.0)


def test_land_step_runs_and_is_finite(t31_grid):
    """Three unforced steps stay finite and physically plausible."""
    model = SlabLandModel(grid=t31_grid)
    carry = model.initialize()
    step_function = model.generate_step_function()

    for step in range(3):
        carry, _ = step_function(carry, step)

    temperature = carry["state"].land_surface_temperature
    assert bool(jnp.all(jnp.isfinite(temperature)))
    assert float(jnp.min(temperature)) > 150.0
    assert float(jnp.max(temperature)) < 350.0
