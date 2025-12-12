"""Shared pytest fixtures for jax_esm tests."""

import pytest
import jax.numpy as jnp
import jax_datetime as jdt


@pytest.fixture
def grid_shape_2d():
    """Standard 2D grid shape for testing (lat, lon)."""
    return (48, 96)


@pytest.fixture
def grid_shape_small():
    """Small 2D grid shape for fast tests."""
    return (8, 16)


@pytest.fixture
def sample_temperature_field(grid_shape_small):
    """Sample temperature field in Kelvin."""
    lat_size, lon_size = grid_shape_small
    # Create a simple latitudinal gradient
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, lat_size)
    temp = 288.0 + 30.0 * jnp.cos(lat)  # Warm at equator, cold at poles
    return jnp.broadcast_to(temp[:, None], grid_shape_small)


@pytest.fixture
def sample_heat_flux(grid_shape_small):
    """Sample heat flux field in W/m²."""
    return jnp.ones(grid_shape_small) * 50.0  # 50 W/m² uniform flux


@pytest.fixture
def start_datetime():
    """Standard start datetime for tests."""
    return jdt.to_datetime("2000-01-01")


@pytest.fixture
def coupling_timestep():
    """Standard coupling timestep (1 day in seconds)."""
    return 86400.0


@pytest.fixture
def ocean_mask(grid_shape_small):
    """Binary mask where 0=ocean, 1=land."""
    lat_size, lon_size = grid_shape_small
    # Simple mask: land in top quarter of domain
    mask = jnp.zeros(grid_shape_small, dtype=jnp.int32)
    mask = mask.at[: lat_size // 4, :].set(1)
    return mask


@pytest.fixture
def fractional_ocean_mask(grid_shape_small):
    """Fractional mask where 0=ocean, 1=land, values in between for coasts."""
    lat_size, lon_size = grid_shape_small
    mask = jnp.zeros(grid_shape_small, dtype=jnp.float32)
    mask = mask.at[: lat_size // 4, :].set(1.0)
    mask = mask.at[lat_size // 4, :].set(0.5)  # Coastal cells
    return mask
