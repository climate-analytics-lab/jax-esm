"""Tests for component interface.

DEPRECATED: These tests use the old API and are kept for reference only.
All tests are skipped. See test_coupler.py, test_slab_ocean.py, and test_flux_model.py
for current API tests.
"""

import jax
import jax.numpy as jnp
import pytest
import unittest

# Old API - no longer exists
# from jem import Component, ComponentConfig, ComponentState, BoundaryFluxes

pytestmark = pytest.mark.skip(reason="Tests use deprecated API - see new tests")

# Skip entire module - don't even try to define classes
if False:

    class MockComponent(Component):
        """Mock component for testing."""

    def initialize(self, rng_key):
        shape = (10, 20)
        return ComponentState(
            prognostic={"temperature": jnp.ones(shape) * 280.0},
            diagnostic={"precipitation": jnp.zeros(shape)},
            boundary={"surface_temp": jnp.ones(shape) * 288.0},
            forcing={"solar": jnp.ones(shape) * 340.0},
            metadata={"time": 0.0},
        )

    def step(self, state, forcing, dt):
        # Simple update
        new_temp = state.prognostic["temperature"] + 0.1 * dt

        new_state = ComponentState(
            prognostic={"temperature": new_temp},
            diagnostic=state.diagnostic,
            boundary=state.boundary,
            forcing=state.forcing,
            metadata={"time": state.metadata["time"] + dt},
        )

        output_fluxes = BoundaryFluxes(
            heat=jnp.ones((10, 20)) * 100.0,
            moisture=jnp.ones((10, 20)) * 0.001,
            momentum_u=jnp.zeros((10, 20)),
            momentum_v=jnp.zeros((10, 20)),
            tracers={},
        )

        return new_state, output_fluxes

    def compute_tendencies(self, state, forcing):
        return {"temperature": jnp.ones_like(state.prognostic["temperature"]) * 0.1}


class TestComponent(unittest.TestCase):
    """Test component interface."""

    def test_component_initialization(self):
        """Test component initialization."""
        config = ComponentConfig(
            name="test",
            timestep=1800.0,
            grid={"type": "latlon", "nlat": 10, "nlon": 20},
            params={"test_param": 42},
        )

        component = MockComponent(config)
        assert component.name == "test"
        assert component.timestep == 1800.0
        assert component.config.params["test_param"] == 42

    def test_component_initialize_state(self):
        """Test state initialization."""
        config = ComponentConfig(
            name="test",
            timestep=1800.0,
            grid={},
            params={},
        )
        component = MockComponent(config)

        rng_key = jax.random.PRNGKey(42)
        state = component.initialize(rng_key)

        assert "temperature" in state.prognostic
        assert state.prognostic["temperature"].shape == (10, 20)
        assert jnp.allclose(state.prognostic["temperature"], 280.0)

    def test_component_step(self):
        """Test component stepping."""
        config = ComponentConfig(
            name="test",
            timestep=1800.0,
            grid={},
            params={},
        )
        component = MockComponent(config)

        rng_key = jax.random.PRNGKey(42)
        state = component.initialize(rng_key)

        forcing = BoundaryFluxes(
            heat=jnp.zeros((10, 20)),
            moisture=jnp.zeros((10, 20)),
            momentum_u=jnp.zeros((10, 20)),
            momentum_v=jnp.zeros((10, 20)),
            tracers={},
        )

        new_state, output_fluxes = component.step(state, forcing, 1800.0)

        # Check state update
        assert new_state.metadata["time"] == 1800.0
        expected_temp = 280.0 + 0.1 * 1800.0
        assert jnp.allclose(new_state.prognostic["temperature"], expected_temp)

        # Check output fluxes
        assert jnp.allclose(output_fluxes.heat, 100.0)
        assert jnp.allclose(output_fluxes.moisture, 0.001)

    def test_component_tendencies(self):
        """Test tendency computation."""
        config = ComponentConfig(
            name="test",
            timestep=1800.0,
            grid={},
            params={},
        )
        component = MockComponent(config)

        rng_key = jax.random.PRNGKey(42)
        state = component.initialize(rng_key)

        forcing = BoundaryFluxes(
            heat=jnp.zeros((10, 20)),
            moisture=jnp.zeros((10, 20)),
            momentum_u=jnp.zeros((10, 20)),
            momentum_v=jnp.zeros((10, 20)),
            tracers={},
        )

        tendencies = component.compute_tendencies(state, forcing)

        assert "temperature" in tendencies
        assert jnp.allclose(tendencies["temperature"], 0.1)

    def test_boundary_fields(self):
        """Test boundary field extraction."""
        config = ComponentConfig(
            name="test",
            timestep=1800.0,
            grid={},
            params={},
        )
        component = MockComponent(config)

        rng_key = jax.random.PRNGKey(42)
        state = component.initialize(rng_key)

        boundary_fields = component.get_boundary_fields(state)
        assert "surface_temp" in boundary_fields
        assert jnp.allclose(boundary_fields["surface_temp"], 288.0)

    def test_flux_requirements(self):
        """Test flux requirement declarations."""
        config = ComponentConfig(
            name="test",
            timestep=1800.0,
            grid={},
            params={},
        )
        component = MockComponent(config)

        required = component.get_required_fluxes()
        provided = component.get_provided_fluxes()

        assert "heat" in required
        assert "moisture" in required
        assert "heat" in provided
        assert "moisture" in provided
