"""Tests for component interface."""

import jax.numpy as jnp
import unittest

from jax_esm.components.SlabOceanModel import SlabOceanModel


import jax_esm.constants as constants


class TestSlabOceanModel(unittest.TestCase):
    """Test component interface."""

    def test_slab_ocean_model_initialization(self):
        """Test component initialization."""
        model = SlabOceanModel(
            grid_specification = "Veros::4deg"
        )

    def test_slab_ocean_model_initialization_with_topography_file(self):
        """Test component initialization with vertical layers and topography file."""
        model = SlabOceanModel(
            grid_specification = "Veros::4deg"
        )

    def test_slab_ocean_model_initialize_state(self):
        """Test state initialization."""
        model = SlabOceanModel(
            grid_specification = "Veros::4deg"
        )

        state = model.initialize()
        assert hasattr(state.prog, "T")
        assert state.prog.T.shape == (40, 90)

    def test_component_step(self):
        """Test component stepping."""

        model = SlabOceanModel(
            grid_specification = "Veros::4deg",
            relaxation_time = jnp.inf,
        )

        assert jnp.isinf(model.relaxation_time)

        init_state = model.initialize()

        # Upward positive, so we are heating up the ocean
        total_heat_flux = -1000.0
        forcing = model.component_forcing_class.zeros()
        forcing = forcing.copy(
            flux_kwargs=dict(
                total_heat_flux=forcing.flux.total_heat_flux + total_heat_flux
            ),
        )
        assert hasattr(forcing, "flux")
        assert hasattr(forcing, "scalar")
        assert hasattr(forcing.flux, "total_heat_flux")
        assert jnp.allclose(forcing.flux.total_heat_flux, total_heat_flux)

        # Check state update
        expected_temp = init_state.prog.T + model.config.timestep * (
            -total_heat_flux
        ) / (
            init_state.prog.mld
            * constants.ocean_density
            * constants.ocean_specific_heat_capacity
        )

        step_function = model.generate_step_function()

        new_state, predictions = step_function(init_state, forcing, 0.0)

        assert jnp.allclose(new_state.prog.T, expected_temp)
