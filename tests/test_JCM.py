"""Tests for component interface."""

import jax
import jax.numpy as jnp
import pytest
import unittest

from jax_esm.components.JCM import JCM
from jax_esm.components.base import ComponentConfig

import jcm
from pathlib import Path

class TestJCM(unittest.TestCase):
    
    """Test component interface."""
     
    def test_JCM_initialization(self):
        """Test component initialization."""
        model = JCM.generate_default_model()
 
    def test_JCM_initialization_with_layers(self):
        """Test component initialization with vertical layers."""
        model = JCM.generate_default_model(layers=8)
        assert model.model.coords.nodal_shape[0] == 8
 
    def test_JCM_initialization_with_layers_and_topo_file(self):
        """Test component initialization with vertical layers and topography file."""
        model = JCM.generate_default_model(layers=8, topo_file=Path(jcm.__file__).parent / "data" / "bc" / "t30" / "clim" / "boundaries.nc")
        assert model.model.coords.nodal_shape[0] == 8
         
    def test_slab_ocean_model_initialize_state(self):
        """Test state initialization."""
        model = JCM.generate_default_model()
        
        state = model.initialize()
        assert hasattr(state.prog, "temperature")
        assert state.prog.temperature.shape == (8, 96, 48)
        
    def test_component_step(self):
        """Test component stepping."""
        model = JCM.generate_default_model()
        
        init_state = model.initialize()
      
        # Upward positive, so we are heating up the ocean 
        SST = 300.0
        forcing = model.component_forcing_class.zeros()
        assert hasattr(forcing, "flux")
        assert hasattr(forcing, "scalar")
        assert hasattr(forcing.scalar, "sea_surface_skin_temperature")

        forcing = forcing.copy(
            scalar_kwargs = dict(
                sea_surface_skin_temperature = forcing.scalar.sea_surface_skin_temperature * 0 + SST,
            ),
        )
        
        assert jnp.allclose(forcing.scalar.sea_surface_skin_temperature, SST)
        
        step_function = model.generate_step_function() 
        new_state, predictions = step_function(init_state, forcing, 0.0)
        assert predictions.physics.surface_flux.tskin.shape[0] == model.config.substeps
        assert jnp.allclose(predictions.physics.surface_flux.tskin, SST)

    
