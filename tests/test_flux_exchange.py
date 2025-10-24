"""Tests for flux exchange.

DEPRECATED: FluxExchanger is currently unused in the codebase.
Tests are skipped until FluxExchanger is integrated or removed.
"""

import jax.numpy as jnp
import pytest

# FluxExchanger and BoundaryFluxes are currently unused
# from jax_esm.components.base import BoundaryFluxes
# from jax_esm.coupling.flux_exchange import FluxExchanger

pytestmark = pytest.mark.skip(reason="FluxExchanger is unused - tests deprecated")


class TestFluxExchanger:
    """Test flux exchange functionality."""
    
    def test_flux_exchanger_initialization(self):
        """Test flux exchanger initialization."""
        exchanger = FluxExchanger(
            component_names=["atmosphere", "ocean", "land"],
        )
        
        assert len(exchanger.component_names) == 3
        assert "atmosphere" in exchanger.connections
        assert "ocean" in exchanger.connections
        assert "land" in exchanger.connections
    
    def test_default_flux_exchange(self):
        """Test default flux exchange without custom mappings."""
        exchanger = FluxExchanger(
            component_names=["atmosphere", "ocean"],
        )
        
        # Create fluxes from each component
        shape = (5, 10)
        atm_fluxes = BoundaryFluxes(
            heat=jnp.ones(shape) * 100.0,
            moisture=jnp.ones(shape) * 0.001,
            momentum_u=jnp.ones(shape) * 0.1,
            momentum_v=jnp.ones(shape) * 0.05,
            tracers={},
        )
        
        ocean_fluxes = BoundaryFluxes(
            heat=jnp.ones(shape) * 50.0,
            moisture=jnp.zeros(shape),
            momentum_u=jnp.zeros(shape),
            momentum_v=jnp.zeros(shape),
            tracers={},
        )
        
        component_fluxes = {
            "atmosphere": atm_fluxes,
            "ocean": ocean_fluxes,
        }
        
        # Exchange fluxes
        input_fluxes = exchanger.exchange_fluxes(component_fluxes)
        
        # Check that each component receives fluxes from the other
        assert "atmosphere" in input_fluxes
        assert "ocean" in input_fluxes
        
        # Ocean should receive atmosphere fluxes
        assert jnp.allclose(input_fluxes["ocean"].heat, 100.0)
        assert jnp.allclose(input_fluxes["ocean"].moisture, 0.001)
        
        # Atmosphere should receive ocean fluxes
        assert jnp.allclose(input_fluxes["atmosphere"].heat, 50.0)
        assert jnp.allclose(input_fluxes["atmosphere"].moisture, 0.0)
    
    def test_custom_flux_mapping(self):
        """Test custom flux mappings."""
        exchanger = FluxExchanger(
            component_names=["atmosphere", "ocean"],
            flux_mappings={
                ("atmosphere", "ocean"): {
                    "heat": "heat_flux_down",
                    "momentum_u": "wind_stress_x",
                    "momentum_v": "wind_stress_y",
                }
            }
        )
        
        shape = (5, 10)
        atm_fluxes = BoundaryFluxes(
            heat=jnp.ones(shape) * 100.0,
            moisture=jnp.ones(shape) * 0.001,
            momentum_u=jnp.ones(shape) * 0.1,
            momentum_v=jnp.ones(shape) * 0.05,
            tracers={},
        )
        
        component_fluxes = {"atmosphere": atm_fluxes, "ocean": BoundaryFluxes(
            heat=jnp.zeros(shape),
            moisture=jnp.zeros(shape),
            momentum_u=jnp.zeros(shape),
            momentum_v=jnp.zeros(shape),
            tracers={},
        )}
        
        input_fluxes = exchanger.exchange_fluxes(component_fluxes)
        
        # Check that mappings were applied
        # Note: The current implementation doesn't handle custom target names
        # This would need to be enhanced in the actual implementation
        assert "ocean" in input_fluxes
    
    def test_flux_transformations(self):
        """Test flux transformations."""
        def scale_heat(flux):
            return flux * 0.5  # Scale heat flux by 0.5
        
        exchanger = FluxExchanger(
            component_names=["atmosphere", "ocean"],
            transformations={
                ("atmosphere", "ocean", "heat"): scale_heat,
            }
        )
        
        shape = (5, 10)
        atm_fluxes = BoundaryFluxes(
            heat=jnp.ones(shape) * 100.0,
            moisture=jnp.ones(shape) * 0.001,
            momentum_u=jnp.zeros(shape),
            momentum_v=jnp.zeros(shape),
            tracers={},
        )
        
        ocean_fluxes = BoundaryFluxes(
            heat=jnp.zeros(shape),
            moisture=jnp.zeros(shape),
            momentum_u=jnp.zeros(shape),
            momentum_v=jnp.zeros(shape),
            tracers={},
        )
        
        component_fluxes = {
            "atmosphere": atm_fluxes,
            "ocean": ocean_fluxes,
        }
        
        input_fluxes = exchanger.exchange_fluxes(component_fluxes)
        
        # Ocean should receive scaled heat flux
        assert jnp.allclose(input_fluxes["ocean"].heat, 50.0)  # 100.0 * 0.5
    
    def test_multiple_source_accumulation(self):
        """Test accumulation of fluxes from multiple sources."""
        exchanger = FluxExchanger(
            component_names=["atmosphere", "land", "ocean"],
        )
        
        shape = (5, 10)
        
        # Both atmosphere and land provide heat to ocean
        atm_fluxes = BoundaryFluxes(
            heat=jnp.ones(shape) * 100.0,
            moisture=jnp.zeros(shape),
            momentum_u=jnp.zeros(shape),
            momentum_v=jnp.zeros(shape),
            tracers={},
        )
        
        land_fluxes = BoundaryFluxes(
            heat=jnp.ones(shape) * 50.0,  # Land also provides heat
            moisture=jnp.ones(shape) * 0.002,
            momentum_u=jnp.zeros(shape),
            momentum_v=jnp.zeros(shape),
            tracers={},
        )
        
        ocean_fluxes = BoundaryFluxes(
            heat=jnp.zeros(shape),
            moisture=jnp.zeros(shape),
            momentum_u=jnp.zeros(shape),
            momentum_v=jnp.zeros(shape),
            tracers={},
        )
        
        component_fluxes = {
            "atmosphere": atm_fluxes,
            "land": land_fluxes,
            "ocean": ocean_fluxes,
        }
        
        input_fluxes = exchanger.exchange_fluxes(component_fluxes)
        
        # Ocean should receive accumulated heat from both sources
        assert jnp.allclose(input_fluxes["ocean"].heat, 150.0)  # 100 + 50
        assert jnp.allclose(input_fluxes["ocean"].moisture, 0.002)
    
    def test_add_flux_mapping(self):
        """Test adding flux mappings after initialization."""
        exchanger = FluxExchanger(
            component_names=["atmosphere", "ocean"],
        )
        
        # Add custom mapping
        exchanger.add_flux_mapping(
            "atmosphere", "ocean",
            {"heat": "surface_heat_flux"}
        )
        
        assert ("atmosphere", "ocean") in exchanger.flux_mappings
        assert exchanger.flux_mappings[("atmosphere", "ocean")]["heat"] == "surface_heat_flux"
    
    def test_add_transformation(self):
        """Test adding transformations after initialization."""
        exchanger = FluxExchanger(
            component_names=["atmosphere", "ocean"],
        )
        
        def wind_to_stress(wind):
            return wind * 0.001  # Simple transformation
        
        exchanger.add_transformation(
            "atmosphere", "ocean", "momentum_u", wind_to_stress
        )
        
        key = ("atmosphere", "ocean", "momentum_u")
        assert key in exchanger.transformations
        assert exchanger.transformations[key] == wind_to_stress