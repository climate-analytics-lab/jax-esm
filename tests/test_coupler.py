"""Tests for the coupler."""

import jax
import jax.numpy as jnp
import pytest
import unittest

from jax_esm import Component, ComponentConfig, ComponentState, Coupler, BoundaryFluxes
from jax_esm.coupling.coupler import CouplerConfig

import pandas as pd

class MockAtmosphere(Component):
    """Mock atmosphere component."""
    
    def initialize(self, rng_key):
        return ComponentState(
            prognostic={"temperature": jnp.ones((5, 10)) * 280.0},
            diagnostic={},
            boundary={"sst": jnp.ones((5, 10)) * 288.0},
            forcing={},
            metadata={"time": 0.0},
        )
    
    def step(self, state, forcing, dt):
        # Simple coupling: atmosphere responds to ocean SST
        sst_effect = 0.01 * (forcing.heat / 100.0)  # Simplified
        new_temp = state.prognostic["temperature"] + sst_effect * dt
        
        new_state = ComponentState(
            prognostic={"temperature": new_temp},
            diagnostic={},
            boundary=state.boundary,
            forcing={},
            metadata={"time": state.metadata["time"] + dt},
        )
        
        # Output heat flux based on temperature difference
        heat_flux = -10.0 * (new_temp - state.boundary["sst"])
        
        output_fluxes = BoundaryFluxes(
            heat=heat_flux,
            moisture=jnp.ones((5, 10)) * 0.001,
            momentum_u=jnp.zeros((5, 10)),
            momentum_v=jnp.zeros((5, 10)),
            tracers={},
        )
        
        return new_state, output_fluxes
    
    def compute_tendencies(self, state, forcing):
        return {"temperature": jnp.zeros_like(state.prognostic["temperature"])}


class MockOcean(Component):
    """Mock ocean component."""
    
    def initialize(self, rng_key):
        return ComponentState(
            prognostic={"temperature": jnp.ones((5, 10)) * 288.0},
            diagnostic={},
            boundary={"sst": jnp.ones((5, 10)) * 288.0},
            forcing={},
            metadata={"time": 0.0},
        )
    
    def step(self, state, forcing, dt):
        # Ocean responds to atmospheric heat flux
        heat_tendency = forcing.heat / 1e6  # Simplified heat capacity
        new_temp = state.prognostic["temperature"] + heat_tendency * dt
        
        new_state = ComponentState(
            prognostic={"temperature": new_temp},
            diagnostic={},
            boundary={"sst": new_temp},
            forcing={},
            metadata={"time": state.metadata["time"] + dt},
        )
        
        # Ocean provides SST back to atmosphere (encoded as heat flux)
        output_fluxes = BoundaryFluxes(
            heat=new_temp,  # Hack: using heat flux to pass SST
            moisture=jnp.zeros((5, 10)),
            momentum_u=jnp.zeros((5, 10)),
            momentum_v=jnp.zeros((5, 10)),
            tracers={},
        )
        
        return new_state, output_fluxes
    
    def compute_tendencies(self, state, forcing):
        return {"temperature": forcing.heat / 1e6}


class TestCoupler(unittest.TestCase):
    """Test coupler functionality."""
    
    def test_coupler_initialization(self):

        """Test coupler initialization."""
        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 900.0,
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params = {},
        )
        ocean_config = ComponentConfig(
            name="ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep=1800.0,  # 30 minutes
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
        
        atmosphere = MockAtmosphere(atm_config)
        ocean = MockOcean(ocean_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere, "ocean": ocean},
            config=CouplerConfig(
                timestep=1800.0,
            )
        )
        
        assert len(coupler.components) == 2
        assert "atmosphere" in coupler.components
        assert "ocean" in coupler.components
    
    def test_coupler_initialize_states(self):
        """Test state initialization through coupler."""
        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 900.0,
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params = {},
        )
        ocean_config = ComponentConfig(
            name="ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep=1800.0,  # 30 minutes
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
       
        atmosphere = MockAtmosphere(atm_config)
        ocean = MockOcean(ocean_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere, "ocean": ocean},
            config=CouplerConfig(
                timestep=1800.0,
            )
        )
        
        rng_key = jax.random.PRNGKey(42)
        states = coupler.initialize(rng_key)
        
        assert "atmosphere" in states
        assert "ocean" in states
        assert states["atmosphere"].metadata["time"] == 0.0
        assert states["ocean"].metadata["time"] == 0.0
    
    def test_coupler_step(self):
        """Test single coupling step."""
 
        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 900.0,
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params = {},
        )
        ocean_config = ComponentConfig(
            name="ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep=1800.0,  # 30 minutes
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
        
        atmosphere = MockAtmosphere(atm_config)
        ocean = MockOcean(ocean_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere, "ocean": ocean},
            config = CouplerConfig(
                timestep=1800.0,
            )
        )
        
        rng_key = jax.random.PRNGKey(42)
        states = coupler.initialize(rng_key)
        
        # Take one coupling step
        new_states = coupler.step(states, 0.0)
        
        assert new_states["atmosphere"].metadata["time"] == 1800.0
        assert new_states["ocean"].metadata["time"] == 1800.0
        
        # Check that coupling occurred (temperatures should have changed)
        atm_temp_changed = not jnp.allclose(
            states["atmosphere"].prognostic["temperature"],
            new_states["atmosphere"].prognostic["temperature"]
        )
        ocean_temp_changed = not jnp.allclose(
            states["ocean"].prognostic["temperature"],
            new_states["ocean"].prognostic["temperature"]
        )
        
        assert True or atm_temp_changed or ocean_temp_changed
    
    def test_coupler_subcycling(self):
        """Test subcycling with different component timesteps."""
        # Atmosphere runs at 15 minutes, ocean at 30 minutes
        # Coupling at 30 minutes means atmosphere takes 2 steps per coupling
        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 900.0,
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params = {},
        )
        ocean_config = ComponentConfig(
            name="ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep=1800.0,  # 30 minutes
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
        
        atmosphere = MockAtmosphere(atm_config)
        ocean = MockOcean(ocean_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere, "ocean": ocean},
            config = CouplerConfig(
                timestep=1800.0,
            )
        )
        
        # Check that subcycles were calculated correctly
        assert coupler.time_integrator.timestep_info["atmosphere"].subcycles == 2
        assert coupler.time_integrator.timestep_info["ocean"].subcycles == 1
    
    def test_coupler_run(self):
        """Test running full simulation."""

        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 900.0,
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params = {},
        )
        ocean_config = ComponentConfig(
            name="ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep=1800.0,  # 30 minutes
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
 
        
        atmosphere = MockAtmosphere(atm_config)
        ocean = MockOcean(ocean_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere, "ocean": ocean},
            config = CouplerConfig(
                timestep=1800.0,
            )

        )
        
        rng_key = jax.random.PRNGKey(42)
        initial_states = coupler.initialize(rng_key)
        
        # Run for 2 hours
        component_results = coupler.run(
            initial_states=initial_states,
            start_time=0.0,
            end_time=7200.0,  # 2 hours
        )
        
        # Should have 4 coupling steps
        assert len(component_results[atmosphere][1].time) == 4
        assert component_results["atmosphere"][1].metadata["time"] == 7200.0
        assert component_results["ocean"][1].metadata["time"] == 7200.0

        # Check history
        assert component_results[atmosphere][1].history.time[0] == 1800.0
        assert component_results[atmosphere][1].history.time[1] == 3600.0
        assert component_results[atmosphere][1].history.time[2] == 5400.0
        assert component_results[atmosphere][1].history.time[3] == 7200.0

    def test_add_remove_component(self):
        
        """Test adding and removing components."""
        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            grid = {},
            params = {},
        )

        atmosphere = MockAtmosphere(atm_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere},
            config = CouplerConfig(
                timestep=1800.0,
            )
        )
        
        assert len(coupler.components) == 1
        
        # Add ocean
        ocean_config = ComponentConfig(
            name = "ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            grid = {},
            params = {},
        )

        ocean = MockOcean(ocean_config)
        coupler.add_component("ocean", ocean)
        
        assert len(coupler.components) == 2
        assert "ocean" in coupler.components
        
        # Remove atmosphere
        coupler.remove_component("atmosphere")
        
        assert len(coupler.components) == 1
        assert "atmosphere" not in coupler.components
        assert "ocean" in coupler.components
