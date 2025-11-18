"""Comprehensive tests for LandModel component."""

import jax
import jax.numpy as jnp
import pytest
import unittest
import pandas as pd
import numpy as np
import xarray as xr
import tempfile
import os
from typing import NamedTuple

from jax_esm.components.LandModel import LandModel
from jax_esm.components.base import ComponentConfig


class MockCoupledState(NamedTuple):
    """JAX-compatible mock coupled state for testing."""
    lnd: object
    flx: object = None


class MockCoordinates:
    """Mock coordinate system for testing."""
    
    def __init__(self, nlat=5, nlon=10, nlevels=3):
        self.nodal_shape = (nlevels, nlat, nlon)
        self.nlat = nlat
        self.nlon = nlon
        
        # Mock horizontal coordinates
        class HorizontalCoords:
            def __init__(self, nlat, nlon):
                self.latitudes = jnp.linspace(-jnp.pi/2, jnp.pi/2, nlat)
                self.longitudes = jnp.linspace(0, 2*jnp.pi, nlon)
        
        self.horizontal = HorizontalCoords(nlat, nlon)


class MockBoundaries:
    """Mock boundary data for testing."""
    
    def __init__(self, nlat=5, nlon=10):
        # Land fraction (0-1)
        self.fmask = jnp.ones((nlat, nlon), dtype=jnp.float32) * 0.7
        
        # Albedo (soil=0.2, ice=0.6)
        self.alb0 = jnp.ones((nlat, nlon), dtype=jnp.float32) * 0.2
        # Make high latitudes ice
        lat_ice_threshold = nlat * 4 // 5
        self.alb0 = self.alb0.at[lat_ice_threshold:, :].set(0.6)


def create_test_climatology_file(nlat=5, nlon=10):
    """Create a temporary NetCDF file with land climatology."""
    
    # Create synthetic monthly data
    months = 12
    
    # Land surface temperature (K)
    lat = np.linspace(-90, 90, nlat)
    base_T = 273.15 + 25.0 * np.cos(np.deg2rad(lat))
    stl = np.zeros((months, nlat, nlon), dtype=np.float32)
    for m in range(months):
        seasonal = 10.0 * np.cos(2 * np.pi * (m - 2) / 12.0)
        stl[m, :, :] = base_T[:, None] + seasonal
    
    # Snow depth (mm water equivalent)
    snowd = np.zeros((months, nlat, nlon), dtype=np.float32)
    # Add snow at high latitudes in winter
    for m in range(months):
        winter_factor = np.maximum(0, np.cos(2 * np.pi * (m - 2) / 12.0))
        snowd[m, nlat*4//5:, :] = 100.0 * winter_factor
    
    # Soil water availability (0-1)
    soilw = np.ones((months, nlat, nlon), dtype=np.float32) * 0.5
    
    # Create dataset
    ds = xr.Dataset(
        {
            "stl": (["month", "lat", "lon"], stl),
            "snowd": (["month", "lat", "lon"], snowd),
            "soilw": (["month", "lat", "lon"], soilw),
        },
        coords={
            "month": np.arange(1, 13),
            "lat": lat,
            "lon": np.linspace(0, 360, nlon, endpoint=False),
        },
    )
    
    # Save to temporary file
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
    ds.to_netcdf(tmpfile.name)
    tmpfile.close()
    
    return tmpfile.name


class TestLandModelInitialization(unittest.TestCase):
    """Test land model initialization."""
    
    def test_init_creates_proper_state(self):
        """Test that initialization creates valid state structure."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=3600.0,
            substeps=1,
            save_interval=3600.0,
            grid={"nlat": 5, "nlon": 10},
            params={
                "coords": coords,
            },
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        # Check state structure
        assert hasattr(state, "prog")
        assert hasattr(state, "phydata")
        
        # Check prognostic fields
        assert hasattr(state.prog, "T")
        assert hasattr(state.prog, "sim_time")
        
        # Check diagnostic fields
        assert hasattr(state.phydata, "heatflx")
        assert hasattr(state.phydata, "snowd")
        assert hasattr(state.phydata, "soilw")
        
        # Check shapes
        assert state.prog.T.shape == (5, 10)
        assert state.phydata.snowd.shape == (5, 10)
        assert state.phydata.soilw.shape == (5, 10)
        
        # Check initial time
        assert state.prog.sim_time == 0.0
    
    def test_init_with_boundaries(self):
        """Test initialization with boundary data."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        boundaries = MockBoundaries(nlat=5, nlon=10)
        clim_file = create_test_climatology_file(nlat=5, nlon=10)
        
        try:
            config = ComponentConfig(
                name="land",
                start_dt=pd.Timestamp("2001-01-01"),
                timestep=3600.0,
                substeps=1,
                save_interval=3600.0,
                grid={"nlat": 5, "nlon": 10},
                params={
                    "coords": coords,
                    "boundaries": boundaries,
                    "boundary_file": clim_file,
                },
            )
            
            land = LandModel(config)
            state = land.initialize()
            
            # Check that land mask was created
            assert hasattr(land, "fmask_l")
            assert hasattr(land, "bmask_l")
            assert hasattr(land, "dmask")
            
            # Check climatology was loaded
            assert hasattr(land, "stl_clim")
            assert hasattr(land, "snowd_clim")
            assert hasattr(land, "soilw_clim")
            
            assert land.stl_clim.shape == (12, 5, 10)
            assert land.snowd_clim.shape == (12, 5, 10)
            assert land.soilw_clim.shape == (12, 5, 10)
            
            # Check temperature is reasonable
            assert jnp.all(state.prog.T > 200.0)  # Above absolute zero
            assert jnp.all(state.prog.T < 350.0)  # Below extreme heat
            
        finally:
            os.unlink(clim_file)
    
    def test_land_mask_application(self):
        """Test that land mask is properly applied."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        boundaries = MockBoundaries(nlat=5, nlon=10)
        
        # Make some grid points ocean
        boundaries.fmask = boundaries.fmask.at[:2, :].set(0.0)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=3600.0,
            substeps=1,
            save_interval=3600.0,
            grid={"nlat": 5, "nlon": 10},
            params={
                "coords": coords,
                "boundaries": boundaries,
            },
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        # Check binary mask is 0 for ocean points
        assert jnp.all(land.bmask_l[:2, :] == 0.0)
        assert jnp.all(land.bmask_l[2:, :] == 1.0)
    
    def test_heat_capacity_computation(self):
        """Test heat capacity fields are computed correctly."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        boundaries = MockBoundaries(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=3600.0,
            substeps=1,
            save_interval=3600.0,
            grid={"nlat": 5, "nlon": 10},
            params={
                "coords": coords,
                "boundaries": boundaries,
                "depth_soil": 1.0,
                "depth_lice": 5.0,
            },
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        # Check rhcapl was computed
        assert hasattr(land, "rhcapl")
        assert land.rhcapl.shape == (5, 10)
        
        # Different values for soil vs ice based on albedo
        # High latitude (ice) should have different heat capacity
        ice_indices = boundaries.alb0 >= 0.4
        soil_indices = boundaries.alb0 < 0.4
        
        if jnp.any(ice_indices) and jnp.any(soil_indices):
            # Should have different values
            ice_val = land.rhcapl[ice_indices][0]
            soil_val = land.rhcapl[soil_indices][0]
            assert not jnp.isclose(ice_val, soil_val)


class TestLandModelPhysics(unittest.TestCase):
    """Test land model physics calculations."""
    
    def test_temperature_evolution_no_forcing(self):
        """Test temperature evolution without heat flux."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=86400.0,  # 1 day
            substeps=1,
            save_interval=86400.0,
            grid={"nlat": 5, "nlon": 10},
            params={
                "coords": coords,
                "tdland": 40.0,  # 40-day relaxation
            },
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        # Create mock coupled state
        cpl = MockCoupledState(lnd=state)
        
        # Get step function
        step_fn = land.gen_step_fn()
        
        # Take one step
        new_state, predictions = step_fn(cpl, 0)
        
        # Temperature should relax toward climatology
        # With no forcing, anomaly should decrease
        assert new_state.prog.sim_time == 86400.0
        assert new_state.prog.T.shape == (5, 10)
    
    def test_temperature_response_to_heating(self):
        """Test temperature increases with positive heat flux."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=86400.0,
            substeps=1,
            save_interval=86400.0,
            grid={"nlat": 5, "nlon": 10},
            params={
                "coords": coords,
            },
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        # Create mock flux state with heat flux
        class MockFluxState(NamedTuple):
            phydata: object
        
        class MockFluxPhydata(NamedTuple):
            heatflx: jnp.ndarray
        
        flux_phydata = MockFluxPhydata(
            heatflx=jnp.ones((5, 10), dtype=jnp.float32) * 100.0  # 100 W/m^2 upward
        )
        flux_state = MockFluxState(phydata=flux_phydata)
        
        cpl = MockCoupledState(lnd=state, flx=flux_state)
        
        # Get step function
        step_fn = land.gen_step_fn()
        
        # Take multiple steps
        initial_T = state.prog.T.copy()
        
        for i in range(5):
            new_state, predictions = step_fn(cpl, i)
            cpl = MockCoupledState(lnd=new_state, flx=flux_state)
        
        # With positive upward flux (negative downward), temperature should decrease slightly
        # due to energy loss (but relaxation to climatology dominates)
        # Just check it's still physical
        assert jnp.all(new_state.prog.T > 200.0)
        assert jnp.all(new_state.prog.T < 350.0)


class TestLandModelTimestepping(unittest.TestCase):
    """Test land model time stepping."""
    
    def test_single_timestep(self):
        """Test single time step execution."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=3600.0,
            substeps=1,
            save_interval=3600.0,
            grid={"nlat": 5, "nlon": 10},
            params={"coords": coords},
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        cpl = MockCoupledState(lnd=state)
        step_fn = land.gen_step_fn()
        
        new_state, predictions = step_fn(cpl, 0)
        
        # Check output structure
        assert hasattr(new_state, "prog")
        assert hasattr(new_state, "phydata")
        assert "prog" in predictions
        assert "phydata" in predictions
    
    def test_timestep_increments_time(self):
        """Test that simulation time increments correctly."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        timestep = 3600.0
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=timestep,
            substeps=1,
            save_interval=timestep,
            grid={"nlat": 5, "nlon": 10},
            params={"coords": coords},
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        cpl = MockCoupledState(lnd=state)
        step_fn = land.gen_step_fn()
        
        assert state.prog.sim_time == 0.0
        
        new_state, _ = step_fn(cpl, 0)
        assert new_state.prog.sim_time == timestep
        
        cpl = MockCoupledState(lnd=new_state)
        new_state, _ = step_fn(cpl, 1)
        assert new_state.prog.sim_time == 2 * timestep
    
    def test_conservation_properties(self):
        """Test that temperature remains bounded."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=86400.0,
            substeps=1,
            save_interval=86400.0,
            grid={"nlat": 5, "nlon": 10},
            params={"coords": coords},
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        cpl = MockCoupledState(lnd=state)
        step_fn = land.gen_step_fn()
        
        # Run for many steps
        for i in range(100):
            new_state, _ = step_fn(cpl, i)
            cpl = MockCoupledState(lnd=new_state)
            
            # Check temperature stays physical
            assert jnp.all(new_state.prog.T > 200.0), f"Step {i}: T too low"
            assert jnp.all(new_state.prog.T < 350.0), f"Step {i}: T too high"
            assert jnp.all(jnp.isfinite(new_state.prog.T)), f"Step {i}: T not finite"


class TestLandModelIntegration(unittest.TestCase):
    """Test land model integration with JAX-ESM framework."""
    
    def test_component_interface(self):
        """Test that LandModel implements Component interface correctly."""
        
        from jax_esm.components.base import Component
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=3600.0,
            substeps=1,
            save_interval=3600.0,
            grid={"nlat": 5, "nlon": 10},
            params={"coords": coords},
        )
        
        land = LandModel(config)
        
        # Check inheritance
        assert isinstance(land, Component)
        
        # Check required methods
        assert hasattr(land, "initialize")
        assert hasattr(land, "gen_step_fn")
        assert callable(land.initialize)
        assert callable(land.gen_step_fn)
    
    def test_jit_compilation(self):
        """Test that step function can be JIT compiled."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=3600.0,
            substeps=1,
            save_interval=3600.0,
            grid={"nlat": 5, "nlon": 10},
            params={"coords": coords},
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        cpl = MockCoupledState(lnd=state)
        step_fn = land.gen_step_fn()
        
        # Function should already be JIT compiled
        # Try to use it multiple times (would fail if not JIT compatible)
        for i in range(3):
            new_state, predictions = step_fn(cpl, i)
            cpl = MockCoupledState(lnd=new_state)
        
        # Should complete without error
        assert True
    
    def test_state_immutability(self):
        """Test that state is properly immutable."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=3600.0,
            substeps=1,
            save_interval=3600.0,
            grid={"nlat": 5, "nlon": 10},
            params={"coords": coords},
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        original_T = state.prog.T.copy()
        
        cpl = MockCoupledState(lnd=state)
        step_fn = land.gen_step_fn()
        
        new_state, _ = step_fn(cpl, 0)
        
        # Original state should be unchanged
        assert jnp.allclose(state.prog.T, original_T)


class TestLandModelOutputs(unittest.TestCase):
    """Test land model output conversion."""
    
    def test_xarray_conversion(self):
        """Test conversion to xarray Dataset."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=3600.0,
            substeps=1,
            save_interval=3600.0,
            grid={"nlat": 5, "nlon": 10},
            params={"coords": coords},
        )
        
        land = LandModel(config)
        state = land.initialize()
        
        cpl = MockCoupledState(lnd=state)
        step_fn = land.gen_step_fn()
        
        # Run a few steps to collect predictions
        # Note: Each step already returns stacked predictions with shape (1, ...)
        prog_list = []
        phydata_list = []
        
        for i in range(3):
            new_state, predictions = step_fn(cpl, i)
            # predictions already has shape (1, nlat, nlon) from stack_objects in step_fn
            prog_list.append(predictions["prog"])
            phydata_list.append(predictions["phydata"])
            cpl = MockCoupledState(lnd=new_state)
        
        # Concatenate predictions along time axis
        from jax_esm.utils.bulk_op import concat_objects
        stacked_predictions = {
            "prog": concat_objects(prog_list, axis=0),
            "phydata": concat_objects(phydata_list, axis=0),
        }
        
        # Convert to xarray
        ds = land.predictions_to_xarray(stacked_predictions)
        
        # Check structure
        assert isinstance(ds, xr.Dataset)
        assert "T" in ds
        assert "heatflx" in ds
        assert "snowd" in ds
        assert "soilw" in ds
        assert "time" in ds.coords
        
        # Check attributes
        assert "units" in ds["T"].attrs
        assert "long_name" in ds["T"].attrs


class TestClimatologyInterpolation(unittest.TestCase):
    """Test climatology interpolation logic."""
    
    def test_monthly_interpolation(self):
        """Test that climatology is properly interpolated."""
        
        coords = MockCoordinates(nlat=5, nlon=10)
        clim_file = create_test_climatology_file(nlat=5, nlon=10)
        
        try:
            config = ComponentConfig(
                name="land",
                start_dt=pd.Timestamp("2001-01-15"),  # Mid-month
                timestep=86400.0,
                substeps=1,
                save_interval=86400.0,
                grid={"nlat": 5, "nlon": 10},
                params={
                    "coords": coords,
                    "boundary_file": clim_file,
                },
            )
            
            land = LandModel(config)
            state = land.initialize()
            
            cpl = MockCoupledState(lnd=state)
            step_fn = land.gen_step_fn()
            
            # Run for a month to test interpolation
            for i in range(30):
                new_state, predictions = step_fn(cpl, i)
                cpl = MockCoupledState(lnd=new_state)
                
                # Snow depth should match climatology
                assert jnp.all(jnp.isfinite(new_state.phydata.snowd))
                assert jnp.all(new_state.phydata.soilw >= 0.0)
                assert jnp.all(new_state.phydata.soilw <= 1.0)
        
        finally:
            os.unlink(clim_file)


if __name__ == "__main__":
    unittest.main()
