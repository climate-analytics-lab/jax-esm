"""Tests for FluxModel component."""

import jax
import jax.numpy as jnp
import pytest
import pandas as pd
import xarray as xr

from jax_esm import ComponentConfig

# Skip all tests if jcm is not available (FluxModel depends on JCM structures)
try:
    from jax_esm.components.FluxModel.FluxModel import FluxModel
    HAS_JCM = True
except ImportError:
    HAS_JCM = False
    pytestmark = pytest.mark.skip(reason="JCM not installed - required for FluxModel")


class TestFluxModel:
    """Test FluxModel physics and integration."""

    @pytest.fixture
    def basic_config(self):
        """Create basic flux model configuration."""
        # Create mock coordinates
        from unittest.mock import Mock
        coords = Mock()
        coords.nodal_shape = (8, 32, 64)  # levels, lon, lat

        return ComponentConfig(
            name="flux",
            start_dt=pd.Timestamp("2000-01-01"),
            timestep=3600.0,  # 1 hour
            substeps=1,
            save_interval=3600.0,
            grid={"type": "latlon"},
            params={
                "coords": coords,
            }
        )

    def test_initialization(self, basic_config):
        """Test flux model initialization."""
        flux_model = FluxModel(basic_config)

        assert flux_model.name == "flux"
        assert flux_model.timestep == 3600.0

    def test_initial_state(self, basic_config):
        """Test initial state creation."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        # Check state structure
        assert hasattr(state, 'prog')
        assert hasattr(state, 'phydata')

        # Check prognostic fields
        assert hasattr(state.prog, 'sim_time')
        assert state.prog.sim_time == 0.0

        # Check phydata fields
        assert hasattr(state.phydata, 'heatflx')
        assert state.phydata.heatflx.shape == (32, 64)

        # Initial heat flux should be zero
        assert jnp.allclose(state.phydata.heatflx, 0.0)

    def test_step_function_generation(self, basic_config):
        """Test step function can be generated and is callable."""
        flux_model = FluxModel(basic_config)
        step_fn = flux_model.gen_step_fn()

        assert callable(step_fn)

    def test_heat_flux_computation(self, basic_config):
        """Test heat flux is computed from atmosphere surface fluxes."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        # Create mock coupled state with atmosphere data
        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        # Create mock atmosphere phydata
        atm_phydata = Mock()
        atm_surface_flux = Mock()

        # Create mock hfluxn with last axis for flux components
        # Shape should be (lon, lat, n_flux_components)
        hfluxn_shape = (32, 64, 4)  # 4 flux components
        atm_surface_flux.hfluxn = jnp.ones(hfluxn_shape) * 25.0  # 25 W/m² per component

        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        # Generate and run step function
        step_fn = flux_model.gen_step_fn()
        new_state, predictions = step_fn(cpl, 0)

        # Heat flux should be negative sum of hfluxn (sign convention)
        # Expected: -(4 * 25.0) = -100.0 W/m²
        expected_flux = -100.0
        assert jnp.allclose(new_state.phydata.heatflx, expected_flux)

    def test_time_advancement(self, basic_config):
        """Test that simulation time advances correctly."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        # Create mock coupled state
        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        # Create minimal atmosphere data
        atm_phydata = Mock()
        atm_surface_flux = Mock()
        atm_surface_flux.hfluxn = jnp.zeros((32, 64, 4))
        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        step_fn = flux_model.gen_step_fn()
        new_state, predictions = step_fn(cpl, 0)

        # Time should advance by timestep
        expected_time = state.prog.sim_time + flux_model.config.timestep
        assert jnp.allclose(new_state.prog.sim_time, expected_time)

    def test_multiple_timesteps(self, basic_config):
        """Test multiple timesteps with consistent results."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        # Create atmosphere data
        atm_phydata = Mock()
        atm_surface_flux = Mock()
        atm_surface_flux.hfluxn = jnp.ones((32, 64, 3)) * 10.0
        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        step_fn = flux_model.gen_step_fn()

        # Run 5 timesteps
        times = []
        fluxes = []
        for i in range(5):
            cpl.flx, predictions = step_fn(cpl, i)
            times.append(cpl.flx.prog.sim_time)
            fluxes.append(cpl.flx.phydata.heatflx)

        # Times should increment correctly
        assert jnp.allclose(times[0], 3600.0)
        assert jnp.allclose(times[1], 7200.0)
        assert jnp.allclose(times[4], 18000.0)

        # Fluxes should be consistent (same atmosphere input)
        for flux in fluxes:
            assert jnp.allclose(flux, fluxes[0])

    def test_predictions_structure(self, basic_config):
        """Test predictions output structure."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        atm_phydata = Mock()
        atm_surface_flux = Mock()
        atm_surface_flux.hfluxn = jnp.zeros((32, 64, 4))
        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        step_fn = flux_model.gen_step_fn()
        new_state, predictions = step_fn(cpl, 0)

        # Predictions should be a dict with 'prog' and 'phydata' keys
        assert isinstance(predictions, dict)
        assert 'prog' in predictions
        assert 'phydata' in predictions

        # Should have expected fields
        assert hasattr(predictions['prog'], 'sim_time')
        assert hasattr(predictions['phydata'], 'heatflx')

    def test_xarray_conversion(self, basic_config):
        """Test conversion of predictions to xarray."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        atm_phydata = Mock()
        atm_surface_flux = Mock()
        atm_surface_flux.hfluxn = jnp.zeros((32, 64, 4))
        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        step_fn = flux_model.gen_step_fn()
        new_state, predictions = step_fn(cpl, 0)

        # Convert to xarray
        ds = flux_model.predictions_to_xarray(predictions)

        # Check structure
        assert isinstance(ds, xr.Dataset)
        assert 'heatflx' in ds.data_vars
        assert 'time' in ds.coords

        # Check dimensions
        assert ds['heatflx'].dims == ('time', 'lon', 'lat')

    def test_zero_flux_handling(self, basic_config):
        """Test handling of zero atmospheric fluxes."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        # Zero fluxes from atmosphere
        atm_phydata = Mock()
        atm_surface_flux = Mock()
        atm_surface_flux.hfluxn = jnp.zeros((32, 64, 4))
        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        step_fn = flux_model.gen_step_fn()
        new_state, predictions = step_fn(cpl, 0)

        # Result should be zero
        assert jnp.allclose(new_state.phydata.heatflx, 0.0)

    def test_positive_flux_convention(self, basic_config):
        """Test flux sign convention (positive upward)."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        # Positive atmospheric flux (upward from surface)
        atm_phydata = Mock()
        atm_surface_flux = Mock()
        atm_surface_flux.hfluxn = jnp.ones((32, 64, 4)) * 50.0  # Positive upward
        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        step_fn = flux_model.gen_step_fn()
        new_state, predictions = step_fn(cpl, 0)

        # FluxModel computes: -hfluxn.sum(axis=-1)
        # So positive hfluxn (heat leaving surface) becomes negative flux
        expected = -200.0  # -(4 * 50)
        assert jnp.allclose(new_state.phydata.heatflx, expected)

    def test_jit_compilation(self, basic_config):
        """Test that step function can be JIT compiled."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        atm_phydata = Mock()
        atm_surface_flux = Mock()
        atm_surface_flux.hfluxn = jnp.ones((32, 64, 4)) * 10.0
        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        step_fn = flux_model.gen_step_fn()

        # Run twice to ensure compiled version is used
        new_state1, pred1 = step_fn(cpl, 0)
        cpl.flx = new_state1
        new_state2, pred2 = step_fn(cpl, 1)

        # Should produce consistent results
        assert jnp.allclose(new_state1.phydata.heatflx, new_state2.phydata.heatflx)

    def test_varying_atmospheric_input(self, basic_config):
        """Test response to varying atmospheric inputs."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        step_fn = flux_model.gen_step_fn()

        # Test with different flux magnitudes
        flux_values = [10.0, 50.0, 100.0, 0.0, -50.0]
        results = []

        for flux_val in flux_values:
            atm_phydata = Mock()
            atm_surface_flux = Mock()
            atm_surface_flux.hfluxn = jnp.ones((32, 64, 4)) * flux_val
            atm_phydata.surface_flux = atm_surface_flux
            cpl.atm = Mock()
            cpl.atm.phydata = atm_phydata

            new_state, predictions = step_fn(cpl, 0)
            results.append(new_state.phydata.heatflx.mean())

        # Results should vary with input
        results = jnp.array(results)
        assert not jnp.allclose(results, results[0])  # Not all the same

        # Should be proportional to input (negative of sum)
        expected = jnp.array(flux_values) * (-4)  # 4 components
        assert jnp.allclose(results, expected)

    def test_spatial_variation(self, basic_config):
        """Test handling of spatially varying fluxes."""
        flux_model = FluxModel(basic_config)
        state = flux_model.initialize()

        from unittest.mock import Mock
        cpl = Mock()
        cpl.flx = state

        # Create spatially varying flux
        atm_phydata = Mock()
        atm_surface_flux = Mock()

        # Create gradient in longitude
        lon_grid = jnp.linspace(0, 100, 32)[:, None]
        lat_grid = jnp.ones((32, 64))
        spatial_flux = lon_grid * lat_grid

        # Expand to include flux component dimension
        hfluxn = jnp.stack([spatial_flux] * 4, axis=-1)
        atm_surface_flux.hfluxn = hfluxn
        atm_phydata.surface_flux = atm_surface_flux
        cpl.atm = Mock()
        cpl.atm.phydata = atm_phydata

        step_fn = flux_model.gen_step_fn()
        new_state, predictions = step_fn(cpl, 0)

        # Output should preserve spatial structure
        # Should be -(sum of 4 identical fields) = -4 * spatial_flux
        expected = -4.0 * spatial_flux
        assert jnp.allclose(new_state.phydata.heatflx, expected)
