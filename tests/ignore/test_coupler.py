"""Tests for the coupler."""

import jax
import jax.numpy as jnp
import pytest
import pandas as pd

from jem import ComponentConfig
from jem.coupling.coupler import Coupler, CouplerConfig
from jem.components.base import (
    Component,
    create_component_state_class,
    create_field_group_class,
)
from jem.utils.bulk_op import stack_objects


class MockAtmosphere(Component):
    """Mock atmosphere component for testing."""

    def __init__(self, config: ComponentConfig):
        super().__init__(config)

        # Create state class
        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="AtmProg",
                fields=[
                    ("temperature", float, (5, 10)),
                    ("sim_time", float, ()),
                ],
            ),
            phydata_cls=create_field_group_class(
                cls_name="AtmPhydata",
                fields=[
                    ("surface_flux", float, (5, 10)),
                ],
            ),
        )

    def initialize(self):
        return self.component_state_class.zeros().copy(
            prog_kwargs={"temperature": jnp.ones((5, 10)) * 280.0},
        )

    def gen_step_fn(self):
        @jax.jit
        def step_fn(cpl, t):
            # Simple coupling: atmosphere responds to ocean SST
            sst = cpl.ocn.prog.T
            temp_effect = 0.01 * jnp.mean(sst)
            new_temp = cpl.atm.prog.temperature + temp_effect

            new_state = cpl.atm.copy(
                prog_kwargs={
                    "temperature": new_temp,
                    "sim_time": cpl.atm.prog.sim_time + self.timestep,
                }
            )

            predictions = stack_objects(
                [
                    {
                        "prog": new_state.prog,
                        "phydata": new_state.phydata,
                    }
                ]
            )

            return new_state, predictions

        return step_fn


class MockFlux(Component):
    """Mock flux component for testing."""

    def __init__(self, config: ComponentConfig):
        super().__init__(config)

        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="FluxProg",
                fields=[("sim_time", float, ())],
            ),
            phydata_cls=create_field_group_class(
                cls_name="FluxPhydata",
                fields=[("heatflx", float, (5, 10))],
            ),
        )

    def initialize(self):
        return self.component_state_class.zeros()

    def gen_step_fn(self):
        @jax.jit
        def step_fn(cpl, t):
            # Compute heat flux from atmosphere temperature
            heat_flux = cpl.atm.prog.temperature * 0.1

            new_state = cpl.flx.copy(
                prog_kwargs={"sim_time": cpl.flx.prog.sim_time + self.timestep},
                phydata_kwargs={"heatflx": heat_flux},
            )

            predictions = stack_objects(
                [
                    {
                        "prog": new_state.prog,
                        "phydata": new_state.phydata,
                    }
                ]
            )

            return new_state, predictions

        return step_fn


class MockOcean(Component):
    """Mock ocean component for testing."""

    def __init__(self, config: ComponentConfig):
        super().__init__(config)

        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="OceanProg",
                fields=[
                    ("T", float, (5, 10)),
                    ("sim_time", float, ()),
                ],
            ),
            phydata_cls=create_field_group_class(
                cls_name="OceanPhydata",
                fields=[("dummy", float, ())],
            ),
        )

    def initialize(self):
        return self.component_state_class.zeros().copy(
            prog_kwargs={"T": jnp.ones((5, 10)) * 288.0},
        )

    def gen_step_fn(self):
        @jax.jit
        def step_fn(cpl, t):
            # Ocean responds to heat flux
            heat_tendency = cpl.flx.phydata.heatflx / 1e6
            new_temp = cpl.ocn.prog.T + heat_tendency * self.timestep

            new_state = cpl.ocn.copy(
                prog_kwargs={
                    "T": new_temp,
                    "sim_time": cpl.ocn.prog.sim_time + self.timestep,
                }
            )

            predictions = stack_objects(
                [
                    {
                        "prog": new_state.prog,
                        "phydata": new_state.phydata,
                    }
                ]
            )

            return new_state, predictions

        return step_fn


class TestCoupler:
    """Test coupler functionality."""

    def test_coupler_initialization(self):
        """Test coupler initialization."""
        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=900.0,
            substeps=2,
            save_interval=1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=900.0,
            substeps=1,
            save_interval=1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=2,
            save_interval=1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )

        atmosphere = MockAtmosphere(atm_config)
        flux = MockFlux(flux_config)
        ocean = MockOcean(ocean_config)

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux, "ocn": ocean},
            config=CouplerConfig(timestep=1800.0),
        )

        assert len(coupler.components) == 3
        assert "atm" in coupler.components
        assert "flx" in coupler.components
        assert "ocn" in coupler.components

    def test_coupler_initialize_states(self):
        """Test state initialization through coupler."""
        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=900.0,
            substeps=2,
            save_interval=1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=900.0,
            substeps=1,
            save_interval=1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=2,
            save_interval=1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )

        atmosphere = MockAtmosphere(atm_config)
        flux = MockFlux(flux_config)
        ocean = MockOcean(ocean_config)

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux, "ocn": ocean},
            config=CouplerConfig(timestep=1800.0),
        )

        states = coupler.initialize()

        # Check that states were created
        assert hasattr(states, "atm")
        assert hasattr(states, "flx")
        assert hasattr(states, "ocn")

        # Check initial values
        assert jnp.allclose(states.atm.prog.temperature, 280.0)
        assert jnp.allclose(states.ocn.prog.T, 288.0)
        assert states.atm.prog.sim_time == 0.0
        assert states.ocn.prog.sim_time == 0.0

    def test_step_function_generation(self):
        """Test step function generation."""
        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )

        atmosphere = MockAtmosphere(atm_config)
        flux = MockFlux(flux_config)
        ocean = MockOcean(ocean_config)

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux, "ocn": ocean},
            config=CouplerConfig(timestep=1800.0),
        )

        step_fn = coupler.gen_step_fn()
        assert callable(step_fn)

    def test_single_coupled_step(self):
        """Test single coupling step."""
        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )

        atmosphere = MockAtmosphere(atm_config)
        flux = MockFlux(flux_config)
        ocean = MockOcean(ocean_config)

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux, "ocn": ocean},
            config=CouplerConfig(timestep=1800.0),
        )

        initial_state = coupler.initialize()
        step_fn = coupler.gen_step_fn()

        new_state, predictions = step_fn(initial_state, 0)

        # Check that all components advanced
        assert new_state.atm.prog.sim_time == 1800.0
        assert new_state.flx.prog.sim_time == 1800.0
        assert new_state.ocn.prog.sim_time == 1800.0

        # Check that coupling occurred (states changed)
        # Atmosphere should change (responds to ocean SST)
        assert not jnp.allclose(
            initial_state.atm.prog.temperature, new_state.atm.prog.temperature
        )
        # Ocean may not change much in one timestep with weak coupling
        # Just check that flux was computed
        assert not jnp.allclose(new_state.flx.phydata.heatflx, 0.0)

    def test_coupler_run(self):
        """Test running full simulation."""
        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )

        atmosphere = MockAtmosphere(atm_config)
        flux = MockFlux(flux_config)
        ocean = MockOcean(ocean_config)

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux, "ocn": ocean},
            config=CouplerConfig(timestep=1800.0),
        )

        initial_states = coupler.initialize()

        # Run for 2 hours (4 timesteps of 1800s)
        final_state, predictions = coupler.run(
            init_cplstate=initial_states,
            start_time=0.0,
            end_time=7200.0,
            timestep=1800.0,
            jax_scan=True,
        )

        # Check final times
        assert jnp.allclose(final_state.atm.prog.sim_time, 7200.0)
        assert jnp.allclose(final_state.flx.prog.sim_time, 7200.0)
        assert jnp.allclose(final_state.ocn.prog.sim_time, 7200.0)

        # Check predictions structure
        assert "atm" in predictions
        assert "flx" in predictions
        assert "ocn" in predictions

    def test_coupler_run_without_jit(self):
        """Test running without JIT compilation (debugging mode)."""
        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )

        atmosphere = MockAtmosphere(atm_config)
        flux = MockFlux(flux_config)
        ocean = MockOcean(ocean_config)

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux, "ocn": ocean},
            config=CouplerConfig(timestep=1800.0),
        )

        initial_states = coupler.initialize()

        # Run without JIT (adhoc_scan)
        final_state, predictions = coupler.run(
            init_cplstate=initial_states,
            start_time=0.0,
            end_time=3600.0,  # Just 2 steps for speed
            timestep=1800.0,
            jax_scan=False,
        )

        # Should still produce correct results
        assert jnp.allclose(final_state.atm.prog.sim_time, 3600.0)

    def test_timestep_validation(self):
        """Test that coupler validates timestep divides total time."""
        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )

        atmosphere = MockAtmosphere(atm_config)
        flux = MockFlux(flux_config)
        ocean = MockOcean(ocean_config)

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux, "ocn": ocean},
            config=CouplerConfig(timestep=1800.0),
        )

        initial_states = coupler.initialize()

        # This should raise an exception (1700 doesn't divide evenly into 1800)
        with pytest.raises(Exception):
            coupler.run(
                init_cplstate=initial_states,
                start_time=0.0,
                end_time=1700.0,
                timestep=1800.0,
            )

    def test_predictions_to_xarray(self):
        """Test conversion of predictions to xarray."""
        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=1800.0,
            substeps=1,
            save_interval=1800.0,
            grid={},
            params={},
        )

        # Add predictions_to_xarray method to mock components
        def mock_to_xarray(predictions):
            import xarray as xr

            return xr.Dataset()

        atmosphere = MockAtmosphere(atm_config)
        atmosphere.predictions_to_xarray = mock_to_xarray

        flux = MockFlux(flux_config)
        flux.predictions_to_xarray = mock_to_xarray

        ocean = MockOcean(ocean_config)
        ocean.predictions_to_xarray = mock_to_xarray

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux, "ocn": ocean},
            config=CouplerConfig(timestep=1800.0),
        )

        initial_states = coupler.initialize()

        final_state, predictions = coupler.run(
            init_cplstate=initial_states,
            start_time=0.0,
            end_time=3600.0,
            timestep=1800.0,
        )

        # Convert to xarray
        datasets = coupler.predictions_to_xarray(predictions)

        assert "atm" in datasets
        assert "flx" in datasets
        assert "ocn" in datasets
