"""Integration tests for full coupled system.

These tests verify that components work together correctly in a coupled simulation.
Tests use simplified mock components rather than full JCM to keep tests fast.
"""

import jax
import jax.numpy as jnp
import pytest
import pandas as pd

from jax_esm import ComponentConfig
from jax_esm.coupling.coupler import Coupler, CouplerConfig
from jax_esm.components.base import (
    Component,
    create_component_state_class,
    create_field_group_class,
)
from jax_esm.utils.bulk_op import stack_objects


class SimplifiedAtmosphere(Component):
    """Simplified atmosphere that responds to SST."""

    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        shape = (16, 32)  # lat, lon

        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="AtmProg",
                fields=[
                    ("T_surface", float, shape),
                    ("humidity", float, shape),
                    ("sim_time", float, ()),
                ],
            ),
            phydata_cls=create_field_group_class(
                cls_name="AtmPhydata",
                fields=[
                    ("surface_flux", float, shape),  # Placeholder for hfluxn structure
                    ("precipitation", float, shape),
                ],
            ),
        )

    def initialize(self):
        return self.component_state_class.zeros().copy(
            prog_kwargs={
                "T_surface": jnp.ones((16, 32)) * 285.0,  # 12°C
                "humidity": jnp.ones((16, 32)) * 0.008,
            },
        )

    def gen_step_fn(self):
        @jax.jit
        def step_fn(cpl, t):
            # Atmosphere responds to ocean SST
            sst = cpl.ocn.prog.T
            T_diff = sst - cpl.atm.prog.T_surface

            # Simple temperature relaxation toward SST
            new_T_surface = cpl.atm.prog.T_surface + 0.001 * T_diff * self.timestep

            # Surface flux proportional to temperature difference
            surface_flux = -50.0 * T_diff  # W/m² per K difference

            new_state = cpl.atm.copy(
                prog_kwargs={
                    "T_surface": new_T_surface,
                    "sim_time": cpl.atm.prog.sim_time + self.timestep,
                },
                phydata_kwargs={
                    "surface_flux": surface_flux,
                    "precipitation": jnp.zeros((16, 32)),
                },
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


class SimplifiedFluxModel(Component):
    """Simplified flux model that computes net heat flux."""

    def __init__(self, config: ComponentConfig):
        super().__init__(config)

        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="FluxProg",
                fields=[("sim_time", float, ())],
            ),
            phydata_cls=create_field_group_class(
                cls_name="FluxPhydata",
                fields=[("heatflx", float, (16, 32))],
            ),
        )

    def initialize(self):
        return self.component_state_class.zeros()

    def gen_step_fn(self):
        @jax.jit
        def step_fn(cpl, t):
            # Get heat flux from atmosphere
            heat_flux = cpl.atm.phydata.surface_flux

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


class SimplifiedOcean(Component):
    """Simplified ocean with heat capacity."""

    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.heat_capacity = 4.0e7  # J/K/m² (50m mixed layer)

        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                cls_name="OceanProg",
                fields=[
                    ("T", float, (16, 32)),
                    ("sim_time", float, ()),
                ],
            ),
            phydata_cls=create_field_group_class(
                cls_name="OceanPhydata",
                fields=[("heat_content", float, (16, 32))],
            ),
        )

    def initialize(self):
        return self.component_state_class.zeros().copy(
            prog_kwargs={"T": jnp.ones((16, 32)) * 288.0},  # 15°C
        )

    def gen_step_fn(self):
        @jax.jit
        def step_fn(cpl, t):
            # Ocean temperature responds to heat flux
            heat_flux = cpl.flx.phydata.heatflx
            dT = (heat_flux * self.timestep) / self.heat_capacity

            new_T = cpl.ocn.prog.T + dT

            new_state = cpl.ocn.copy(
                prog_kwargs={
                    "T": new_T,
                    "sim_time": cpl.ocn.prog.sim_time + self.timestep,
                },
                phydata_kwargs={
                    "heat_content": new_T * self.heat_capacity,
                },
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


class TestIntegration:
    """Integration tests for coupled system."""

    @pytest.fixture
    def coupled_system(self):
        """Create a coupled atmosphere-flux-ocean system."""
        timestep = 3600.0  # 1 hour

        atm_config = ComponentConfig(
            name="atm",
            start_dt=pd.Timestamp("2000-01-01"),
            timestep=timestep,
            substeps=1,
            save_interval=timestep,
            grid={},
            params={},
        )
        flux_config = ComponentConfig(
            name="flx",
            start_dt=pd.Timestamp("2000-01-01"),
            timestep=timestep,
            substeps=1,
            save_interval=timestep,
            grid={},
            params={},
        )
        ocean_config = ComponentConfig(
            name="ocn",
            start_dt=pd.Timestamp("2000-01-01"),
            timestep=timestep,
            substeps=1,
            save_interval=timestep,
            grid={},
            params={},
        )

        atmosphere = SimplifiedAtmosphere(atm_config)
        flux_model = SimplifiedFluxModel(flux_config)
        ocean = SimplifiedOcean(ocean_config)

        coupler = Coupler(
            components={"atm": atmosphere, "flx": flux_model, "ocn": ocean},
            config=CouplerConfig(timestep=timestep),
        )

        return coupler

    def test_coupled_initialization(self, coupled_system):
        """Test that coupled system initializes correctly."""
        initial_state = coupled_system.initialize()

        # Check all components initialized
        assert hasattr(initial_state, "atm")
        assert hasattr(initial_state, "flx")
        assert hasattr(initial_state, "ocn")

        # Check initial values are reasonable
        assert jnp.all(initial_state.atm.prog.T_surface > 200.0)
        assert jnp.all(initial_state.atm.prog.T_surface < 350.0)
        assert jnp.all(initial_state.ocn.prog.T > 200.0)
        assert jnp.all(initial_state.ocn.prog.T < 350.0)

    def test_energy_exchange(self, coupled_system):
        """Test that energy is exchanged between components."""
        initial_state = coupled_system.initialize()

        # Get initial temperatures
        initial_atm_T = initial_state.atm.prog.T_surface.copy()
        initial_ocn_T = initial_state.ocn.prog.T.copy()

        # Run for one day
        final_state, predictions = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=86400.0,  # 1 day
            timestep=3600.0,
            jax_scan=True,
        )

        # Temperatures should have changed
        assert not jnp.allclose(final_state.atm.prog.T_surface, initial_atm_T)
        assert not jnp.allclose(final_state.ocn.prog.T, initial_ocn_T)

        # Atmosphere should warm (ocean is warmer initially)
        mean_atm_change = jnp.mean(final_state.atm.prog.T_surface - initial_atm_T)
        assert mean_atm_change > 0.0, "Atmosphere should warm toward ocean SST"

        # Ocean should cool (atmosphere is cooler initially)
        mean_ocn_change = jnp.mean(final_state.ocn.prog.T - initial_ocn_T)
        assert mean_ocn_change < 0.0, "Ocean should cool toward atmosphere temperature"

    def test_equilibration(self, coupled_system):
        """Test that system approaches equilibrium."""
        initial_state = coupled_system.initialize()

        # Run for 30 days
        final_state, predictions = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=86400.0 * 30,  # 30 days
            timestep=3600.0,
            jax_scan=True,
        )

        # Temperature difference should be smaller than initially
        initial_diff = jnp.abs(
            initial_state.ocn.prog.T - initial_state.atm.prog.T_surface
        )
        final_diff = jnp.abs(final_state.ocn.prog.T - final_state.atm.prog.T_surface)

        mean_initial_diff = jnp.mean(initial_diff)
        mean_final_diff = jnp.mean(final_diff)

        assert mean_final_diff < mean_initial_diff, (
            "System should equilibrate (temperature difference should decrease)"
        )

    def test_conservation_check(self, coupled_system):
        """Test approximate energy conservation in the system."""
        initial_state = coupled_system.initialize()

        # Compute initial total heat content
        # For atmosphere: C_atm * T_atm (simplified, no mass conservation)
        # For ocean: C_ocn * T_ocn
        C_atm = 1.0e7  # Simplified atmospheric heat capacity
        C_ocn = 4.0e7  # Ocean heat capacity

        initial_energy = jnp.sum(C_atm * initial_state.atm.prog.T_surface) + jnp.sum(
            C_ocn * initial_state.ocn.prog.T
        )

        # Run simulation
        final_state, predictions = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=86400.0,  # 1 day
            timestep=3600.0,
            jax_scan=True,
        )

        # Compute final total heat content
        final_energy = jnp.sum(C_atm * final_state.atm.prog.T_surface) + jnp.sum(
            C_ocn * final_state.ocn.prog.T
        )

        # Energy should be approximately conserved (no external forcing)
        # Allow small numerical errors
        relative_change = jnp.abs(final_energy - initial_energy) / initial_energy
        assert relative_change < 1e-6, (
            f"Energy should be conserved (relative change: {relative_change})"
        )

    def test_reproducibility(self, coupled_system):
        """Test that simulations are reproducible."""
        initial_state = coupled_system.initialize()

        # Run simulation twice
        final_state1, predictions1 = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=86400.0,
            timestep=3600.0,
            jax_scan=True,
        )

        final_state2, predictions2 = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=86400.0,
            timestep=3600.0,
            jax_scan=True,
        )

        # Results should be identical
        assert jnp.allclose(
            final_state1.atm.prog.T_surface, final_state2.atm.prog.T_surface
        )
        assert jnp.allclose(final_state1.ocn.prog.T, final_state2.ocn.prog.T)

    def test_jit_vs_nojit_equivalence(self, coupled_system):
        """Test that JIT and non-JIT modes give same results."""
        initial_state = coupled_system.initialize()

        # Run with JIT
        final_jit, pred_jit = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=7200.0,  # 2 hours (fast)
            timestep=3600.0,
            jax_scan=True,
        )

        # Run without JIT
        final_nojit, pred_nojit = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=7200.0,
            timestep=3600.0,
            jax_scan=False,
        )

        # Results should be very close
        assert jnp.allclose(
            final_jit.atm.prog.T_surface, final_nojit.atm.prog.T_surface, atol=1e-5
        )
        assert jnp.allclose(final_jit.ocn.prog.T, final_nojit.ocn.prog.T, atol=1e-5)

    def test_time_advancement(self, coupled_system):
        """Test that simulation time advances correctly."""
        initial_state = coupled_system.initialize()

        run_time = 86400.0  # 1 day
        timestep = 3600.0  # 1 hour

        final_state, predictions = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=run_time,
            timestep=timestep,
            jax_scan=True,
        )

        # All components should have same final time
        assert jnp.allclose(final_state.atm.prog.sim_time, run_time)
        assert jnp.allclose(final_state.flx.prog.sim_time, run_time)
        assert jnp.allclose(final_state.ocn.prog.sim_time, run_time)

    def test_spatial_consistency(self, coupled_system):
        """Test that coupling preserves spatial structure."""
        initial_state = coupled_system.initialize()

        # Create initial state with spatial gradient
        lat_gradient = jnp.linspace(270, 300, 16)[:, None]  # Pole to equator
        lon_uniform = jnp.ones((16, 32))
        spatial_T = lat_gradient * lon_uniform

        initial_state = initial_state.copy(
            atm=initial_state.atm.copy(prog_kwargs={"T_surface": spatial_T}),
            ocn=initial_state.ocn.copy(
                prog_kwargs={"T": spatial_T + 5.0}  # Ocean 5K warmer
            ),
        )

        # Run simulation
        final_state, predictions = coupled_system.run(
            init_cplstate=initial_state,
            start_time=0.0,
            end_time=86400.0,
            timestep=3600.0,
            jax_scan=True,
        )

        # Spatial gradient should be preserved (though smoothed)
        # Check that polar regions are still cooler than tropics
        atm_pole = jnp.mean(final_state.atm.prog.T_surface[0, :])
        atm_equator = jnp.mean(final_state.atm.prog.T_surface[-1, :])
        assert atm_pole < atm_equator, "Pole should be cooler than equator"

        ocn_pole = jnp.mean(final_state.ocn.prog.T[0, :])
        ocn_equator = jnp.mean(final_state.ocn.prog.T[-1, :])
        assert ocn_pole < ocn_equator, "Ocean pole should be cooler than equator"
