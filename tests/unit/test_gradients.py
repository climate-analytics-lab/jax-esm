"""Unit tests for gradient flow through JAX-ESM components.

These tests verify that gradients can be computed through:
1. Component state operations (copy, arithmetic)
2. Single component step functions
3. Coupled component simulations
4. Various forcing transformations
"""

import pytest
import jax
import jax.numpy as jnp

from jax_esm.components.base import (
    create_field_group_class,
    create_component_state_class,
    create_component_forcing_class,
    Component,
    CoupledComponentConfig,
)
from jax_esm.coupling.forcing_mapper import ForcingMapper
from jax_esm.utils.bulk_op import stack_objects


# =============================================================================
# Helper classes for gradient testing
# =============================================================================


class SimpleSlabComponent(Component):
    """Simplified slab component for gradient testing.

    Physics: dT/dt = (forcing - T) / tau

    This is a simple relaxation equation that should have well-defined
    gradients with respect to initial conditions, forcing, and tau.
    """

    def __init__(self, name, grid_shape=(4, 8), timestep=86400.0, tau=10 * 86400.0):
        super().__init__(CoupledComponentConfig(name=name, timestep=timestep))
        self.grid_shape = grid_shape
        self.timestep = timestep
        self.tau = tau  # Relaxation timescale

        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                "Prog",
                [
                    ("temperature", float, grid_shape),
                    ("sim_time", float, ()),
                ],
            ),
            phydata_cls=create_field_group_class("Phydata", []),
        )

        self.component_forcing_class = create_component_forcing_class(
            flux_cls=create_field_group_class("Flux", []),
            scalar_cls=create_field_group_class(
                "Scalar",
                [("target_temperature", float, grid_shape)],
            ),
            cls_name=f"{name}Forcing",
        )

    def initialize(self, initial_temp=None):
        if initial_temp is None:
            initial_temp = jnp.ones(self.grid_shape) * 288.0
        return self.component_state_class.zeros().copy(
            prog_kwargs=dict(temperature=initial_temp)
        )

    def generate_step_function(self, jitted=False):
        tau = self.tau
        dt = self.timestep

        def step(state, forcing, t):
            # Simple relaxation: dT/dt = (T_target - T) / tau
            T = state.prog.temperature
            T_target = forcing.scalar.target_temperature

            # Euler forward
            dT = (T_target - T) / tau * dt
            new_T = T + dT

            new_state = state.copy(
                prog_kwargs=dict(
                    temperature=new_T,
                    sim_time=state.prog.sim_time + dt,
                )
            )
            return new_state, stack_objects([{"temperature": new_T}])

        return jax.jit(step) if jitted else step

    def validate(self):
        pass

    def predictions_to_xarray(self, predictions):
        return None


# =============================================================================
# Tests for gradient flow through state operations
# =============================================================================


class TestStateGradients:
    """Tests for gradients through component state operations."""

    def test_gradient_through_field_group_copy(self):
        """Test that gradients flow through field group copy operation."""
        cls = create_field_group_class(
            "TestFields",
            [("x", float, (3,))],
        )

        def loss_fn(x_init):
            fg = cls.zeros().copy(x=x_init)
            return jnp.sum(fg.x ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(loss_fn)(x)
        expected_grad = 2 * x  # d/dx (x^2) = 2x
        assert jnp.allclose(grad, expected_grad)

    def test_gradient_through_state_copy(self):
        """Test that gradients flow through component state copy."""
        prog_cls = create_field_group_class("Prog", [("T", float, (2,))])
        phydata_cls = create_field_group_class("Phydata", [])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        def loss_fn(T_init):
            state = state_cls.zeros().copy(prog_kwargs=dict(T=T_init))
            return jnp.sum(state.prog.T ** 2)

        T = jnp.array([3.0, 4.0])
        grad = jax.grad(loss_fn)(T)
        expected_grad = 2 * T
        assert jnp.allclose(grad, expected_grad)

    def test_gradient_through_nested_operations(self):
        """Test gradients through multiple nested copy operations."""
        prog_cls = create_field_group_class("Prog", [("T", float, (2,))])
        phydata_cls = create_field_group_class("Phydata", [])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        def loss_fn(T_init):
            state = state_cls.zeros().copy(prog_kwargs=dict(T=T_init))
            # Multiple operations
            state = state.copy(prog_kwargs=dict(T=state.prog.T * 2))
            state = state.copy(prog_kwargs=dict(T=state.prog.T + 1))
            return jnp.sum(state.prog.T)

        T = jnp.array([1.0, 2.0])
        grad = jax.grad(loss_fn)(T)
        # T -> 2*T -> 2*T + 1, d(sum)/dT = 2
        expected_grad = jnp.array([2.0, 2.0])
        assert jnp.allclose(grad, expected_grad)


# =============================================================================
# Tests for gradient flow through single component step
# =============================================================================


class TestSingleComponentGradients:
    """Tests for gradients through single component step functions."""

    def test_gradient_wrt_initial_temperature(self):
        """Test gradient of final temperature w.r.t. initial temperature."""
        component = SimpleSlabComponent("test", grid_shape=(2,), timestep=86400.0)
        step_fn = component.generate_step_function(jitted=False)

        def loss_fn(T_init):
            state = component.initialize(initial_temp=T_init)
            forcing = component.component_forcing_class.zeros().copy(
                scalar_kwargs=dict(target_temperature=jnp.ones(2) * 300.0)
            )
            new_state, _ = step_fn(state, forcing, 0.0)
            return jnp.sum(new_state.prog.temperature)

        T_init = jnp.array([280.0, 290.0])
        grad = jax.grad(loss_fn)(T_init)

        # Gradient should be non-zero and reasonable
        assert not jnp.allclose(grad, 0.0)
        # For relaxation equation, dT_new/dT_init = 1 - dt/tau
        expected = 1 - component.timestep / component.tau
        assert jnp.allclose(grad, expected)

    def test_gradient_wrt_forcing(self):
        """Test gradient of final temperature w.r.t. target temperature."""
        component = SimpleSlabComponent("test", grid_shape=(2,), timestep=86400.0)
        step_fn = component.generate_step_function(jitted=False)

        def loss_fn(T_target):
            state = component.initialize(initial_temp=jnp.array([280.0, 280.0]))
            forcing = component.component_forcing_class.zeros().copy(
                scalar_kwargs=dict(target_temperature=T_target)
            )
            new_state, _ = step_fn(state, forcing, 0.0)
            return jnp.sum(new_state.prog.temperature)

        T_target = jnp.array([300.0, 310.0])
        grad = jax.grad(loss_fn)(T_target)

        # For relaxation, dT_new/dT_target = dt/tau
        expected = component.timestep / component.tau
        assert jnp.allclose(grad, expected)

    def test_gradient_through_multiple_steps(self):
        """Test gradient flows through multiple timesteps."""
        component = SimpleSlabComponent("test", grid_shape=(2,), timestep=86400.0)
        step_fn = component.generate_step_function(jitted=False)

        def loss_fn(T_init):
            state = component.initialize(initial_temp=T_init)
            forcing = component.component_forcing_class.zeros().copy(
                scalar_kwargs=dict(target_temperature=jnp.ones(2) * 300.0)
            )

            # Run 5 steps
            for i in range(5):
                state, _ = step_fn(state, forcing, float(i * 86400.0))

            return jnp.sum(state.prog.temperature)

        T_init = jnp.array([280.0, 290.0])
        grad = jax.grad(loss_fn)(T_init)

        # Gradient should decay with number of steps
        # After n steps: dT_final/dT_init = (1 - dt/tau)^n
        decay = (1 - component.timestep / component.tau) ** 5
        assert jnp.allclose(grad, decay)

    def test_gradient_with_jit(self):
        """Test that JIT compilation preserves gradients."""
        component = SimpleSlabComponent("test", grid_shape=(2,), timestep=86400.0)

        # Non-JIT version
        step_fn_nojit = component.generate_step_function(jitted=False)

        # JIT version
        step_fn_jit = component.generate_step_function(jitted=True)

        def make_loss_fn(step_fn):
            def loss_fn(T_init):
                state = component.initialize(initial_temp=T_init)
                forcing = component.component_forcing_class.zeros().copy(
                    scalar_kwargs=dict(target_temperature=jnp.ones(2) * 300.0)
                )
                new_state, _ = step_fn(state, forcing, 0.0)
                return jnp.sum(new_state.prog.temperature)
            return loss_fn

        T_init = jnp.array([280.0, 290.0])

        grad_nojit = jax.grad(make_loss_fn(step_fn_nojit))(T_init)
        grad_jit = jax.grad(make_loss_fn(step_fn_jit))(T_init)

        assert jnp.allclose(grad_nojit, grad_jit)


# =============================================================================
# Tests for gradient flow through scan
# =============================================================================


class TestScanGradients:
    """Tests for gradients through jax.lax.scan."""

    def test_gradient_through_scan(self):
        """Test gradient flows through jax.lax.scan."""
        component = SimpleSlabComponent("test", grid_shape=(2,), timestep=86400.0)
        step_fn = component.generate_step_function(jitted=False)

        def loss_fn(T_init):
            state = component.initialize(initial_temp=T_init)
            forcing = component.component_forcing_class.zeros().copy(
                scalar_kwargs=dict(target_temperature=jnp.ones(2) * 300.0)
            )

            def scan_step(carry, t):
                state = carry
                new_state, predictions = step_fn(state, forcing, t)
                return new_state, predictions

            times = jnp.arange(5) * 86400.0
            final_state, _ = jax.lax.scan(scan_step, state, times)

            return jnp.sum(final_state.prog.temperature)

        T_init = jnp.array([280.0, 290.0])
        grad = jax.grad(loss_fn)(T_init)

        # Should match the explicit loop result
        decay = (1 - component.timestep / component.tau) ** 5
        assert jnp.allclose(grad, decay, rtol=1e-5)


# =============================================================================
# Tests for gradient flow through ForcingMapper
# =============================================================================


class TestForcingMapperGradients:
    """Tests for gradients through ForcingMapper operations."""

    def test_gradient_through_mapping(self):
        """Test that gradients flow through forcing mapping."""
        # Create two simple components
        comp1 = SimpleSlabComponent("comp1", grid_shape=(2,))
        comp2 = SimpleSlabComponent("comp2", grid_shape=(2,))

        components = {"comp1": comp1, "comp2": comp2}

        mapper = ForcingMapper(components)
        mapper.add_forcing_mapping(
            "comp1", "comp2",
            {"prog.temperature": "scalar.target_temperature"}
        )

        def loss_fn(T1_init):
            # Create states
            state1 = comp1.initialize(initial_temp=T1_init)
            state2 = comp2.initialize(initial_temp=jnp.ones(2) * 280.0)

            states = {"comp1": state1, "comp2": state2}

            # Map forcings
            forcings = mapper.map_forcings(states)

            # Loss is sum of target temperature received by comp2
            return jnp.sum(forcings["comp2"].scalar.target_temperature)

        T1 = jnp.array([290.0, 300.0])
        grad = jax.grad(loss_fn)(T1)

        # Direct mapping, so gradient should be 1
        assert jnp.allclose(grad, 1.0)

    def test_gradient_through_transformation(self):
        """Test gradients through forcing transformation."""
        comp1 = SimpleSlabComponent("comp1", grid_shape=(2,))
        comp2 = SimpleSlabComponent("comp2", grid_shape=(2,))

        components = {"comp1": comp1, "comp2": comp2}

        mapper = ForcingMapper(components)
        mapper.add_forcing_mapping(
            "comp1", "comp2",
            {"prog.temperature": "scalar.target_temperature"}
        )

        # Add transformation that scales by 2
        mapper.add_transformation(
            "comp1", "comp2",
            "prog.temperature", "scalar.target_temperature",
            lambda x: x * 2.0
        )

        def loss_fn(T1_init):
            state1 = comp1.initialize(initial_temp=T1_init)
            state2 = comp2.initialize()

            states = {"comp1": state1, "comp2": state2}
            forcings = mapper.map_forcings(states)

            return jnp.sum(forcings["comp2"].scalar.target_temperature)

        T1 = jnp.array([290.0, 300.0])
        grad = jax.grad(loss_fn)(T1)

        # Transformation scales by 2, so gradient should be 2
        assert jnp.allclose(grad, 2.0)


# =============================================================================
# Integration tests - coupled simulation gradients
# =============================================================================


class TestCoupledGradients:
    """Integration tests for gradients through coupled simulations."""

    def test_coupled_gradient_single_step(self):
        """Test gradient through single step of coupled simulation."""
        comp1 = SimpleSlabComponent("comp1", grid_shape=(2,), timestep=86400.0)
        comp2 = SimpleSlabComponent("comp2", grid_shape=(2,), timestep=86400.0)

        step_fn1 = comp1.generate_step_function(jitted=False)
        step_fn2 = comp2.generate_step_function(jitted=False)

        components = {"comp1": comp1, "comp2": comp2}
        mapper = ForcingMapper(components)

        # Bidirectional coupling
        mapper.add_forcing_mapping("comp1", "comp2", {"prog.temperature": "scalar.target_temperature"})
        mapper.add_forcing_mapping("comp2", "comp1", {"prog.temperature": "scalar.target_temperature"})

        def loss_fn(T1_init):
            state1 = comp1.initialize(initial_temp=T1_init)
            state2 = comp2.initialize(initial_temp=jnp.ones(2) * 300.0)

            states = {"comp1": state1, "comp2": state2}

            # Exchange forcings
            forcings = mapper.map_forcings(states)

            # Step both components
            new_state1, _ = step_fn1(state1, forcings["comp1"], 0.0)
            new_state2, _ = step_fn2(state2, forcings["comp2"], 0.0)

            # Loss based on both components
            return jnp.sum(new_state1.prog.temperature) + jnp.sum(new_state2.prog.temperature)

        T1_init = jnp.array([280.0, 285.0])
        grad = jax.grad(loss_fn)(T1_init)

        # Gradient should be non-zero
        assert not jnp.allclose(grad, 0.0)

        # Verify gradient is correct analytically
        dt = comp1.timestep
        tau = comp1.tau

        # For comp1: dT1_new/dT1_init = 1 - dt/tau (from its own evolution)
        # For comp2: dT2_new/dT1_init = dt/tau (T1 influences T2's target)
        expected = (1 - dt/tau) + (dt/tau)  # = 1
        assert jnp.allclose(grad, expected)

    def test_value_and_grad(self):
        """Test jax.value_and_grad works for coupled system."""
        comp = SimpleSlabComponent("test", grid_shape=(2,), timestep=86400.0)
        step_fn = comp.generate_step_function(jitted=False)

        def loss_fn(T_init):
            state = comp.initialize(initial_temp=T_init)
            forcing = comp.component_forcing_class.zeros().copy(
                scalar_kwargs=dict(target_temperature=jnp.ones(2) * 300.0)
            )

            for i in range(3):
                state, _ = step_fn(state, forcing, float(i * 86400.0))

            return jnp.mean(state.prog.temperature)

        T_init = jnp.array([280.0, 290.0])
        value, grad = jax.value_and_grad(loss_fn)(T_init)

        # Value should be reasonable temperature
        assert 280.0 < value < 300.0

        # Gradient should be non-zero
        assert not jnp.allclose(grad, 0.0)
