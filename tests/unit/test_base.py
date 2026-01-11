"""Unit tests for jem.components.base module."""

import pytest
import jax.numpy as jnp
import jax
from jem.components.base import (
    create_field_group_class,
    create_component_state_class,
    create_component_forcing_class,
)


class TestCreateFieldGroupClass:
    """Tests for create_field_group_class factory function."""

    def test_creates_class_with_correct_name(self):
        """Test that the created class has the correct name."""
        cls = create_field_group_class("TestFields", [("x", float, (2, 3))])
        assert cls.__name__ == "TestFields"

    def test_zeros_creates_zero_arrays(self):
        """Test that zeros() creates arrays filled with zeros."""
        cls = create_field_group_class(
            "TestFields",
            [
                ("temperature", float, (4, 8)),
                ("pressure", float, (4, 8)),
            ],
        )
        instance = cls.zeros()
        assert jnp.allclose(instance.temperature, 0.0)
        assert jnp.allclose(instance.pressure, 0.0)
        assert instance.temperature.shape == (4, 8)
        assert instance.pressure.shape == (4, 8)

    def test_ones_creates_one_arrays(self):
        """Test that ones() creates arrays filled with ones."""
        cls = create_field_group_class(
            "TestFields",
            [("value", float, (3, 5))],
        )
        instance = cls.ones()
        assert jnp.allclose(instance.value, 1.0)

    def test_copy_creates_independent_copy(self):
        """Test that copy() creates an independent copy."""
        cls = create_field_group_class(
            "TestFields",
            [("data", float, (2, 2))],
        )
        original = cls.zeros()
        copied = original.copy()

        # Modify original
        original_data = original.data.at[0, 0].set(999.0)

        # Copy should be unchanged
        assert copied.data[0, 0] == 0.0

    def test_copy_with_kwargs_overrides_fields(self):
        """Test that copy() with kwargs replaces specified fields."""
        cls = create_field_group_class(
            "TestFields",
            [
                ("a", float, (2,)),
                ("b", float, (2,)),
            ],
        )
        original = cls.zeros()
        new_a = jnp.array([1.0, 2.0])
        copied = original.copy(a=new_a)

        assert jnp.allclose(copied.a, new_a)
        assert jnp.allclose(copied.b, 0.0)  # Unchanged

    def test_scalar_field(self):
        """Test field with scalar (empty tuple) shape."""
        cls = create_field_group_class(
            "TestFields",
            [("sim_time", float, ())],
        )
        instance = cls.zeros()
        assert instance.sim_time.shape == ()
        assert float(instance.sim_time) == 0.0

    def test_tree_math_arithmetic(self):
        """Test that field groups support tree_math arithmetic."""
        cls = create_field_group_class(
            "TestFields",
            [("x", float, (3,))],
        )
        a = cls.ones()
        b = cls.ones()

        # Test addition via tree_math
        result = jax.tree_util.tree_map(lambda x, y: x + y, a, b)
        assert jnp.allclose(result.x, 2.0)

    def test_multiple_fields_different_shapes(self):
        """Test creating field group with multiple fields of different shapes."""
        cls = create_field_group_class(
            "MixedFields",
            [
                ("scalar", float, ()),
                ("vector", float, (10,)),
                ("matrix", float, (4, 5)),
                ("tensor", float, (2, 3, 4)),
            ],
        )
        instance = cls.zeros()
        assert instance.scalar.shape == ()
        assert instance.vector.shape == (10,)
        assert instance.matrix.shape == (4, 5)
        assert instance.tensor.shape == (2, 3, 4)


class TestCreateComponentStateClass:
    """Tests for create_component_state_class factory function."""

    def test_creates_class_with_prog_and_phydata(self):
        """Test that state class has prog and phydata attributes."""
        prog_cls = create_field_group_class(
            "Prog", [("temperature", float, (4, 8))]
        )
        phydata_cls = create_field_group_class(
            "Phydata", [("flux", float, (4, 8))]
        )
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        state = state_cls.zeros()
        assert hasattr(state, "prog")
        assert hasattr(state, "phydata")
        assert hasattr(state.prog, "temperature")
        assert hasattr(state.phydata, "flux")

    def test_zeros_creates_nested_zeros(self):
        """Test that zeros() creates zero arrays in nested structure."""
        prog_cls = create_field_group_class("Prog", [("T", float, (2, 2))])
        phydata_cls = create_field_group_class("Phydata", [("Q", float, (2, 2))])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        state = state_cls.zeros()
        assert jnp.allclose(state.prog.T, 0.0)
        assert jnp.allclose(state.phydata.Q, 0.0)

    def test_ones_creates_nested_ones(self):
        """Test that ones() creates one arrays in nested structure."""
        prog_cls = create_field_group_class("Prog", [("T", float, (2, 2))])
        phydata_cls = create_field_group_class("Phydata", [("Q", float, (2, 2))])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        state = state_cls.ones()
        assert jnp.allclose(state.prog.T, 1.0)
        assert jnp.allclose(state.phydata.Q, 1.0)

    def test_copy_with_prog_kwargs(self):
        """Test copy() with prog_kwargs modifies prog fields."""
        prog_cls = create_field_group_class(
            "Prog", [("T", float, (2,)), ("S", float, (2,))]
        )
        phydata_cls = create_field_group_class("Phydata", [("Q", float, (2,))])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        original = state_cls.zeros()
        new_T = jnp.array([300.0, 310.0])
        copied = original.copy(prog_kwargs=dict(T=new_T))

        assert jnp.allclose(copied.prog.T, new_T)
        assert jnp.allclose(copied.prog.S, 0.0)  # Unchanged
        assert jnp.allclose(copied.phydata.Q, 0.0)  # Unchanged

    def test_copy_with_phydata_kwargs(self):
        """Test copy() with phydata_kwargs modifies phydata fields."""
        prog_cls = create_field_group_class("Prog", [("T", float, (2,))])
        phydata_cls = create_field_group_class("Phydata", [("Q", float, (2,))])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        original = state_cls.zeros()
        new_Q = jnp.array([100.0, 200.0])
        copied = original.copy(phydata_kwargs=dict(Q=new_Q))

        assert jnp.allclose(copied.prog.T, 0.0)  # Unchanged
        assert jnp.allclose(copied.phydata.Q, new_Q)

    def test_copy_default_none_kwargs_work(self):
        """Test that copy() works correctly with default None kwargs."""
        prog_cls = create_field_group_class("Prog", [("T", float, (2,))])
        phydata_cls = create_field_group_class("Phydata", [("Q", float, (2,))])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        original = state_cls.ones()
        # Call copy with no arguments - should work without mutable default issues
        copied = original.copy()

        assert jnp.allclose(copied.prog.T, 1.0)
        assert jnp.allclose(copied.phydata.Q, 1.0)

    def test_is_pytree(self):
        """Test that state class is registered as a JAX pytree."""
        prog_cls = create_field_group_class("Prog", [("T", float, (3,))])
        phydata_cls = create_field_group_class("Phydata", [("Q", float, (3,))])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        state = state_cls.ones()

        # Should be able to use with jax.tree_util
        leaves = jax.tree_util.tree_leaves(state)
        assert len(leaves) == 2  # T and Q
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)


class TestCreateComponentForcingClass:
    """Tests for create_component_forcing_class factory function."""

    def test_creates_class_with_flux_and_scalar(self):
        """Test that forcing class has flux and scalar attributes."""
        flux_cls = create_field_group_class(
            "Flux", [("heat_flux", float, (4, 8))]
        )
        scalar_cls = create_field_group_class(
            "Scalar", [("sst", float, (4, 8))]
        )
        forcing_cls = create_component_forcing_class(
            flux_cls, scalar_cls, "TestForcing"
        )

        forcing = forcing_cls.zeros()
        assert hasattr(forcing, "flux")
        assert hasattr(forcing, "scalar")
        assert hasattr(forcing.flux, "heat_flux")
        assert hasattr(forcing.scalar, "sst")

    def test_zeros_creates_nested_zeros(self):
        """Test that zeros() creates zero arrays in nested structure."""
        flux_cls = create_field_group_class("Flux", [("F", float, (2, 2))])
        scalar_cls = create_field_group_class("Scalar", [("S", float, (2, 2))])
        forcing_cls = create_component_forcing_class(flux_cls, scalar_cls, "TestForcing")

        forcing = forcing_cls.zeros()
        assert jnp.allclose(forcing.flux.F, 0.0)
        assert jnp.allclose(forcing.scalar.S, 0.0)

    def test_copy_with_flux_kwargs(self):
        """Test copy() with flux_kwargs modifies flux fields."""
        flux_cls = create_field_group_class("Flux", [("F", float, (2,))])
        scalar_cls = create_field_group_class("Scalar", [("S", float, (2,))])
        forcing_cls = create_component_forcing_class(flux_cls, scalar_cls, "TestForcing")

        original = forcing_cls.zeros()
        new_F = jnp.array([50.0, 100.0])
        copied = original.copy(flux_kwargs=dict(F=new_F))

        assert jnp.allclose(copied.flux.F, new_F)
        assert jnp.allclose(copied.scalar.S, 0.0)

    def test_copy_with_scalar_kwargs(self):
        """Test copy() with scalar_kwargs modifies scalar fields."""
        flux_cls = create_field_group_class("Flux", [("F", float, (2,))])
        scalar_cls = create_field_group_class("Scalar", [("S", float, (2,))])
        forcing_cls = create_component_forcing_class(flux_cls, scalar_cls, "TestForcing")

        original = forcing_cls.zeros()
        new_S = jnp.array([290.0, 295.0])
        copied = original.copy(scalar_kwargs=dict(S=new_S))

        assert jnp.allclose(copied.flux.F, 0.0)
        assert jnp.allclose(copied.scalar.S, new_S)

    def test_copy_default_none_kwargs_work(self):
        """Test that copy() works correctly with default None kwargs."""
        flux_cls = create_field_group_class("Flux", [("F", float, (2,))])
        scalar_cls = create_field_group_class("Scalar", [("S", float, (2,))])
        forcing_cls = create_component_forcing_class(flux_cls, scalar_cls, "TestForcing")

        original = forcing_cls.ones()
        # Call copy with no arguments - should work without mutable default issues
        copied = original.copy()

        assert jnp.allclose(copied.flux.F, 1.0)
        assert jnp.allclose(copied.scalar.S, 1.0)


class TestJITCompatibility:
    """Tests for JIT compatibility of created classes."""

    def test_field_group_in_jitted_function(self):
        """Test that field groups work inside JIT-compiled functions."""
        cls = create_field_group_class(
            "TestFields",
            [("x", float, (3,)), ("y", float, (3,))],
        )

        @jax.jit
        def add_fields(fg):
            return fg.x + fg.y

        fg = cls.ones()
        result = add_fields(fg)
        assert jnp.allclose(result, 2.0)

    def test_state_in_jitted_function(self):
        """Test that component states work inside JIT-compiled functions."""
        prog_cls = create_field_group_class("Prog", [("T", float, (4,))])
        phydata_cls = create_field_group_class("Phydata", [("Q", float, (4,))])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        @jax.jit
        def update_temperature(state, delta):
            new_T = state.prog.T + delta
            return state.copy(prog_kwargs=dict(T=new_T))

        state = state_cls.zeros()
        delta = jnp.array([1.0, 2.0, 3.0, 4.0])
        new_state = update_temperature(state, delta)

        assert jnp.allclose(new_state.prog.T, delta)
        assert jnp.allclose(new_state.phydata.Q, 0.0)

    def test_state_in_scan(self):
        """Test that component states work with jax.lax.scan."""
        prog_cls = create_field_group_class("Prog", [("T", float, (2,))])
        phydata_cls = create_field_group_class("Phydata", [])
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        def step(state, x):
            new_T = state.prog.T + x
            new_state = state.copy(prog_kwargs=dict(T=new_T))
            return new_state, state.prog.T

        initial_state = state_cls.zeros()
        xs = jnp.ones((5, 2))  # 5 steps

        final_state, trajectory = jax.lax.scan(step, initial_state, xs)

        assert jnp.allclose(final_state.prog.T, 5.0)
        assert trajectory.shape == (5, 2)
