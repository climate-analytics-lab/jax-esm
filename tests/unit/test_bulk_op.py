"""Unit tests for jax_esm.utils.bulk_op module."""

import pytest
import jax
import jax.numpy as jnp
from dataclasses import dataclass

import tree_math

from jax_esm.utils.bulk_op import (
    mean_leaf,
    unwrap_leading_dims,
    stack_objects,
    concat_objects,
)
from jax_esm.components.base import create_field_group_class


# Helper to create pytree-compatible dataclasses
def pytree_dataclass(cls):
    """Decorator to make a dataclass a JAX pytree using tree_math."""
    return tree_math.struct(dataclass(cls))


# =============================================================================
# Tests for mean_leaf
# =============================================================================


class TestMeanLeaf:
    """Tests for mean_leaf function."""

    def test_mean_single_array(self):
        """Test mean_leaf on a single array."""
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = mean_leaf(arr, axis=0)
        expected = jnp.array([2.0, 3.0])
        assert jnp.allclose(result, expected)

    def test_mean_dict_of_arrays(self):
        """Test mean_leaf on a dictionary of arrays."""
        tree = {
            "a": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": jnp.array([[10.0, 20.0], [30.0, 40.0]]),
        }
        result = mean_leaf(tree, axis=0)

        assert jnp.allclose(result["a"], jnp.array([2.0, 3.0]))
        assert jnp.allclose(result["b"], jnp.array([20.0, 30.0]))

    def test_mean_nested_structure(self):
        """Test mean_leaf on nested dict structure."""
        # Use nested dicts instead of dataclasses for simplicity
        tree = {
            "inner": {"x": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
            "y": jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        }
        result = mean_leaf(tree, axis=0)

        assert jnp.allclose(result["inner"]["x"], jnp.array([2.0, 3.0]))
        assert jnp.allclose(result["y"], jnp.array([6.0, 7.0]))

    def test_mean_multiple_axes(self):
        """Test mean_leaf with multiple axes."""
        arr = jnp.ones((2, 3, 4))
        result = mean_leaf(arr, axis=[0, 1])
        assert result.shape == (4,)
        assert jnp.allclose(result, 1.0)

    def test_mean_with_field_group(self):
        """Test mean_leaf with field group class."""
        cls = create_field_group_class(
            "TestFields",
            [("temp", float, (5, 4, 8))],
        )
        fg = cls.ones()
        result = mean_leaf(fg, axis=0)
        assert result.temp.shape == (4, 8)
        assert jnp.allclose(result.temp, 1.0)


# =============================================================================
# Tests for unwrap_leading_dims
# =============================================================================


class TestUnwrapLeadingDims:
    """Tests for unwrap_leading_dims function."""

    def test_unwrap_single_array(self):
        """Test unwrapping leading dims of single array."""
        arr = jnp.ones((2, 3, 4, 5))
        result = unwrap_leading_dims(arr, first_n_dim=2)
        assert result.shape == (6, 4, 5)  # 2*3=6

    def test_unwrap_dict_of_arrays(self):
        """Test unwrapping dict of arrays."""
        tree = {
            "a": jnp.ones((2, 3, 4)),
            "b": jnp.ones((2, 3, 5, 6)),
        }
        result = unwrap_leading_dims(tree, first_n_dim=2)

        assert result["a"].shape == (6, 4)
        assert result["b"].shape == (6, 5, 6)

    def test_unwrap_nested_structure(self):
        """Test unwrapping nested dict structure."""
        tree = {
            "x": jnp.ones((2, 3, 4)),
            "y": jnp.ones((2, 3, 5)),
        }
        result = unwrap_leading_dims(tree, first_n_dim=2)

        assert result["x"].shape == (6, 4)
        assert result["y"].shape == (6, 5)

    def test_unwrap_first_n_dim_1(self):
        """Test unwrapping with first_n_dim=1 (no change)."""
        arr = jnp.ones((5, 4, 3))
        result = unwrap_leading_dims(arr, first_n_dim=1)
        # Should flatten first dim only, but since it's already flat, shape is same
        assert result.shape == (5, 4, 3)

    def test_unwrap_first_n_dim_3(self):
        """Test unwrapping with first_n_dim=3."""
        arr = jnp.ones((2, 3, 4, 5, 6))
        result = unwrap_leading_dims(arr, first_n_dim=3)
        assert result.shape == (24, 5, 6)  # 2*3*4=24

    def test_unwrap_with_field_group(self):
        """Test unwrap with field group class."""
        cls = create_field_group_class(
            "TestFields",
            [
                ("temp", float, (10, 5, 4, 8)),  # time, substeps, lat, lon
                ("time", float, (10, 5)),
            ],
        )
        fg = cls.ones()
        result = unwrap_leading_dims(fg, first_n_dim=2)

        assert result.temp.shape == (50, 4, 8)  # 10*5=50
        assert result.time.shape == (50,)


# =============================================================================
# Tests for stack_objects
# =============================================================================


class TestStackObjects:
    """Tests for stack_objects function."""

    def test_stack_dicts(self):
        """Test stacking list of dicts."""
        objs = [
            {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])},
            {"a": jnp.array([5.0, 6.0]), "b": jnp.array([7.0, 8.0])},
        ]
        result = stack_objects(objs)

        assert result["a"].shape == (2, 2)
        assert result["b"].shape == (2, 2)
        assert jnp.allclose(result["a"][0], jnp.array([1.0, 2.0]))
        assert jnp.allclose(result["a"][1], jnp.array([5.0, 6.0]))

    def test_stack_dataclasses(self):
        """Test stacking list of dicts (mimicking dataclass behavior)."""
        objs = [
            {"x": jnp.array([1.0]), "y": jnp.array([2.0])},
            {"x": jnp.array([3.0]), "y": jnp.array([4.0])},
            {"x": jnp.array([5.0]), "y": jnp.array([6.0])},
        ]
        result = stack_objects(objs)

        assert result["x"].shape == (3, 1)
        assert result["y"].shape == (3, 1)
        assert jnp.allclose(result["x"].flatten(), jnp.array([1.0, 3.0, 5.0]))

    def test_stack_nested_structures(self):
        """Test stacking nested dict structures."""
        objs = [
            {"inner": {"data": jnp.array([1.0])}, "value": jnp.array([10.0])},
            {"inner": {"data": jnp.array([2.0])}, "value": jnp.array([20.0])},
        ]
        result = stack_objects(objs)

        assert result["inner"]["data"].shape == (2, 1)
        assert result["value"].shape == (2, 1)

    def test_stack_single_object(self):
        """Test stacking a single object (list of one)."""
        objs = [{"x": jnp.array([1.0, 2.0, 3.0])}]
        result = stack_objects(objs)

        assert result["x"].shape == (1, 3)

    def test_stack_with_field_groups(self):
        """Test stacking field group instances."""
        cls = create_field_group_class(
            "TestFields",
            [("temp", float, (4, 8))],
        )

        objs = [cls.zeros(), cls.ones()]
        objs[0] = objs[0].copy(temp=jnp.ones((4, 8)) * 280.0)
        objs[1] = objs[1].copy(temp=jnp.ones((4, 8)) * 290.0)

        result = stack_objects(objs)

        assert result.temp.shape == (2, 4, 8)
        assert jnp.allclose(result.temp[0], 280.0)
        assert jnp.allclose(result.temp[1], 290.0)


# =============================================================================
# Tests for concat_objects
# =============================================================================


class TestConcatObjects:
    """Tests for concat_objects function."""

    def test_concat_dicts_axis0(self):
        """Test concatenating dicts along axis 0."""
        objs = [
            {"a": jnp.array([[1.0, 2.0]]), "b": jnp.array([[3.0, 4.0]])},
            {"a": jnp.array([[5.0, 6.0]]), "b": jnp.array([[7.0, 8.0]])},
        ]
        result = concat_objects(objs, axis=0)

        assert result["a"].shape == (2, 2)
        assert result["b"].shape == (2, 2)
        assert jnp.allclose(result["a"], jnp.array([[1.0, 2.0], [5.0, 6.0]]))

    def test_concat_dicts_axis1(self):
        """Test concatenating dicts along axis 1."""
        objs = [
            {"a": jnp.array([[1.0], [2.0]])},
            {"a": jnp.array([[3.0], [4.0]])},
        ]
        result = concat_objects(objs, axis=1)

        assert result["a"].shape == (2, 2)
        assert jnp.allclose(result["a"], jnp.array([[1.0, 3.0], [2.0, 4.0]]))

    def test_concat_dataclasses(self):
        """Test concatenating dict structures (mimicking dataclasses)."""
        objs = [
            {"positions": jnp.array([[0.0, 0.0]]), "times": jnp.array([0.0])},
            {"positions": jnp.array([[1.0, 1.0]]), "times": jnp.array([1.0])},
        ]
        result = concat_objects(objs, axis=0)

        assert result["positions"].shape == (2, 2)
        assert result["times"].shape == (2,)

    def test_concat_multiple_objects(self):
        """Test concatenating more than two objects."""
        objs = [
            {"x": jnp.array([1.0])},
            {"x": jnp.array([2.0])},
            {"x": jnp.array([3.0])},
            {"x": jnp.array([4.0])},
        ]
        result = concat_objects(objs, axis=0)

        assert result["x"].shape == (4,)
        assert jnp.allclose(result["x"], jnp.array([1.0, 2.0, 3.0, 4.0]))


# =============================================================================
# Integration tests - common usage patterns
# =============================================================================


class TestBulkOpIntegration:
    """Integration tests for bulk operations in typical ESM workflows."""

    def test_stack_then_unwrap(self):
        """Test stacking predictions then unwrapping (common in scan output)."""
        # Simulate scan output: list of predictions at each timestep
        predictions = [
            {"temp": jnp.ones((4, 8)) * (280.0 + i)} for i in range(10)
        ]

        # Stack to get (time, lat, lon)
        stacked = stack_objects(predictions)
        assert stacked["temp"].shape == (10, 4, 8)

        # This is the typical output shape from scan

    def test_nested_stack_unwrap_pattern(self):
        """Test the nested stack/unwrap pattern from coupler output."""
        # Simulate: outer scan produces list of inner scan results
        # Inner scan produces list of substep results

        # Create nested structure like coupler output
        outer_results = []
        for outer_step in range(5):  # 5 coupling steps
            inner_results = []
            for inner_step in range(3):  # 3 substeps per coupling
                inner_results.append({
                    "temp": jnp.ones((4, 8)) * (280.0 + outer_step + inner_step * 0.1)
                })
            outer_results.append(stack_objects(inner_results))

        # Stack outer results
        final = stack_objects(outer_results)
        assert final["temp"].shape == (5, 3, 4, 8)

        # Unwrap to flatten time dimensions
        unwrapped = unwrap_leading_dims(final, first_n_dim=2)
        assert unwrapped["temp"].shape == (15, 4, 8)

    def test_mean_over_time(self):
        """Test computing time mean of stacked predictions."""
        predictions = [
            {"temp": jnp.ones((4, 8)) * temp}
            for temp in [280.0, 285.0, 290.0, 295.0, 300.0]
        ]
        stacked = stack_objects(predictions)
        time_mean = mean_leaf(stacked, axis=0)

        assert time_mean["temp"].shape == (4, 8)
        assert jnp.allclose(time_mean["temp"], 290.0)  # Mean of 280-300
