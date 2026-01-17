from typing import Any, List
import jax
import jax.numpy as jnp


def mean_leaf(
    tree: Any,
    axis: int | list,
) -> Any:
    """
    Apply :code:`jnp.mean` to leaf nodes.

    Args:
        tree : a tree object

    Returns:
        tree_mean : tree with jnp.mean applied to each of its leaf node.

    """
    return jax.tree_util.tree_map(lambda arr: jnp.mean(arr, axis=axis), tree)


def unwrap_leading_dims(
    obj: Any,
    first_n_dim: int = 2,
) -> Any:
    """
    Unwrap the leading dimensions of jax arrays

    Args:
        obj: A structure containining jax arrays

    Returns:
        Unwrapped object.

    """

    def _unwrap(arr):
        new_shape = (-1,) + arr.shape[first_n_dim:]
        return jnp.reshape(arr, new_shape)

    return jax.tree_util.tree_map(_unwrap, obj)


def stack_objects(
    objs: List,
):
    """
    Stack stack dataclasses together.

    Args:
        objs: A list of objects that need to be stacked

    Returns:
        Stacked object.

    """
    # objs is a list of pytrees with same structure
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return stacked


def concat_objects(
    objs: List,
    axis: int,
):
    """
    Concats dataclasses together.

    Args:
        objs: A list of objects that need to be concat

    Returns:
        Concatenated object.

    """
    # objs is a list of pytrees with same structure
    concatenated = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=axis), *objs
    )
    return concatenated
