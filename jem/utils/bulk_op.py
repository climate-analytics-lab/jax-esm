from typing import Any

import jax
import jax.numpy as jnp


def unwrap_leading_dims(
    obj: Any,
    first_n_dim: int = 2,
) -> Any:
    """
    A tool function that unwraps the leading dimensions of jax arrays

    Args:

        obj : A structure containining jax arrays

    Returns:

        unwrapped object.

    """

    def _unwrap(arr):
        new_shape = (-1,) + arr.shape[first_n_dim:]
        return jnp.reshape(arr, new_shape)

    return jax.tree_util.tree_map(_unwrap, obj)


def stack_objects(
    objs: list,
):
    """
    A tool function that stack dataclasses together.

    Args:

        objs : A list of objects that need to be stacked

    Returns:

        stacked : Stacked object.

    """
    # objs is a list of pytrees with same structure
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return stacked
