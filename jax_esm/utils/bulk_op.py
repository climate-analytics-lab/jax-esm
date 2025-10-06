
from typing import Dict, Tuple, Any, List
import jax
import jax.numpy as jnp
from jax import tree_util

def stack_objects(
    objs : List,
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

def concat_objects(
    objs : List,
    axis : int,
):
    """
    A tool function that concats dataclasses together.

    Args:

        objs : A list of objects that need to be concat

    Returns:

        concatenated : Concatenated object.
        
    """
    # objs is a list of pytrees with same structure
    concatenated = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=axis), *objs)
    return concatenated


