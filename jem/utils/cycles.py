import coordax as cx

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

def evaluate_periodic(
    x: ArrayLike,
    data: cx.Field,
    name_of_interpolated_axis: str = "time",
) -> ArrayLike:
    """Interpolate sparse data to equally spaced
       on a periodic cycle over domain [0, 1). 
      
    Args:
        x: A real number between that will be wrapped into [0, 1)
        data: A coordax.Field whose `interpolate_axis` marking which
              LabeledAxis will be used for interpolation.
        name_of_interpolated_axis: Name of the axis that will be interapolated.
              The value of this axis should be in [0, 1). If any value
              goes beyond the interval [0, 1), ValueError will be raised.
    Return:
        The untagged interpolated data based on the value of `x`. The interpolated 
        dimension is removed.
         
    """

    reordered_data = data.order_as(name_of_interpolated_axis, ...)
    interpolated_axis = reordered_data.axes[name_of_interpolated_axis]
    if not hasattr(interpolated_axis, "ticks"):  # Changed 'axis' to 'interpolated_axis'
        raise ValueError("The input data's interpolation axis must be `LabeledAxis`.")

    # Wrap x to [0, 1)
    x_wrapped = jnp.mod(x, 1.0)
    
    # Find the index of the left boundary
    idx = jnp.searchsorted(interpolated_axis.ticks, x_wrapped, side='right') - 1
    
    # Handle periodic boundary (wrap around)
    idx_next = jnp.mod(idx + 1, len(interpolated_axis.ticks))
    
    # Get the tick values
    t0 = interpolated_axis.ticks[idx]
    t1 = interpolated_axis.ticks[idx_next]
    
    # Handle wraparound for periodic interpolation
    spacing_between_coarse_data = jnp.where(t0 < t1, t1 - t0, t1 + 1.0 - t0)
    distance_to_the_left_point = jnp.where(t0 < t1, x_wrapped - t0, x_wrapped + 1.0 - t0)
    
    # Linear interpolation weight
    alpha = distance_to_the_left_point / spacing_between_coarse_data
    
    return (1.0 - alpha) * reordered_data.data[idx] + alpha * reordered_data.data[idx_next]  # type: ignore

# Define a vectorized version
vmap_evaluate_periodic = jax.vmap(evaluate_periodic, in_axes=(0, None, None))  # vmap over x, keep data and name constant
