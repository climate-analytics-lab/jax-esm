import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array

def evaluate_periodic_1D(
    pivot_x: ArrayLike,
    pivot_y: ArrayLike,
    interpolated_x: ArrayLike,
    axis: int = 0,
) -> Array:
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

    if len(pivot_x.shape) != 1:
        raise ValueError("Only 1D pivot_x is allowed")

    if len(interpolated_x.shape) != 1:
        raise ValueError("Only 1D interpolated_x is allowed")

    if jnp.any(pivot_x[1:] - pivot_x[:-1] < 0):
        raise ValueError("Input pivot_x has to be monotincally increasing")
   
    interpolated_x = jnp.mod(interpolated_x, 1.0)

    y_dim = len(y.shape)
    move_interpolate_axis_to_the_first = (axis,) + tuple(range(axis)) + tuple(range(y_dim))[axis+1:]
    restore_to_original_axes = tuple(range(1, axis+1)) + (0,) + tuple(range(y_dim))[axis+1:]


    pivot_y  = jnp.transpose(pivot_y, axes=move_interpolate_axis_to_the_first)
    pivot_x = jnp.concatenate([ pivot_x[-1:] - 1.0, pivot_x, 1.0 + pivot_x[:1] ])
    pivot_y = jnp.concatenate([ pivot_y[-1:]      , pivot_y,       pivot_y[:1] ])

    # Find the index of the left boundary
    idx = jnp.searchsorted(pivot_x, interpolated_x, side='right') - 1
    
    # Handle periodic boundary (wrap around)
    idx_next = jnp.mod(idx + 1, len(interpolated_x))
    
    # Get the tick values
    x_left  = pivot_x[idx]
    x_right = pivot_x[idx_next]
    
    # Handle wraparound for periodic interpolation
    spacing_between_coarse_data = jnp.where(x_left < x_right, x_right - x_left, (1.0 + x_right) - x_left)
    distance_to_the_left_point = jnp.where(x_left < x_right, interpolated_x - x_left, (1.0 + interpolated_x) - x_left)
   
    # Linear interpolation weight
    alpha = distance_to_the_left_point / spacing_between_coarse_data
    
    interpolated_y = (1.0 - alpha) * pivot_y[idx] + alpha * pivot_y[idx_next]
    y_restored     = jnp.transpose(interpolated_y, axes=restore_to_original_axes)
    
    return y_restored

# Define a vectorized version
#vmap_evaluate_periodic = jax.vmap(evaluate_periodic, in_axes=(0, None, None, None))  # vmap over x, keep data and name constant


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    
    x = jnp.linspace(0, 1, 21)[:-1]
    y = jnp.sin(x * jnp.pi * 2)
    
    interpolated_x = jnp.linspace(0.85, 3.05, 50)[:-1]
    interpolated_y = evaluate_periodic_1D(x, y, interpolated_x)

    plt.plot(interpolated_x, interpolated_y)
    plt.scatter(x, y)
    plt.show()
    



