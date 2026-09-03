import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def evaluate_cyclic_linear(x: ArrayLike, data: jax.Array) -> jax.Array:
    """Linearly interpolate equally-spaced periodic records at cycle position x.

    Records are assumed equally spaced at positions 0, 1/n, 2/n, ..., (n-1)/n
    in the cycle [0, 1). Interpolation wraps around (record n-1 connects back
    to record 0).

    Args:
        x: Scalar cycle position. Wrapped into [0, 1) via mod before use.
        data: Array of shape (..., n) — n equally-spaced records along the last axis.

    Returns:
        Interpolated array of shape (...) with the record axis removed.
    """
    n = data.shape[-1]
    x_wrapped = jnp.mod(x, 1.0)
    pos = x_wrapped * n                        # position in [0, n)
    idx0 = jnp.int32(jnp.floor(pos)) % n
    idx1 = (idx0 + 1) % n
    alpha = pos - jnp.floor(pos)              # weight toward idx1, in [0, 1)
    return (1.0 - alpha) * data[..., idx0] + alpha * data[..., idx1]
