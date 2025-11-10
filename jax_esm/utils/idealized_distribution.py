import jax.numpy as jnp

def positive_cosine_cubic_latitude_squared(
    lat,
    amplitude: float = 1.0,
) -> jnp.array:
    return jnp.where(jnp.abs(lat) < jnp.pi/3, amplitude * jnp.cos(3*lat/2)**2, 0)


