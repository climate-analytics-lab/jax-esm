
from collections import abc

import jax.numpy as jnp
import tree_math
from typing import Callable

from jax import tree_util

@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    geopotential: jnp.ndarray
    surface_pressure: jnp.ndarray  # normalized surface pressure (normalized by p0)