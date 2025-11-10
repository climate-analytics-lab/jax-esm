"""JAX-ESM: A JAX-based Earth System Model coupler."""

from jax_esm.components.base import Component, ComponentConfig
from jax_esm.coupling.coupler import Coupler

__version__ = "0.1.0"

__all__ = [
    "Component",
    "ComponentConfig",
    "Coupler",
]
