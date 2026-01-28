"""JAX-ESM: A JAX-based Earth System Model coupler."""

__version__ = "0.1.0"

from jem.base.coupler import Coupler
import jem.base.typing as typing

__all__ = [
    "typing",
    "Coupler",
]
