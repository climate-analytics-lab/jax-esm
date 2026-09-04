"""JAX-ESM: A JAX-based Earth System Model coupler."""

__version__ = "0.2.0"

from jem.base import typing
from jem.base.coupler import Coupler

__all__ = [
    "Coupler",
    "typing",
]
