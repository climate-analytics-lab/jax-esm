"""JAX-ESM: A JAX-based Earth System Model coupler."""

from jax_esm.components.base import Component, ComponentConfig, BoundaryFluxes
from jax_esm.coupling.coupler import Coupler
from jax_esm.coupling.coupler import CouplerConfig
#from jax_esm.coupling.flux_exchange import FluxExchanger
#from jax_esm.coupling.time_integration import TimeIntegrator

__version__ = "0.1.0"

__all__ = [
    "Component",
    "ComponentConfig",
    "BoundaryFluxes",
    "Coupler",
    "CouplerConfig",
#    "FluxExchanger",
#    "TimeIntegrator",
]
