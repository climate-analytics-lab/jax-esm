"""Adapter for the JCM spectral atmosphere (the ``jcm`` / jax-gcm package)."""

from jem.components.jcm.component import JCMComponent, JCMDerived
from jem.components.jcm.exchange_fields import SurfaceExchange

__all__ = [
    "JCMComponent",
    "JCMDerived",
    "SurfaceExchange",
]
