"""Earth system components module."""

from jax_esm.components.base import Component
from jax_esm.components.JCM import JCM
from jax_esm.components.SlabOceanModel import SlabOceanModel
from jax_esm.components.SlabAtmosphereModel import SlabAtmosphereModel
from jax_esm.components.SlabLandModel import SlabLandModel

__all__ = [
    "Component",
    "JCM",
    "SlabOceanModel",
    "SlabAtmosphereModel",
    "SlabLandModel",
]
