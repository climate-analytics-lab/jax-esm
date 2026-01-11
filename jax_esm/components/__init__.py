"""Earth system components module."""

from jax_esm.components.JCM import JCM
from jax_esm.components.slab_atmosphere_model import SlabAtmosphereModel
from jax_esm.components.slab_ocean_model import SlabOceanModel
from jax_esm.components.slab_land_model import SlabLandModel

__all__ = [
    "JCM",
    "SlabLandModel",
    "SlabAtmosphereModel",
    "SlabOceanModel",
]
