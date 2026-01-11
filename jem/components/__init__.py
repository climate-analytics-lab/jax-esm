"""Earth system components module."""

from jem.components.JCM import JCM
from jem.components.slab_atmosphere_model import SlabAtmosphereModel
from jem.components.slab_ocean_model import SlabOceanModel
from jem.components.slab_land_model import SlabLandModel

__all__ = [
    "JCM",
    "SlabLandModel",
    "SlabAtmosphereModel",
    "SlabOceanModel",
]
