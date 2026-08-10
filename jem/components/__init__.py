"""Earth system components module."""

from jem.components import JCM
from jem.components.slab.slab_atmosphere_model import SlabAtmosphereModel
from jem.components.slab.slab_land_model import SlabLandModel
from jem.components.slab.slab_ocean_model import SlabOceanModel

__all__ = [
    "JCM",
    "SlabAtmosphereModel",
    "SlabLandModel",
    "SlabOceanModel",
]
