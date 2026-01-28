"""Earth system components module."""

import jem.components.JCM as JCM
from jem.components.slab_atmosphere_model import SlabAtmosphereModel
from jem.components.slab_ocean_model import SlabOceanModel
from jem.components.slab_land_model import SlabLandModel
from jem.components.land_model import LandModel

__all__ = [
    "JCM",
    "SlabLandModel",
    "SlabAtmosphereModel",
    "SlabOceanModel",
    "LandModel",
]
