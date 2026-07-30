"""Earth system components module."""

import jem.components.JCM as JCM
from jem.components.slab.slab_atmosphere_model import SlabAtmosphereModel
from jem.components.slab.slab_ocean_model import SlabOceanModel
from jem.components.slab.slab_land_model import SlabLandModel
from jem.components.slab.slab_seaice_model import SlabSeaiceModel

__all__ = [
    "JCM",
    "SlabLandModel",
    "SlabAtmosphereModel",
    "SlabOceanModel",
    "SlabSeaiceModel",
]
