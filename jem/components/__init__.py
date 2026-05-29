"""Earth system components module."""

import jem.components.JCM as JCM
from jem.components.slab.slab_atmosphere_model import SlabAtmosphereModel
from jem.components.slab.slab_ocean_model.slab_ocean_model_biogeochem import SlabOceanModelBGC
from jem.components.slab.slab_ocean_model import SlabOceanModel
from jem.components.slab.slab_land_model import SlabLandModel

from jem.components.CO2AtmosphereBoxModel import CO2AtmosphereBoxModel


__all__ = [
    "JCM",
    "SlabLandModel",
    "SlabAtmosphereModel",
    "SlabOceanModel",
    "SlabOceanModelBGC",
    "CO2AtmosphereBoxModel",
]
