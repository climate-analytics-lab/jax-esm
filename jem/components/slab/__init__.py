"""Slab model components with shared base class."""

from jem.components.slab.base import SlabModelBase, load_monthly_climatology
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_atmosphere_model import (
    SlabAtmosphereModel,
    SlabAtmosphereParameters,
)
from jem.components.slab.slab_land_model import SlabLandModel, SlabLandParameters
from jem.components.slab.slab_ocean_model import SlabOceanModel, SlabOceanParameters
from jem.components.slab.slab_seaice_model import (
    SlabSeaiceModel,
    SlabSeaiceParameters,
)

__all__ = [
    "SlabAtmosphereModel",
    "SlabAtmosphereParameters",
    "SlabGrid",
    "SlabLandModel",
    "SlabLandParameters",
    "SlabModelBase",
    "SlabOceanModel",
    "SlabOceanParameters",
    "SlabSeaiceModel",
    "SlabSeaiceParameters",
    "load_monthly_climatology",
]
