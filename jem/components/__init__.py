"""Earth system components module."""

from jem.components import jcm_component as JCM
from jem.components.slab.slab_atmosphere_model import SlabAtmosphereModel
from jem.components.slab.slab_land_model import SlabLandModel
from jem.components.slab.slab_ocean_model import SlabOceanModel
from jem.components.slab.slab_seaice_model import SlabSeaiceModel

__all__ = [
    "JCM",
    "SlabAtmosphereModel",
    "SlabLandModel",
    "SlabOceanModel",
    "SlabSeaiceModel",
]


def __getattr__(name):
    # Veros is an optional dependency: import it lazily so `import jem.components`
    # keeps working in environments without `veros` installed, and `from
    # jem.components import Veros` only fails when Veros is actually used.
    if name == "Veros":
        from jem.components import veros_component
        return veros_component
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
