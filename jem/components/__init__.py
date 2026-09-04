"""The components JAX-ESM can couple, and what it takes to configure them.

Everything a coupled run needs is importable from here: the JCM atmosphere
wrapper, the four slab models with their parameter structs, the grid they run
on and the loader for their boundary climatologies. Veros is the exception --
it is an optional dependency, so ``VerosComponent`` is resolved lazily by
``__getattr__`` and only fails if it is actually asked for.
"""

from jem.components import jcm_component as JCM
from jem.components.jcm import JCMComponent
from jem.components.slab import (
    SlabAtmosphereModel,
    SlabAtmosphereParameters,
    SlabGrid,
    SlabLandModel,
    SlabLandParameters,
    SlabModelBase,
    SlabOceanModel,
    SlabOceanParameters,
    SlabSeaiceModel,
    SlabSeaiceParameters,
    load_monthly_climatology,
)

__all__ = [
    "JCM",
    "JCMComponent",
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
    "VerosComponent",
    "load_monthly_climatology",
]


def __getattr__(name):
    # Veros is an optional dependency: import it lazily so `import jem.components`
    # keeps working in environments without `veros` installed, and `from
    # jem.components import Veros` only fails when Veros is actually used.
    if name == "Veros":
        from jem.components import veros_component
        return veros_component
    if name == "VerosComponent":
        from jem.components.veros_component import VerosComponent
        return VerosComponent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
