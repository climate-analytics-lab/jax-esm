"""The components JAX-ESM can couple, and what it takes to configure them.

Everything a coupled run needs is importable from here: the JCM atmosphere
wrapper, the slab models, the bucket land and the Winton sea ice with their parameter structs, the grid they run
on and the loader for their boundary climatologies. Veros is the exception --
it is an optional dependency, so ``VerosComponent`` is resolved lazily by
``__getattr__`` and only fails if it is actually asked for. It is therefore
deliberately absent from ``__all__``: ``from jem.components import *`` has to
work on an install without Veros, and a name listed in ``__all__`` is looked up
by the star-import. ``jem.components.VerosComponent`` stays reachable.
"""

from jem.components.jcm import JCMComponent
from jem.components.slab import (
    BucketLandModel,
    BucketLandParameters,
    SlabAtmosphereModel,
    SlabAtmosphereParameters,
    SlabGrid,
    SlabModelBase,
    SlabOceanModel,
    SlabOceanParameters,
    SlabSeaiceModel,
    SlabSeaiceParameters,
    WintonSeaiceModel,
    WintonSeaiceParameters,
    load_monthly_climatology,
)

__all__ = [
    "BucketLandModel",
    "BucketLandParameters",
    "JCMComponent",
    "SlabAtmosphereModel",
    "SlabAtmosphereParameters",
    "SlabGrid",
    "SlabModelBase",
    "SlabOceanModel",
    "SlabOceanParameters",
    "SlabSeaiceModel",
    "SlabSeaiceParameters",
    "WintonSeaiceModel",
    "WintonSeaiceParameters",
    "load_monthly_climatology",
]


_VEROS_INSTALL_HINT = (
    "VerosComponent needs the jittable Veros fork, "
    "https://github.com/meteorologytoday/veros-jittable , installed from "
    "source as described in docs/source/getting_started.rst. The `veros` "
    "package on PyPI is the upstream, non-jittable one and does not work "
    "with JAX-ESM."
)


def __getattr__(name):
    # Veros is an optional dependency: import it lazily so `import jem.components`
    # keeps working in environments without `veros` installed, and
    # `jem.components.VerosComponent` only fails when Veros is actually used.
    # The bare "No module named 'veros'" does not say which `veros` to install,
    # and the obvious `pip install veros` fetches a package that cannot run
    # here, so the failure is re-raised with the instruction.
    if name == "VerosComponent":
        try:
            from jem.components.veros_component import VerosComponent
        except ModuleNotFoundError as error:
            if error.name is None or error.name.split(".")[0] != "veros":
                raise
            raise ImportError(_VEROS_INSTALL_HINT) from error
        return VerosComponent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
