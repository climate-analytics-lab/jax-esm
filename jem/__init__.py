"""JAX-ESM: A JAX-based Earth System Model coupler."""

__version__ = "0.2.0"

# The coupling core only. The components live in `jem.components`, which is
# not imported here: pulling in the JCM wrapper would import the whole
# atmosphere (jcm, dinosaur, flax) just to say `import jem`.
from jem.base.component import (
    Carry,
    Component,
    CoupledCarry,
    CouplingTime,
    Diagnostics,
    Exchanger,
    SupportsBind,
    SupportsCheckpoint,
    SupportsXarray,
    TimeAxis,
)
from jem.base.coupler import Coupler

__all__ = [
    "Carry",
    "Component",
    "CoupledCarry",
    "Coupler",
    "CouplingTime",
    "Diagnostics",
    "Exchanger",
    "SupportsBind",
    "SupportsCheckpoint",
    "SupportsXarray",
    "TimeAxis",
]
