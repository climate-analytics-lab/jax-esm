"""JAX-ESM: A JAX-based Earth System Model coupler."""

__version__ = "1.0.0a0"

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
from jem.base.coupler import Coupler, nested_carry, with_nested_carry

# The declarative exchangers and the output helpers sit on the coupling core
# and import nothing from `jem.components`, so they cost nothing to export.
from jem.exchangers import (
    Exchange,
    ExchangeSpec,
    default_exchangers,
    default_exchanges,
    default_workflow,
)
from jem.output import datasets_for_chunk, postprocess, write_chunk

__all__ = [
    "Carry",
    "Component",
    "CoupledCarry",
    "Coupler",
    "CouplingTime",
    "Diagnostics",
    "Exchange",
    "ExchangeSpec",
    "Exchanger",
    "SupportsBind",
    "SupportsCheckpoint",
    "SupportsXarray",
    "TimeAxis",
    "datasets_for_chunk",
    "default_exchangers",
    "default_exchanges",
    "default_workflow",
    "nested_carry",
    "postprocess",
    "with_nested_carry",
    "write_chunk",
]
