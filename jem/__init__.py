"""JAX-ESM: A JAX-based Earth System Model coupler."""

__version__ = "0.1.0"

from .base.coupler import Coupler
from .base.forcing_mapper import BasicForcingMapper
from .base.regridder import IdentityRegridder

from .base.grid import Grid
from .base.grid import GridSpecification

from .base.domain import Domain
from .base.variable import (
    VariableMetadata,
    VariableRegistry,
)

from .base.data_structure import (
    typed_and_dimensioned,
    build_dataclass_from_typed_and_dimensioned,
)

__all__ = [
    "Coupler",
    "BasicForcingMapper",
    "IdentityRegridder",
    "Grid",
    "GridSpecification",
    "Domain",
    "VariableRegistry",
    "VariableMetadata",
    "typed_and_dimensioned",
    "build_dataclass_from_typed_and_dimensioned",
]
