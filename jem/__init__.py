"""JAX-ESM: A JAX-based Earth System Model coupler."""

__version__ = "0.1.0"

from .coupling.coupler import Coupler
from .coupling.transformer import Transformer
from .coupling.forcing_mapper import ForcingMapper

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
    "Transformer",
    "ForcingMapper",
    "IdentityTransformer",
    "Grid",
    "GridSpecification",
    "Domain",
    "VariableRegistry",
    "VariableMetadata",
    "typed_and_dimensioned",
    "build_dataclass_from_typed_and_dimensioned",
]
