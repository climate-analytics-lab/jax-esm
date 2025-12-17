from dataclasses import dataclass
import coordax as cx
from jax import Array
from typing import Dict, Any, Optional

@dataclass
class VariableMetadata:
    """Metadata for a single field."""
    name: str
    dimensions: tuple[str, ...]
    coords: Optional[Dict[str, Array]] = None
    attrs: Optional[Dict[str, Any]] = None
    
    def to_coordax_field(self, data: Array) -> cx.Field:
        """Convert raw array to coordax Variable."""
        return cx.Field(
            data,
            dims=self.dimensions,
        )
    
