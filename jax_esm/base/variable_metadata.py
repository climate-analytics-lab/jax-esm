from dataclasses import dataclass
import coordax as cx
from jax import Array
from typing import Dict, Any

@dataclass
class VariableMetadata:
    """Metadata for a single field."""
    name: str
    dimensions: tuple[str, ...]
    coords: Optional[Dict[str, Array]] = None
    attrs: Optional[Dict[str, Any]] = None
    
    def to_coordax(self, data: jnp.ndarray) -> cx.Variable:
        """Convert raw array to coordax Variable."""
        return cx.Variable(
            data=data,
            dimensions=self.dimensions,
            coords=self.coords,
            attrs=self.attrs or {}
        )
    
    @staticmethod
    def from_coordax(var: cx.Variable) -> tuple[Array, 'VariableMetadata']:
        """Extract data and metadata from coordax Variable."""
        metadata = VariableMetadata(
            name=var.attrs.get('name', 'unknown'),
            dimensions=var.dimensions,
            coords=dict(var.coords),
            attrs=dict(var.attrs)
        )
        return var.data, metadata

