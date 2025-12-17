from jax_esm.base.variable_metadata import VariableMetadata
from jax import Array
from typing import Dict
import coordax as cx
class VariableRegistry:
    """
    Registry maintaining coordinate metadata for all exchanged fields.
    
    This is the single source of truth for coordinate information,
    keeping it separate from component state objects.
    """
    
    def __init__(self):
        self._metadata: Dict[str, Dict[str, VariableMetadata]] = {}
    
    def register_component(
        self,
        component_name: str,
        field_metadata: Dict[str, VariableMetadata]
    ):
        """Register all fields for a component."""
        self._metadata[component_name] = field_metadata
    
    def get_metadata(self, component_name: str, variable_name: str) -> VariableMetadata:
        """Retrieve metadata for a specific field."""
        return self._metadata[component_name][variable_name]
    
    def tag_variable(
        self,
        component_name: str,
        variable_name: str,
        data: Array,
    ) -> cx.Field:
        """Tag a raw array with coordinate metadata."""
        metadata = self.get_metadata(component_name, variable_name)
        return metadata.wrap_with_coordax(data)
    
    def untag_variable(self, var: cx.Field) -> Array:
        """Extract raw data from tagged variable."""
        return var.data

