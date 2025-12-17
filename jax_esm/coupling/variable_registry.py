from jax_esm.base.variable_metadata import VariableMetadata
from jax import Array

class VariableRegistry:
    """
    Registry maintaining coordinate metadata for all exchanged fields.
    
    This is the single source of truth for coordinate information,
    keeping it separate from component state objects.
    """
    
    def __init__(self):
        self._metadata: Dict[str, Dict[str, FieldMetadata]] = {}
    
    def register_component(
        self,
        component_name: str,
        field_metadata: Dict[str, FieldMetadata]
    ):
        """Register all fields for a component."""
        self._metadata[component_name] = field_metadata
    
    def get_metadata(self, component_name: str, variable_name: str) -> FieldMetadata:
        """Retrieve metadata for a specific field."""
        return self._metadata[component_name][variable_name]
    
    def tag_variable(
        self,
        component_name: str,
        variable_name: str,
        data: Array,
    ) -> cx.Variable:
        """Tag a raw array with coordinate metadata."""
        metadata = self.get_metadata(component_name, variable_name)
        return metadata.to_coordax(data)
    
    def untag_variable(self, var: cx.Variable) -> Array:
        """Extract raw data from tagged variable."""
        return var.data

