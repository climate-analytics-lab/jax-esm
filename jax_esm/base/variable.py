from jax import Array
from typing import Dict, Any, Optional, List
import coordax as cx
from dataclasses import dataclass
from jem.base.exceptions import ValidationError


@dataclass
class VariableMetadata:
    """Metadata for a single field."""

    name: str
    shape: tuple[int, ...]
    dimensions: tuple[str, ...]
    coords: Optional[Dict[str, Array]] = None
    attrs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if len(self.shape) != len(self.dimensions):
            print(self.name, " : ", self.shape, ": ", self.dimensions)
            raise ValidationError(
                "Length of metadata shape does not match length of dimensions."
            )

    def to_coordax_field(self, data: Array) -> cx.Field:
        """Convert raw array to coordax Variable."""

        if data.shape != self.shape:
            raise ValidationError(
                f"Shape of input data ({','.merge(data.shape)}) does not match metadata ({','.merge(self.shape)})."
            )

        return cx.Field(
            data,
            dims=self.dimensions,
        )


class VariableRegistry:
    """
    Registry maintaining coordinate metadata for all exchanged fields.

    This is the single source of truth for coordinate information,
    keeping it separate from component state objects.
    """

    def __init__(self, variable_metadatas: Optional[List[VariableMetadata]] = None):
        self._metadata: Dict[str, VariableMetadata] = {}

        if variable_metadatas is not None:
            for variable_metadata in variable_metadatas:
                self.register_variable(variable_metadata)

    def register_variable(
        self,
        variable_metadata: VariableMetadata,
    ) -> "VariableRegistry":
        """Register all fields for a component."""
        self._metadata[variable_metadata.name] = variable_metadata
        return self

    def get_metadata(self, variable_name: str) -> VariableMetadata:
        """Retrieve metadata for a specific field."""
        return self._metadata[variable_name]

    def tag_variable(
        self,
        variable_name: str,
        data: Array,
    ) -> cx.Field:
        """Tag a raw array with coordinate metadata."""
        return self._metadata[variable_name].wrap_with_coordax(data)
