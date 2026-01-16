from jax import Array
from typing import Dict, Any, Optional, List
import coordax as cx
from dataclasses import dataclass
from jem.base.exceptions import ValidationError


@dataclass
class VariableMetadata:
    """Metadata for a single variable.
    
    When coupling, it is important to keep track of the dimensions of the
    individual variable. :code:`VariableMetadata` adopts :code:`Coordax.Field`
    to do so. Most importantly, :code:`VariableMetadata` will be used by
    :code:`Transformer` to re-order the dimension.

    Attributes:
        name: Access name of the variable.
        shape: A tuple holding the shape of the variable.
        dimension: A tuple holding the name of each dimension.
        coords: The coordinate information of each dimension (not used).
        attrs: The additional information of each dimension (not used).
    """

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

    A :code:`VariableResgistry` document the information of variable.

    Attributes:
        _metadata: The dict object holding all registered
            :code:`VariableMetadata`. 
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
