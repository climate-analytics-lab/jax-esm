from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from jax import Array
from typeguard import typechecked

from jem.base.exceptions import ValidationError
from jem.base.typing import (
    VariableMetadata,
)
from jem.mapping.grid import Grid


class BasicRegridder(ABC):
    """
    Abstract base class for interpolating between climate model grids.

    This class provides a framework for regriding data between different
    grid configurations (e.g., atmosphere to ocean) with built-in validation.

    Attributes:
        source_grid: A Grid object holding the information of the source grid.
        target_grid: A Grid object holding the information of the target grid.
        validate_shape: A flag indicating whether to check the compatibility of the input
            and output array when performing regridation.
         
    """

    source_grid: Grid | None = None
    target_grid: Grid | None = None

    validate_shape: bool

    @typechecked
    def __init__(
        self,
        source_grid: Grid | None = None,
        target_grid: Grid | None = None,
        validate_shape: bool = True,
    ):
        """Initialize the regridder."""

        self.source_grid = source_grid
        self.target_grid = target_grid
        self.validate_shape = validate_shape

        # Store validation results
        self.last_validation: dict[str, Any] = {}

    def __call__(self, data: Array) -> Array:
        """Apply regridation with validation.

        Args:
            data: Data on the source grid

        Returns:
            Validated regrided data

        Raises:
            ValidationError: If validation checks fail
        """
        # Check input shape
        if (self.source_grid is not None) and (data.shape != self.source_grid.shape):
            raise ValueError(
                f"Input shape {data.shape} does not match source grid "
                f"shape {self.source_grid.shape}" 
            )

        # Apply regridation
        result = self(data)

        # Validate output
        self._validate(data, result)

        return result

    def _validate(self, source_data: Array, target_data: Array):
        """
        Perform validation checks on regrided data.

        Parameters
        ----------
        source_data : Array
            Original data on source grid
        target_data : Array
            Transformed data on target grid

        Raises
        ------
        ValidationError
            If any validation check fails
        """
        self.last_validation = {}

        # Shape validation
        if self.validate_shape:
            self._validate_shape(target_data)

    def _validate_shape(self, target_data: Array):
        """Validate that output has correct shape."""
        if not self.target_grid:
            raise ValidationError("target_grid was not provided. Cannot validate shape")

        if (not self.target_grid) and (target_data.shape != self.target_grid.shape):
            raise ValidationError(
                f"Output shape {target_data.shape} does not match "
                f"target grid shape {self.target_grid.shape}"
            )
        self.last_validation["shape_valid"] = True

    @abstractmethod
    def validate_metadata(
        self,
        source_metadata: VariableMetadata,
        target_metadata: VariableMetadata,
    ):
        """Validate the metadata"""

    def get_info(self):
       
        return {
            'type': str(self.__class__),
            'source_grid' : self.source_grid.get_info() if self.source_grid is not None else None, 
            'target_grid' : self.target_grid.get_info() if self.target_grid is not None else None, 
            'validate_shape' : self.validate_shape,
        }
       

class IdentityRegridder(BasicRegridder):
    """Identity mapping (no interpolation)."""

    def __call__(self, data: Array) -> Array:
        return data

    def validate_metadata(
        self, source_metadata: VariableMetadata, target_metadata: VariableMetadata
    ):
        if source_metadata[0] != target_metadata[0]: # shape
            raise ValidationError(f"Source {source_metadata[0]!s} and target metadata {target_metadata[0]!s} must have the same shape")


class BilinearRegridder(BasicRegridder):
    """Simple bilinear interpolation (for demonstration)."""

    def __call__(self, data: Array) -> Array:
        """Apply bilinear interpolation."""
        from scipy.interpolate import RegularRegridder

        if self.source_grid is None:
            raise ValidationError("Source grid cannot be None when regrid with BilinearRegridder.")
        if self.target_grid is None:
            raise ValidationError("Target grid cannot be None when regrid with BilinearRegridder.")

        # Create regridder for source grid
        source_shape = self.source_grid.shape
        x = jnp.linspace(0, 1, source_shape[0])
        y = jnp.linspace(0, 1, source_shape[1])
        interp = RegularRegridder((x, y), data, method="linear")

        # Create target grid coordinates
        target_shape = self.target_grid.shape
        x_new = jnp.linspace(0, 1, target_shape[0])
        y_new = jnp.linspace(0, 1, target_shape[1])
        xx, yy = jnp.meshgrid(x_new, y_new, indexing="ij")

        # Interpolate
        return interp((xx, yy)) # type: ignore


