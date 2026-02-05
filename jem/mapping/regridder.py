from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Dict, Any, Optional

from jem.base.exceptions import ValidationError
from jem.base.typing import (
    VariableMetadata,
)
from jem.mapping.grid import Grid

from jax import Array
from typeguard import typechecked


class BasicRegridder(ABC):
    """
    Abstract base class for interpolating between climate model grids.

    This class provides a framework for regriding data between different
    grid configurations (e.g., atmosphere to ocean) with built-in validation.

    Attributes:
        source_grid: A Grid object holding the information of the source grid.
        target_grid: A Grid object holding the information of the target grid.
        conservation_tolerance: The precision of tolerance of the conservative mapping
        validate_shape: A flag indicating whether to check the compatibility of the input
            and output array when performing regridation.
        validate_conservation: A flag indicating whether to check the conservation using
            convervation_tolerance
         
    """

    source_grid: Optional[Grid] = None
    target_grid: Optional[Grid] = None

    conservation_tolerance: float
    validate_shape: bool
    validate_conservation: bool

    @typechecked
    def __init__(
        self,
        source_grid: Optional[Grid] = None,
        target_grid: Optional[Grid] = None,
        conservation_tolerance: float = 1e-6,
        validate_shape: bool = True,
        validate_conservation: bool = True,
    ):
        """Initialize the regridder."""

        self.source_grid = source_grid
        self.target_grid = target_grid
        self.conservation_tolerance = conservation_tolerance
        self.validate_shape = validate_shape
        self.validate_conservation = validate_conservation

        # Store validation results
        self.last_validation: Dict[str, Any] = {}

    @abstractmethod
    def regrid(self, data: Array) -> Array:
        """
        Transform data from source grid to target grid.

        This method must be implemented by subclasses to define the
        specific interpolation/mapping method.

        Args:
            data: Data on source grid

        Returns:
            Array: Data that is interpolated to target grid.
        """
        pass

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
        if (not self.source_grid) and (data.shape != self.source_grid.shape):
            raise ValueError(
                f"Ijnput shape {data.shape} does not match source grid "
                f"shape {self.source_grid.shape}"
            )

        # Apply regridation
        result = self.regrid(data)

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

        # Conservation validation
        if self.validate_conservation:
            self._validate_conservation(source_data, target_data)

    def _validate_shape(self, target_data: Array):
        """Validate that output has correct shape."""
        if not self.target_grid:
            raise Exception("target_grid was not provided. Cannot validate shape")

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
        pass

    def _validate_conservation(self, source_data: Array, target_data: Array):
        """
        Validate conservation of integrated quantity.

        For extensive quantities (e.g., heat content), the integral
        over the domain should be conserved during interpolation.
        """
        # Calculate source integral
        if self.source_grid.grid_weights is not None:
            source_mask = (
                self.source_grid.mask
                if self.source_grid.mask is not None
                else jnp.ones_like(source_data, dtype=bool)
            )
            source_integral = jnp.sum(
                source_data * self.source_grid.grid_weights * source_mask
            )
        else:
            source_integral = jnp.sum(source_data)

        # Calculate target integral
        if self.target_grid.grid_weights is not None:
            target_mask = (
                self.target_grid.mask
                if self.target_grid.mask is not None
                else jnp.ones_like(target_data, dtype=bool)
            )
            target_integral = jnp.sum(
                target_data * self.target_grid.grid_weights * target_mask
            )
        else:
            target_integral = jnp.sum(target_data)

        # Check conservation
        if abs(source_integral) > 1e-10:  # Avoid division by very small numbers
            relative_error = abs((target_integral - source_integral) / source_integral)
        else:
            relative_error = abs(target_integral - source_integral)

        self.last_validation["source_integral"] = source_integral
        self.last_validation["target_integral"] = target_integral
        self.last_validation["relative_error"] = relative_error
        self.last_validation["conservation_valid"] = (
            relative_error <= self.conservation_tolerance
        )

        if relative_error > self.conservation_tolerance:
            raise ValidationError(
                f"Conservation violation: relative error {relative_error:.2e} "
                f"exceeds tolerance {self.conservation_tolerance:.2e}. "
                f"Source integral: {source_integral:.6e}, "
                f"Target integral: {target_integral:.6e}"
            )


    def get_info(self):
       
        return {
            'type': str(self.__class__),
            'source_grid' : self.source_grid.get_info() if self.source_grid is not None else None, 
            'target_grid' : self.target_grid.get_info() if self.target_grid is not None else None, 
            'validate_conservation' : self.validate_conservation,
            'validate_shape' : self.validate_shape,
        }
       

class IdentityRegridder(BasicRegridder):
    """Identity mapping (no interpolation)."""

    def regrid(self, data: Array) -> Array:
        return data

    def validate_metadata(
        self, source_metadata: VariableMetadata, target_metadata: VariableMetadata
    ):
        if source_metadata[0] != target_metadata[0]: # shape
            raise ValidationError(f"Source {str(source_metadata[0])} and target metadata {str(target_metadata[0])} must have the same shape")


class BilinearRegridder(BasicRegridder):
    """Simple bilinear interpolation (for demonstration)."""

    def regrid(self, data: Array) -> Array:
        """Apply bilinear interpolation."""
        from scipy.interpolate import RegularRegridder

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
        return interp((xx, yy))


class ConservativeRegridder(BasicRegridder):
    """Conservative remapping (placeholder for actual conservative method)."""

    def regrid(self, data: Array) -> Array:
        """Apply conservative remapping."""
        # This is a simplified example - real conservative remapping
        # would use proper overlap calculations
        from scipy.ndimage import zoom

        # Calculate zoom factors
        zoom_factors = tuple(
            t / s for t, s in zip(self.target_grid.shape, self.source_grid.shape)
        )

        # Use order=1 for conservative-like behavior
        result = zoom(data, zoom_factors, order=1)

        # Rescale to conserve integral
        if (
            self.source_grid.grid_weights is None
            and self.target_grid.grid_weights is None
        ):
            result *= jnp.prod(self.source_grid.shape) / jnp.prod(
                self.target_grid.shape
            )

        return result
