from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Optional, Dict, Any, Tuple
from jax import Array
from dataclasses import dataclass

from jax_esm.grid import Grid

class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class GridInterpolator(ABC):
    """
    Abstract base class for interpolating between climate model grids.
    
    This class provides a framework for transforming data between different
    grid configurations (e.g., atmosphere to ocean) with built-in validation.
    """
    
    def __init__(
        self,
        source_grid: Grid,
        target_grid: Grid,
        conservation_tol: float = 1e-6,
        validate_shape: bool = True,
        validate_conservation: bool = True
    ):
        """
        Initialize the interpolator.
        
        Parameters
        ----------
        source_grid : GridInfo
            Information about the source grid
        target_grid : GridInfo
            Information about the target grid
        conservation_tol : float
            Tolerance for conservation checks (relative error)
        validate_shape : bool
            Whether to validate output shape
        validate_conservation : bool
            Whether to validate conservation
        """
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.conservation_tol = conservation_tol
        self.validate_shape = validate_shape
        self.validate_conservation = validate_conservation
        
        # Store validation results
        self.last_validation: Dict[str, Any] = {}
    
    @abstractmethod
    def transform(self, data: Array) -> Array:
        """
        Transform data from source grid to target grid.
        
        This method must be implemented by subclasses to define the
        specific interpolation/mapping method.
        
        Parameters
        ----------
        data : Array
            Data on source grid
            
        Returns
        -------
        Array
            Data interpolated to target grid
        """
        pass
    
    def __call__(self, data: Array) -> Array:
        """
        Apply transformation with validation.
        
        Parameters
        ----------
        data : Array
            Data on source grid
            
        Returns
        -------
        Array
            Validated transformed data
            
        Raises
        ------
        ValidationError
            If validation checks fail
        """
        # Check ijnput shape
        if data.shape != self.source_grid.shape:
            raise ValueError(
                f"Ijnput shape {data.shape} does not match source grid "
                f"shape {self.source_grid.shape}"
            )
        
        # Apply transformation
        result = self.transform(data)
        
        # Validate output
        self._validate(data, result)
        
        return result
    
    def _validate(self, source_data: Array, target_data: Array):
        """
        Perform validation checks on transformed data.
        
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
        if target_data.shape != self.target_grid.shape:
            raise ValidationError(
                f"Output shape {target_data.shape} does not match "
                f"target grid shape {self.target_grid.shape}"
            )
        self.last_validation['shape_valid'] = True
    
    def _validate_conservation(
        self,
        source_data: Array,
        target_data: Array
    ):
        """
        Validate conservation of integrated quantity.
        
        For extensive quantities (e.g., heat content), the integral
        over the domain should be conserved during interpolation.
        """
        # Calculate source integral
        if self.source_grid.grid_weights is not None:
            source_mask = self.source_grid.mask if self.source_grid.mask is not None else jnp.ones_like(source_data, dtype=bool)
            source_integral = jnp.sum(
                source_data * self.source_grid.grid_weights * source_mask
            )
        else:
            source_integral = jnp.sum(source_data)
        
        # Calculate target integral
        if self.target_grid.grid_weights is not None:
            target_mask = self.target_grid.mask if self.target_grid.mask is not None else jnp.ones_like(target_data, dtype=bool)
            target_integral = jnp.sum(
                target_data * self.target_grid.grid_weights * target_mask
            )
        else:
            target_integral = jnp.sum(target_data)
        
        # Check conservation
        if abs(source_integral) > 1e-10:  # Avoid division by very small numbers
            relative_error = abs(
                (target_integral - source_integral) / source_integral
            )
        else:
            relative_error = abs(target_integral - source_integral)
        
        self.last_validation['source_integral'] = source_integral
        self.last_validation['target_integral'] = target_integral
        self.last_validation['relative_error'] = relative_error
        self.last_validation['conservation_valid'] = relative_error <= self.conservation_tol
        
        if relative_error > self.conservation_tol:
            raise ValidationError(
                f"Conservation violation: relative error {relative_error:.2e} "
                f"exceeds tolerance {self.conservation_tol:.2e}. "
                f"Source integral: {source_integral:.6e}, "
                f"Target integral: {target_integral:.6e}"
            )


# Example implementations

class IdentityInterpolator(GridInterpolator):
    """Identity mapping (no interpolation)."""
    
    def transform(self, data: Array) -> Array:
        return data


class BilinearInterpolator(GridInterpolator):
    """Simple bilinear interpolation (for demonstration)."""
    
    def transform(self, data: Array) -> Array:
        """Apply bilinear interpolation."""
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator for source grid
        source_shape = self.source_grid.shape
        x = jnp.linspace(0, 1, source_shape[0])
        y = jnp.linspace(0, 1, source_shape[1])
        interp = RegularGridInterpolator((x, y), data, method='linear')
        
        # Create target grid coordinates
        target_shape = self.target_grid.shape
        x_new = jnp.linspace(0, 1, target_shape[0])
        y_new = jnp.linspace(0, 1, target_shape[1])
        xx, yy = jnp.meshgrid(x_new, y_new, indexing='ij')
        
        # Interpolate
        return interp((xx, yy))


class ConservativeInterpolator(GridInterpolator):
    """Conservative remapping (placeholder for actual conservative method)."""
    
    def transform(self, data: Array) -> Array:
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
        if self.source_grid.grid_weights is None and self.target_grid.grid_weights is None:
            result *= jnp.prod(self.source_grid.shape) / jnp.prod(self.target_grid.shape)
        
        return result


# Example usage
if __name__ == "__main__":
    # Define grids
    atm_grid = GridInfo(
        shape=(180, 360),
        grid_weights=jnp.ones((180, 360))  # Simplified, should be cos(lat)
    )
    
    ocean_grid = GridInfo(
        shape=(100, 200),
        grid_weights=jnp.ones((100, 200))
    )
    
    # Create interpolator
    interpolator = ConservativeInterpolator(
        source_grid=atm_grid,
        target_grid=ocean_grid,
        conservation_tol=1e-3
    )
    
    # Test data
    test_data = jnp.random.randn(180, 360)
    
    try:
        result = interpolator(test_data)
        print(f"Transformation successful!")
        print(f"Output shape: {result.shape}")
        print(f"Validation results: {interpolator.last_validation}")
    except ValidationError as e:
        print(f"Validation failed: {e}")
