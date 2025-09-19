"""Boundary condition translation utilities."""

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


class BoundaryConditionTranslator:
    """Translates boundary conditions between different components."""
    
    def __init__(self):
        """Initialize boundary condition translator."""
        self.transformations = {}
        self._register_default_transformations()
    
    def _register_default_transformations(self):
        """Register default boundary condition transformations."""
        # Atmosphere to ocean transformations
        self.register_transformation(
            "atmosphere", "ocean", "wind_stress",
            self._compute_wind_stress
        )
        self.register_transformation(
            "atmosphere", "ocean", "heat_flux",
            self._atmosphere_to_ocean_heat
        )
        
        # Ocean to atmosphere transformations
        self.register_transformation(
            "ocean", "atmosphere", "sst",
            lambda x: x  # Direct pass-through
        )
        
        # Atmosphere to land transformations
        self.register_transformation(
            "atmosphere", "land", "precipitation",
            lambda x: x  # Direct pass-through
        )
        self.register_transformation(
            "atmosphere", "land", "radiation",
            self._split_radiation
        )
        
        # Land to atmosphere transformations
        self.register_transformation(
            "land", "atmosphere", "evaporation",
            lambda x: x  # Direct pass-through
        )
    
    def register_transformation(
        self,
        source_component: str,
        target_component: str,
        variable: str,
        transform_fn: Callable[[Array], Array],
    ):
        """Register a transformation function for a specific variable.
        
        Args:
            source_component: Source component name
            target_component: Target component name
            variable: Variable name
            transform_fn: Transformation function
        """
        key = (source_component, target_component, variable)
        self.transformations[key] = transform_fn
    
    def translate(
        self,
        source_component: str,
        target_component: str,
        variable: str,
        data: Array,
        metadata: Optional[Dict] = None,
    ) -> Array:
        """Translate boundary condition from source to target component.
        
        Args:
            source_component: Source component name
            target_component: Target component name
            variable: Variable name
            data: Variable data to translate
            metadata: Optional metadata for transformation
            
        Returns:
            Translated data for target component
        """
        key = (source_component, target_component, variable)
        
        if key in self.transformations:
            transform_fn = self.transformations[key]
            if metadata:
                # Some transformations might need metadata
                return transform_fn(data, metadata)
            else:
                return transform_fn(data)
        else:
            # Default: pass through unchanged
            return data
    
    def _compute_wind_stress(self, wind_speed: Array) -> Array:
        """Compute wind stress from wind speed.
        
        Args:
            wind_speed: Wind speed (m/s)
            
        Returns:
            Wind stress (N/m²)
        """
        # Simplified bulk formula
        rho_air = 1.225  # kg/m³
        cd = 0.001  # Drag coefficient (simplified)
        
        wind_magnitude = jnp.sqrt(wind_speed[..., 0]**2 + wind_speed[..., 1]**2)
        stress_magnitude = rho_air * cd * wind_magnitude**2
        
        # Compute stress components
        stress = jnp.stack([
            stress_magnitude * wind_speed[..., 0] / (wind_magnitude + 1e-10),
            stress_magnitude * wind_speed[..., 1] / (wind_magnitude + 1e-10),
        ], axis=-1)
        
        return stress
    
    def _atmosphere_to_ocean_heat(self, heat_flux: Array) -> Array:
        """Convert atmospheric heat flux convention to ocean convention.
        
        Args:
            heat_flux: Heat flux from atmosphere (positive downward)
            
        Returns:
            Heat flux for ocean (positive into ocean)
        """
        # Ocean models often use opposite sign convention
        return -heat_flux
    
    def _split_radiation(self, radiation: Array) -> Dict[str, Array]:
        """Split total radiation into components for land model.
        
        Args:
            radiation: Total radiation flux
            
        Returns:
            Dictionary with shortwave and longwave components
        """
        # Simplified split (real implementation would be more sophisticated)
        return {
            "shortwave": radiation * 0.6,
            "longwave": radiation * 0.4,
        }
    
    def get_required_transformations(
        self,
        source_component: str,
        target_component: str,
    ) -> Dict[str, Callable]:
        """Get all transformations between two components.
        
        Args:
            source_component: Source component name
            target_component: Target component name
            
        Returns:
            Dictionary of variable names to transformation functions
        """
        transformations = {}
        
        for (src, tgt, var), fn in self.transformations.items():
            if src == source_component and tgt == target_component:
                transformations[var] = fn
        
        return transformations