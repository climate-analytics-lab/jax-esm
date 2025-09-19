"""Flux exchange and boundary condition translation utilities."""

from typing import Dict, List, Optional, Tuple, Callable

import jax
import jax.numpy as jnp
from jax import Array

from jax_esm.components.base import BoundaryFluxes, ComponentState


class FluxExchanger:
    """Manages flux exchange and boundary condition translation between components."""
    
    def __init__(
        self,
        component_names: List[str],
        flux_mappings: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
        transformations: Optional[Dict[Tuple[str, str, str], Callable]] = None,
    ):
        """Initialize flux exchanger.
        
        Args:
            component_names: List of component names
            flux_mappings: Optional mapping of flux names between components.
                          Keys are (source, target) component pairs,
                          values are dicts mapping source flux names to target names.
            transformations: Optional transformations for fluxes.
                           Keys are (source, target, flux_name) tuples,
                           values are transformation functions.
        """
        self.component_names = component_names
        self.flux_mappings = flux_mappings or {}
        self.transformations = transformations or {}
        
        # Build connectivity graph
        self.connections = self._build_connections()
    
    def _build_connections(self) -> Dict[str, List[str]]:
        """Build connectivity graph between components."""
        connections = {name: [] for name in self.component_names}
        
        for (source, target) in self.flux_mappings.keys():
            if source in connections:
                connections[source].append(target)
        
        return connections
    
    def exchange_fluxes(
        self,
        component_fluxes: Dict[str, BoundaryFluxes],
    ) -> Dict[str, BoundaryFluxes]:
        """Exchange fluxes between components.
        
        Args:
            component_fluxes: Dictionary mapping component names to their output fluxes
            
        Returns:
            Dictionary mapping component names to their input fluxes
        """
        input_fluxes = {}
        
        for target in self.component_names:
            # Collect fluxes from all sources for this target
            collected_fluxes = {
                "heat": jnp.zeros_like(next(iter(component_fluxes.values())).heat),
                "moisture": jnp.zeros_like(next(iter(component_fluxes.values())).moisture),
                "momentum_u": jnp.zeros_like(next(iter(component_fluxes.values())).momentum_u),
                "momentum_v": jnp.zeros_like(next(iter(component_fluxes.values())).momentum_v),
                "tracers": {},
            }
            
            for source in self.component_names:
                if source == target:
                    continue
                    
                # Get flux mapping for this source-target pair
                mapping = self.flux_mappings.get((source, target), {})
                if not mapping:
                    # Default identity mapping
                    mapping = {
                        "heat": "heat",
                        "moisture": "moisture",
                        "momentum_u": "momentum_u",
                        "momentum_v": "momentum_v",
                    }
                
                source_fluxes = component_fluxes.get(source)
                if source_fluxes is None:
                    continue
                
                # Apply mappings and transformations
                for source_name, target_name in mapping.items():
                    if hasattr(source_fluxes, source_name):
                        flux_value = getattr(source_fluxes, source_name)
                        
                        # Apply transformation if defined
                        transform_key = (source, target, source_name)
                        if transform_key in self.transformations:
                            flux_value = self.transformations[transform_key](flux_value)
                        
                        # Accumulate flux (for multiple sources)
                        if target_name in collected_fluxes:
                            collected_fluxes[target_name] = (
                                collected_fluxes[target_name] + flux_value
                            )
                        elif target_name in collected_fluxes["tracers"]:
                            collected_fluxes["tracers"][target_name] = (
                                collected_fluxes["tracers"][target_name] + flux_value
                            )
                        else:
                            collected_fluxes["tracers"][target_name] = flux_value
            
            input_fluxes[target] = BoundaryFluxes(**collected_fluxes)
        
        return input_fluxes
    
    def couple_components(
        self,
        states: Dict[str, ComponentState],
        output_fluxes: Dict[str, BoundaryFluxes],
    ) -> Dict[str, BoundaryFluxes]:
        """Couple components by exchanging fluxes with conservation checks.
        
        Args:
            states: Current states of all components
            output_fluxes: Output fluxes from all components
            
        Returns:
            Input fluxes for all components
        """
        # Exchange fluxes
        input_fluxes = self.exchange_fluxes(output_fluxes)
        
        # Optional: Add conservation checks here
        self._check_conservation(output_fluxes, input_fluxes)
        
        return input_fluxes
    
    def _check_conservation(
        self,
        output_fluxes: Dict[str, BoundaryFluxes],
        input_fluxes: Dict[str, BoundaryFluxes],
    ) -> None:
        """Check conservation of fluxes (placeholder for conservation checks)."""
        # This could check that total heat/moisture/momentum is conserved
        # across the coupling interface
        pass
    
    def add_flux_mapping(
        self,
        source: str,
        target: str,
        mapping: Dict[str, str],
    ) -> None:
        """Add or update flux mapping between components.
        
        Args:
            source: Source component name
            target: Target component name
            mapping: Dictionary mapping source flux names to target names
        """
        self.flux_mappings[(source, target)] = mapping
        self.connections = self._build_connections()
    
    def add_transformation(
        self,
        source: str,
        target: str,
        flux_name: str,
        transform_fn: Callable[[Array], Array],
    ) -> None:
        """Add transformation function for a specific flux.
        
        Args:
            source: Source component name
            target: Target component name
            flux_name: Name of the flux to transform
            transform_fn: Transformation function
        """
        self.transformations[(source, target, flux_name)] = transform_fn