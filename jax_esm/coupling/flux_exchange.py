"""Flux exchange and boundary condition translation utilities."""

from typing import Dict, List, Optional, Tuple, Callable

import jax
import jax.numpy as jnp
from jax import Array

from jax_esm.components.base import BoundaryFluxes, ComponentState

def generate_simple_transformer(
    source_componet_name: str,
    target_
):



class FluxExchanger:
    
    """Manages mapping variables between grids of target components with flux computation involved. """
    
    def __init__(
        self,
        coupled_forcing_class  : Coupler.CoupledForcing,
        components             : Dict[ str, Component ],
        source_variables       : Dict[ str, List[ Tuple[str, str] ] ],
        target_variables       : Dict[ str, List[ Tuple[str, str] ] ],
        transformations        : Optional[ Callable ] = None,
    ):
        """Initialize flux exchanger.
        
        Args:
            components : A dictionary of components
            source_variables: A dictionary of what variables will be involved.
                               Keys are component names, values are variables involved.
            target_variables: A dictionary of what forcing will be output.
                              Keys are target component names, and value is a list of forcing names.
            transformations: Optional transformations for fluxes. It takes the source variables and output target forcing.
                             If there is only one source_component and one source variable, and the grids from source and target components have the same grid, then `None` can be given, and the variable will be directly transfer from source to target forcing
        """
        source_component_names = list(source_variables.keys())
        target_component_names = list(target_variables.keys())

        _components = {}
        _components.update({ component_name : component for component_name in source_component_names})
        _components.update({ component_name : component for component_name in target_component_names})

        self.components = _components
        self.source_variables = source_variables
        self.target_variables = target_variables
        self.coupled_forcing_class = coupled_forcing_class

        # TODO: check if same destination flux being output twice


        if transformations is None:
            
            single_source = len(source_component_names) == 1 and len(source_variables[source_component_names[0]]) == 1
            
            source_grid = self.components[source_component_names[0]].config.grid
            same_grid = all([source_grid == self.components[target_component_name].config.grid for target_component_name in target_component_names])
            
            if not (single_source and same_grid):
                raise Exception("Error: transformations can only be None when it is single source and on the same grid.")
             
            source_component_name = source_component_names[0]
            source_variable_name  = source_variable_names[source_component_names[0]][0]
            def default_transformation(cpl_state, cpl_forcing):
                source_data = getattr( getattr(cpl_state, source_variable_name[0] ), source_variable_name[1] ).copy()
                cpl_forcing = cpl_forcing.copy()
                # send information to different components
                for target_component_name, target_variable_names in target_variables.item():
                    target_component_forcing = getter( cpl_forcing )
                    for target_variable_name in target_variable_names:
                        field_group = getter( target_component_forcing, target_variable_name[0] )
                        setter( field_group, target_variable_name, source_data )
                return cpl_forcing

        self.transformations = transformations or [ default_transformation, ]
        
        # Comment this out. Not sure what it is doing
        # Build connectivity graph
        # self.connections = self._build_connections()

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
