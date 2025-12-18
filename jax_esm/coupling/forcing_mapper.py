"""Flux exchange and boundary condition translation utilities."""

from typing import Any, Dict, List, Optional, Tuple, Callable

from jax import Array

from jax_esm.components.base import (
    CoupledComponent,
)


class ForcingMapper:
    """Manages flux exchange and boundary condition translation between components."""

    components: Dict[str, CoupledComponent]
    forcing_mappings: Dict[Tuple[str, str], Dict[str, str]]
    transformations: Dict[Tuple[str, str, str, str], Callable]
    component_forcing_classes: Dict[str, type]

    def __init__(
        self,
        components: Dict[str, CoupledComponent],
        forcing_mappings: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
        transformations: Optional[Dict[Tuple[str, str, str, str], Callable]] = None,
    ):
        """Initialize flux exchanger.

        Args:
            component_names: List of component names
            forcing_mappings: Optional mapping of flux names between components.
                          Keys are (source, target) component pairs,
                          values are dicts mapping source flux names to target names.
            transformations: Optional transformations for fluxes.
                           Keys are (source, target, flux_name) tuples,
                           values are transformation functions.
        """

        self.components = components
        self.component_names = list(self.components.keys())
        self.forcing_mappings = forcing_mappings or {}
        self.transformations = transformations or {}

        self.component_forcing_classes = {
            component_name: component.component_forcing_class
            for component_name, component in components.items()
        }

        # Build connectivity graph
        self.connections = self._build_connections()

    def _build_connections(self) -> Dict[str, List[str]]:
        """Build connectivity graph between components."""
        connections: Dict[str, List[str]] = {name: [] for name in self.component_names}

        for source, target in self.forcing_mappings.keys():
            if source in connections:
                connections[source].append(target)

        return connections

    def map_forcings(
        self,
        component_forcings: Dict[str, type],
        component_states: Dict[str, type],
    ) -> Dict[str, type]:
        """Map fluxes and scalars between components.

        Args:

        Returns:
            Dictionary mapping component names to their input fluxes
        """

        for (
            target_component_name,
            target_component_forcing_class,
        ) in self.component_forcing_classes.items():
            forcing = component_forcings[target_component_name]
            for (
                source_component_name,
                source_component_state,
            ) in component_states.items():
                if source_component_name == target_component_name:
                    continue

                # Get mapping of variable names for this source-target pair
                mapping = self.forcing_mappings.get(
                    (source_component_name, target_component_name), {}
                )
                if not mapping:
                    print(
                        f"Notice: Mapping for {source_component_name:s} -> {target_component_name:s} does not exist."
                    )

                # Apply mappings and transformations
                for source_variable_name, target_variable_name in mapping.items():
                    source_variable = strget(
                        source_component_state, source_variable_name
                    )
                    # Apply transformation if defined
                    transform_key = (
                        source_component_name,
                        target_component_name,
                        source_variable_name,
                        target_variable_name,
                    )
                    if transform_key in self.transformations:
                        print(f"{source_component_name:s}.{source_variable_name} => {target_component_name:s}.{target_variable_name}")
                        source_variable = self.transformations[transform_key](
                            source_variable
                        )

                    strset(
                        forcing,
                        target_variable_name,
                        source_variable,
                    )

            component_forcings[target_component_name] = forcing

        return component_forcings

    def couple_components(
        self,
        component_forcings: Dict[str, type],
        component_states: Dict[str, type],
    ) -> Dict[str, type]:
        """Couple components by remapping forcings with conservation checks.

        Args:
            states: Current states of all components

        Returns:
            A dictionary of forcing of each components
        """

        component_forcings = self.map_forcings(component_forcings, component_states)

        # Optional: Add conservation checks here
        # self._check_conservation(output_fluxes, input_fluxes)

        return component_forcings

    def _check_conservation(
        self,
        output_fluxes: Dict[str, Any],
        input_fluxes: Dict[str, Any],
    ) -> None:
        """Check conservation of fluxes (placeholder for conservation checks)."""
        # This could check that total heat/moisture/momentum is conserved
        # across the coupling interface
        pass

    def add_forcing_mapping(
        self,
        source_component_name: str,
        target_component_name: str,
        mapping: Dict[str, str],
    ) -> None:
        """Add or update flux mapping between components.

        Args:
            source_component_name: Source component name
            target_component_name: Target component name
            mapping: Dictionary mapping source state names to target forcing
        """
        self.forcing_mappings[(source_component_name, target_component_name)] = mapping
        self.connections = self._build_connections()

    def add_transformation(
        self,
        source_component_name: str,
        target_component_name: str,
        source_variable_name: str,
        target_variable_name: str,
        transform_fn: Callable[[Array], Array],
    ) -> None:
        """Add transformation function for a specific flux.

        Args:
            source: Source component name
            target: Target component name
            source_variable_name: Name of the source variable to transform
            target_variable_name: Name of the target variable to transform
            transform_fn: Transformation function
        """
        self.transformations[
            (
                source_component_name,
                target_component_name,
                source_variable_name,
                target_variable_name,
            )
        ] = transform_fn


def strget(obj, flattened_variable_name):
    splitted_names = flattened_variable_name.split(".")

    target = obj
    for splitted_name in splitted_names:
        target = getattr(target, splitted_name)

    return target


def strset(obj, flattened_variable_name, value):
    splitted_names = flattened_variable_name.split(".")

    target = obj
    for splitted_name in splitted_names[:-1]:
        target = getattr(target, splitted_name)

    setattr(target, splitted_names[-1], value)
