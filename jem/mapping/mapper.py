"""Flux exchange and boundary condition translation utilities."""
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from jax.typing import ArrayLike as Array
from typeguard import typechecked

from jem.base.typing import (
    CoupledCarry,
    JEMComponent,
)

RegridderFunction = Callable[[Array], Array]
JEMComponentType = JEMComponent | Any

class BasicMapper:
    """Manages flux exchange and boundary condition translation between components.
    
    Attributes:
        component_names: List of component names
        mappings: Optional mapping of flux names between components.
            Keys are (source, target) component pairs, values are dicts 
            mapping source flux names to target names.
        regridders: Optional regridders for fluxes.
            Keys are (source, target, variable_name) tuples, values are regridder
            functions. If the variable is nested in additional data structure, variable_name
            can be a dotted representation, such as "surface.sea_surface_temperature", 
            "radiation.longwave_radiation", ... etc. To access the structure, if the underlying
            data struct is a dict, Mapper will use :code:`__getitem__`; if the data struct
            is not a dict, :code:`getattr` will be used.

    Example:

    .. code-block:: python
       :linenos:

       from jem.mapping.regridder import IdentityRegridder
       from jem.mapping.mapper import BasicMapper

       mapper = BasicMapper()
       mapper.add_mapping(
           source = ("atm", "phydata.total_heat_flux"),
           target = ("ocn", "flux.total_heat_flux"),
           regridder = IdentityRegridder(),
       )
       mapper.add_mapping(
           source = ("ocn", "prog.sea_surface_temperature"),
           target = ("atm", "scalar.sea_surface_temperature"),
           regridder = IdentityRegridder(),
       )

    """
    components: dict[str, JEMComponentType]
    mappings: dict[tuple[str, str], dict[str, str]]
    regridders: dict[tuple[str, str, str, str], Callable]

    @typechecked
    def __init__(
        self,
    ):
        """Initialize flux exchanger."""
        self.mappings = {}
        self.regridders = {}
        
        # Build connectivity graph
        self._build_connections()

    def _build_connections(self) -> dict[str, list[str]]:
        """Build connectivity graph between components."""
        connections: dict[str, list[str]] = {}
        for source, target in self.mappings:
            if source not in connections:
                connections[source] = []
            connections[source].append(target)

        return connections

    @typechecked
    def __call__(
        self,
        coupled_carry: CoupledCarry,
    ) -> CoupledCarry:
        """Map fluxes and scalars between components.

        Args:
            component_states: A dict that maps component name to
                component state class.
            component_forcings: A dict that maps component name
                to component forcing class.

        Returns:
            A dict that maps component names to the resulting
            forcing obejcts.
        """

        # Loop through each involved component. 
        # If there are N component involves, this loop is N by N.
        # The combination is not valid when source == target
        for target_component_name in coupled_carry:
            target_carry = coupled_carry[target_component_name]
            for source_component_name in coupled_carry:
                if source_component_name == target_component_name:
                    continue
                
                source_component_carry = coupled_carry[source_component_name]
                
                # Get mapping of variable names for this source-target pair
                mapping = self.mappings.get(
                    (source_component_name, target_component_name), {}
                )
                if not mapping:
                    print(
                        f"Notice: Mapping for {source_component_name:s} -> {target_component_name:s} does not exist."
                    )
                
                # Apply mappings and regridders
                for source_variable_name, target_variable_name in mapping.items():
                    source_variable = strget(
                        source_component_carry, source_variable_name
                    )
                    # Apply regridder if defined
                    regrid_key = (
                        source_component_name,
                        target_component_name,
                        source_variable_name,
                        target_variable_name,
                    )
                    if regrid_key in self.regridders:
                        source_variable = self.regridders[regrid_key](
                            source_variable
                        )
                    print(
                        f"Doing: {source_component_name:s}.{source_variable_name:s} -> {target_component_name:s}.{target_variable_name:s}"
                    )
                    strset(
                        target_carry,
                        target_variable_name,
                        source_variable,
                    )
            # update carry
            coupled_carry[target_component_name] = target_carry
        
        return coupled_carry

    def _check_conservation(
        self,
        output_fluxes: dict[str, Any],
        input_fluxes: dict[str, Any],
    ) -> None:
        """Check conservation of fluxes (placeholder for conservation checks)."""
        # This could check that total heat/moisture/momentum is conserved
        # across the coupling interface

    @typechecked
    def add_mapping(
        self,
        source: tuple[str, str],
        target: tuple[str, str],
        regridder: RegridderFunction | None = None,
    ) -> None:
        """Add or update flux mapping between components.

        Args:
            source_component_name: Source component name
            target_component_name: Target component name
            mapping: A dict mapping source state names to target forcing
        """
        source_component_name, source_variable_name = source
        target_component_name, target_variable_name = target
        pair = (source_component_name, target_component_name)
        if pair not in self.mappings:
            self.mappings[pair] = {}
        self.mappings[pair][source_variable_name] = target_variable_name
        self.connections = self._build_connections()

        regridder = regridder or ( lambda x: x )
        
        self.add_regridder(
            source_component_name,
            target_component_name,
            source_variable_name,
            target_variable_name,
            regridder,
        )

    @typechecked
    def add_regridder(
        self,
        source_component_name: str,
        target_component_name: str,
        source_variable_name: str,
        target_variable_name: str,
        regridder: RegridderFunction,
    ) -> None:
        """Add regridder function for a specific flux.

        Args:
            source_component_name: Source component name
            target_component_name: Target component name
            source_variable_name: Name of the source variable to regrid
            target_variable_name: Name of the target variable to regrid
            regridder: A :code:`Transformer` that will be used to regrid the input.
        """

        self.regridders[
            (
                source_component_name,
                target_component_name,
                source_variable_name,
                target_variable_name,
            )
        ] = regridder

    def get_info(self):
        regridder_info = {}
        for i, key in enumerate(self.regridders.keys()):
            source_component_name, target_component_name, source_variable_name, target_variable_name = key
            regridder_info[f"regridder {i:d}"] = {
                'source variable' : f"{source_component_name}[{source_variable_name}]",
                'target variable' : f"{target_component_name}[{target_variable_name}]",
            }

            if getattr(self.regridders, "get_info", None) is not None:
                regridder_info['regridder_info'] = self.regridders[key].get_info()
            else:
                regridder_info['regridder_info'] = "get_info not provided."

        return {
            "mappings": self.mappings, 
            "regridders" : regridder_info,
        }


def strget(obj, flattened_variable_name):
    splitted_names = flattened_variable_name.split(".")
    target = obj
    for splitted_name in splitted_names:
        if isinstance(target, Sequence):
            target = target[int(splitted_name)]
        elif isinstance(target, Mapping):
            target = target[splitted_name]
        else:
            target = getattr(target, splitted_name)

    return target


def strset(obj, flattened_variable_name, value):
    splitted_names = flattened_variable_name.split(".")
    target = strget(obj, ".".join(splitted_names[:-1]))
    if isinstance(target, Sequence):
        try:
            target[int(splitted_names[-1])] = value
        except (TypeError, IndexError):
            print(f"Cannot update: {flattened_variable_name}. Maybe the leaf is not mutable?")
            raise
    elif isinstance(target, Mapping):
        if splitted_names[-1] not in target:
            raise KeyError(f"Error: Cannot find {flattened_variable_name:s}.")
        target[splitted_names[-1]] = value
    else:
        if not hasattr(target, splitted_names[-1]):
            raise AttributeError(f"Error: Cannot find {flattened_variable_name:s}.")
        setattr(target, splitted_names[-1], value)
