"""Flux exchange and boundary condition translation utilities."""
from typing import Any, Dict, List, Optional, Tuple, Callable
from jem.base.typing import (
    Array,
    JEMComponentType,
    JEMForcingMapperType,
    CoupledCarry,
    VariableRegistry,
)

from typeguard import typechecked

RegridderFunction = Callable[[Array], Array]

class BasicForcingMapper:
    """Manages flux exchange and boundary condition translation between components.
    
    Attributes:
        component_names: List of component names
        forcing_mappings: Optional mapping of flux names between components.
            Keys are (source, target) component pairs, values are dicts 
            mapping source flux names to target names.
        regridders: Optional regridders for fluxes.
            Keys are (source, target, variable_name) tuples, values are regridder
            functions. If the variable is nested in additional data structure, variable_name
            can be a dotted representation, such as "surface.sea_surface_temperature", 
            "radiation.longwave_radiation", ... etc. To access the structure, if the underlying
            data struct is a dict, ForcingMapper will use :code:`__getitem__`; if the data struct
            is not a dict, :code:`getattr` will be used.

    Example:

    .. code-block:: python
       :linenos:

       from jem.coupling.regridder import IdentityTransformer
       from jem.coupling.forcing_mapper import ForcingMapper
       
       # ... construct models ...
       atm_model = ... 
       ocn_model = ...
       

       forcing_mapper = ForcingMapper(
           components=dict(
               atm=atm_model,
               ocn=ocn_model,
           )
       )
       forcing_mapper.add_forcing_mapping(
           source = ("atm", "phydata.total_heat_flux"),
           target = ("ocn", "flux.total_heat_flux"),
           regridder = IdentityTransformer(
               source_grid=atm_model.grid,
               target_grid=ocn_model.grid,
           )
       )
       forcing_mapper.add_forcing_mapping(
           source = ("ocn", "prog.sea_surface_temperature"),
           target = ("atm", "scalar.sea_surface_temperature"),
           regridder = IdentityTransformer(
               source_grid=ocn_model.grid,
               target_grid=atm_model.grid,
           )
       )

    """
    components: Dict[str, JEMComponentType]
    forcing_mappings: Dict[Tuple[str, str], Dict[str, str]]
    regridders: Dict[Tuple[str, str, str, str], Callable]
    component_forcing_classes: Dict[str, JEMForcingMapperType]
    component_state_variable_registries: Dict[str, VariableRegistry]
    component_forcing_variable_registries: Dict[str, VariableRegistry]
    involved_component_names: List[str]

    @typechecked
    def __init__(
        self,
        components: Dict[str, JEMComponentType],
        forcing_mappings: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
        regridders: Optional[Dict[Tuple[str, str, str, str], Callable]] = None,
    ):
        """Initialize flux exchanger."""
        self.components = components
        self.component_names = list(self.components.keys())
        self.forcing_mappings = forcing_mappings or {}
        self.regridders = regridders or {}

        # validate
        #check_type(self.component_state_variable_registries, Dict[str, VariableRegistry])
        #check_type(self.component_forcing_variable_registries, Dict[str, VariableRegistry])
 
        # Build connectivity graph
        self.connections = self._build_connections()

    def _build_connections(self) -> Dict[str, List[str]]:
        """Build connectivity graph between components."""
        connections: Dict[str, List[str]] = {name: [] for name in self.component_names}

        for source, target in self.forcing_mappings.keys():
            if source in connections:
                connections[source].append(target)

        return connections

    @typechecked
    def map_forcings(
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

        if not set(self.involved_component_names).issubset(set(coupled_carry.keys())):
            raise Exception(
                f"Not all involved_component_names ({str(self.involved_component_names)})"
                f"can be found in coupled_carry ({str(list(coupled_carry.keys()))})"
            )

        # Crop state and derived out. 
        component_states_and_deriveds = {
            component_name: {
                "state": component_carry["state"],
                "derived": component_carry["derived"],
            } for component_name, component_carry in coupled_carry.items()
        }

        # Loop through each involved component. 
        # If there are N component involves, this loop is N by N.
        # The combination is not valid when source == target
        for target_component_name in coupled_carry.keys():
            target_forcing = coupled_carry[target_component_name]["forcing"]
            for source_component_name in coupled_carry.keys():
                if source_component_name == target_component_name:
                    continue
                
                source_component_state_and_derived = component_states_and_deriveds[source_component_name]
                
                # Get mapping of variable names for this source-target pair
                mapping = self.forcing_mappings.get(
                    (source_component_name, target_component_name), {}
                )
                if not mapping:
                    print(
                        f"Notice: Mapping for {source_component_name:s} -> {target_component_name:s} does not exist."
                    )
                
                # Apply mappings and regridders
                for source_variable_name, target_variable_name in mapping.items():
                    source_variable = strget(
                        source_component_state_and_derived, source_variable_name
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
                        target_forcing,
                        target_variable_name,
                        source_variable,
                    )
            # update forcing
            coupled_carry[target_component_name]["forcing"] = target_forcing
        
        return coupled_carry

    def couple_components(
        self,
        component_forcings: Dict[str, type],
        component_deriveds: Dict[str, type],
        component_states: Dict[str, type],
    ) -> Dict[str, type]:
        """Couple components by remapping forcings with conservation checks.

        Args:
            states: Current states of all components

        Returns:
            A dictionary of forcing of each components
        """

        component_forcings = self.map_forcings(component_forcings, component_deriveds, component_states)

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

    @typechecked
    def add_forcing_mapping(
        self,
        source: Tuple[str, str],
        target: Tuple[str, str],
        regridder: Optional[RegridderFunction] = None,
    ) -> None:
        """Add or update flux mapping between components.

        Args:
            source_component_name: Source component name
            target_component_name: Target component name
            mapping: A dict mapping source state names to target forcing
        """
        source_component_name, source_variable_name = source
        target_component_name, target_variable_name = target
        self.forcing_mappings[(source_component_name, target_component_name)] = {
            source_variable_name: target_variable_name
        }
        self.connections = self._build_connections()

        if regridder is not None:
            
            # Check if regridder is valid
            
            self.add_regridder(
                source_component_name,
                target_component_name,
                source_variable_name,
                target_variable_name,
                regridder,
            )

        involved_component_names = []
        for source_component_name, target_variable_name, _, _ in self.regridders.keys():
            involved_component_names.append(source_component_name)
            involved_component_names.append(target_variable_name)
        
        # Use "set" to achieve uniqueness
        self.involved_component_names = list(set(involved_component_names))

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

        regridder_info = dict()
        for i, key in enumerate(self.regridders.keys()):
            source_component_name, target_component_name, source_variable_name, target_variable_name = key
            regridder_info[f"regridder {i:d}"] = {
                'source variable' : f"{source_component_name}[{source_variable_name}]",
                'target variable' : f"{target_component_name}[{target_variable_name}]",
                'regridder' : self.regridders[key].get_info(),
            }
            


        return {
            "forcing_mappings": self.forcing_mappings, 
            "regridders" : regridder_info,
        }


def strget(obj, flattened_variable_name):
    splitted_names = flattened_variable_name.split(".")
    target = obj
    for splitted_name in splitted_names:
        if isinstance(target, dict):
            target = target[splitted_name]
        else:
            target = getattr(target, splitted_name)

    return target


def strset(obj, flattened_variable_name, value):
    splitted_names = flattened_variable_name.split(".")

    target = obj
    for splitted_name in splitted_names[:-1]:
        if isinstance(target, dict):
            target = target[splitted_name]
        else:
            target = getattr(target, splitted_name)

    if isinstance(target, dict):
        target[splitted_names[-1]] = value
    else:
        setattr(target, splitted_names[-1], value)
