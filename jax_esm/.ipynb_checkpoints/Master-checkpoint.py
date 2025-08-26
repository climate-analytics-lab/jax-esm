"""Main coupler class for Earth system model coupling."""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_esm.components.base import ComponentState, CoupledComponent
from jax_esm.coupling.flux_exchange import FluxExchanger
from jax_esm.coupling.time_integration import IntegrationState, TimeIntegrator

from jax_esm.DataCenter import DataCenter, DataPermission

class Master:
    """Main coupler for Earth system components."""
    
    def __init__(
        self,
        components: Dict[str, CoupledComponent],
        config: Dict[str, any],
        permissions = None,
        #coupling_timestep: float = 3600.0,  # 1 hour default
        #flux_mappings: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
        #flux_transformations: Optional[Dict[Tuple[str, str, str], callable]] = None,
    ):
        """Initialize the coupler.
        
        Args:
            components: Dictionary of components to couple
            coupling_timestep: Coupling time step in seconds
            flux_mappings: Optional custom flux mappings between components
            flux_transformations: Optional flux transformation functions
        """
        self.components = components
        self.execution_order = list(components.keys())
        self.config = config
        self.time = 0.0 # time in sec

        if permissions is None:
            permissions = DataPermission.getTemplatePermission(list(components.keys()))

        self.data_center = DataCenter(
            components = components,
            permissions = permissions,
        )

        for _, component in components.items():
            setattr(component, "data_center", self.data_center)
        #self.component_names = list(components.keys())
        #self.coupling_timestep = coupling_timestep

    def checkConfig(self):
        ...
    
    def checkPlan(self):

        component_used = {
            component_name : 0
            for component_name in self.components.keys()
        }

        for i, name in enumerate(self.execution_order):
            if name not in self.components:
                raise Exception(f"Non existing components: {name:s}.")
            
            component_used[name] += 1

        for component_name, count in component_used.items():
            if count == 0:
                print(f"Warning: Unused component `{component_name:s}`.")
            elif count > 1:
                print(f"Warning: Component `{component_name:s}` is called {count:d} > 1 times.")
        
        return 0
        
    def printPlan(self):

        print("Print execution plan:")
        for i, name in enumerate(self.execution_order):
            print(f"[{i+1:2d}] : {name:s} ")


    def init(self):
        ...
        
    def run(self):
        for i, name in enumerate(self.execution_order):
            component = self.components[name]
            component.run(master=self)
            
        self.time += self.config["time_step"]           
    def reportComponents(self):

        for i, name in enumerate(self.execution_order):
            component = self.components[name]
            component.report()




        
    