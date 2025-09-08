"""Main coupler class for Earth system model coupling."""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_esm.components.base import ComponentState, CoupledComponent
from jax_esm.coupling.flux_exchange import FluxExchanger
from jax_esm.coupling.time_integration import IntegrationState, TimeIntegrator

from jax_esm.DataCenter import DataCenter, DataPermission

from dataclasses import make_dataclass
import tree_math

def stack_objects(objs):
    # objs is a list of pytrees with same structure
    leaves = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return leaves


class Coupler:
    """Main coupler for Earth system components."""


    @classmethod
    def createCoupledClass(
        cls,
        cls_name,
        name_cls_pairs,
        bases = (),
    ):
        print("fields = ", name_cls_pairs)
        cls = make_dataclass(
            cls_name = cls_name,
            fields = name_cls_pairs,
            bases = bases,
        )
        
        @classmethod
        def zeros(_cls, **kwargs):
            
            init_args = dict()
            for varname, cls in name_cls_pairs:
                if (varname in kwargs) and (kwargs[varname] is not None):
                    init_args[varname] = kwargs[varname]
                else:
                    init_args[varname] = cls.zeros()
                    
            return _cls(**init_args)
    
        @classmethod
        def ones(_cls, **kwargs):
            
            init_args = dict()
            for varname, cls in name_cls_pairs:
                if (varname in kwargs) and (kwargs[varname] is not None):
                    init_args[varname] = kwargs[varname]
                else:
                    init_args[varname] = cls.ones()
    
            return _cls(**init_args)
    
        def copy(self, **kwargs):
            
            init_args = dict()
            for varname, cls in name_cls_pairs:
                if (varname in kwargs) and (kwargs[varname] is not None):
                    init_args[varname] = kwargs[varname]
                else:
                    init_args[varname] = getattr(self, varname)
    
            return type(self)(**init_args)
    
        cls.zeros = zeros
        cls.ones = ones
        cls.copy = copy
        
        return tree_math.struct(cls)
    
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

        name_cls_pairs = [ (component_name, self.components[component_name].stateDiagClass) for component_name in self.components.keys() ]
        self.stateClass = self.__class__.createCoupledClass(
            cls_name = "CoupledState",
            name_cls_pairs = name_cls_pairs,
        )

        kwargs = {
            component_name : self.components[component_name].state_diag for component_name in self.components.keys()
        }
        
        self.state = self.stateClass.zeros(**kwargs)

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

    def record(self, cplstate):       
        for component_name in self.components.keys():
            self.components[component_name].record(getattr(cplstate, component_name))

    
    def genForwardFunc(self, first_time=False):

        sub_forward_func = {
            component_name : self.components[component_name].genForwardFunc()
            for component_name in self.components.keys()
        }

        if first_time:
            @jax.jit
            def forward_func(cplstate):
                
                new_atmstate = sub_forward_func["atm"](cplstate)
    
                new_cplstate = cplstate.copy(
                    atm = new_atmstate,
                )
            
                return new_cplstate
        else:
            # Consider meta-programming to dynamically generate `forward_fun`
            @jax.jit
            def forward_func(cplstate):
                
                new_atm_sd = sub_forward_func["atm"](cplstate)
                new_flx_sd = sub_forward_func["flx"](cplstate)
                new_ocn_sd = sub_forward_func["ocn"](cplstate)

                #print("New state updating...")
                new_cplstate = cplstate.copy(
                    atm = new_atm_sd,
                    flx = new_flx_sd,
                    ocn = new_ocn_sd,
                )
                #print("Return cplstate")
                
                return new_cplstate
                
        return forward_func
    
    def run(self):
        for i, name in enumerate(self.execution_order):
            component = self.components[name]
            component.run(master=self)
            
        self.time += self.config["time_step"]           
    def reportComponents(self):

        for i, name in enumerate(self.execution_order):
            component = self.components[name]
            component.report()




        
    