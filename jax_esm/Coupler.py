"""Main coupler class for Earth system model coupling."""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from dataclasses import make_dataclass
import tree_math
import time

from jcm.date import DateData, Timestamp, Timedelta

from jax_esm.components.base import Component

from jax_esm.components.util import createPhysicalFieldsClass, createBundledClass

import pandas as pd
import numpy as np

def adhoc_scan(f, init, xs=None, length=None):

    from jax_esm.components.util import stack_objects, concat_objects
    
    if xs is None:
        xs = [1] * length
    carry = init
    ys = []
    for i, x in enumerate(xs):
        print(f"The {i:d}-th iteration. ")
        carry, y = f(carry, x)
        ys.append(y)

    #return carry, np.stack(ys)

    return carry, concat_objects(ys, axis=0)


class Coupler:
    """Main coupler for Earth system components."""
    
    def __init__(
        self,
        components: Dict[str, Component],
        config: Dict[str, any],
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

    def genInitState(self):
        # Collect initial states from components
        init_cplstate = {
            component_name : self.components[component_name].getInitState()
            for component_name in self.components.keys()
        }

        return init_cplstate

        
    def init(self):
        ...

    def record(self, cplstate):       
        for component_name in self.components.keys():
            self.components[component_name].record(getattr(cplstate, component_name))
        
    def genForwardFunc(
        self,
    ):

        # Collect forward functions from components
        sub_forward_func = {
            component_name : self.components[component_name].genForwardFunc()
            for component_name in self.components.keys()
        }
        
        @jax.jit
        def forward_func(cplstate, t):
            
            # Consider meta-programming to dynamically generate `forward_fun`
            # Call forward functions of each component

            new_atmstate, atm_predictions = sub_forward_func["atm"](cplstate, t)
            new_flxstate, flx_predictions = sub_forward_func["flx"](cplstate, t)
            new_ocnstate, ocn_predictions = sub_forward_func["ocn"](cplstate, t)

            new_cplstate = dict(
                atm = new_atmstate,
                flx = new_flxstate,
                ocn = new_ocnstate,
            )

            cpl_predictions = dict(
                atm = atm_predictions,
                flx = flx_predictions,
                ocn = ocn_predictions,
            )

            return new_cplstate, cpl_predictions

        return forward_func


    def run(
        self,
        total_steps,
        timestep   : Timedelta,
        save_interval_steps = 1,
        jax_scan: bool = True,
    ):

        coupler = self
        
        # Collect initial states from components
        init_cplstate = {
            component_name : self.components[component_name].getInitState()
            for component_name in self.components.keys()
        }
        
        _start_time = time.time()

        def step_fn(_cplstate, t):
            return coupler.genForwardFunc()(_cplstate, t)
            

        scan_func = jax.lax.scan if jax_scan else adhoc_scan
        
        final_state, traj = scan_func(
            step_fn,
            init_cplstate,
            length=total_steps,
        )

        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        print(f"Execution time: {_elapsed_time:.1f} seconds.")

        return final_state, traj

    
    def reportComponents(self):

        for i, name in enumerate(self.execution_order):
            component = self.components[name]
            component.report()




        
    