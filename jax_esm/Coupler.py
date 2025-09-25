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

        self.stateClass = createBundledClass(
            cls_name = "CoupledState",
            name_cls_pairs = [ (component_name, self.components[component_name].stateDiagClass) for component_name in self.components.keys() ],
        )

        kwargs = {
            component_name : self.components[component_name].state_diag for component_name in self.components.keys()
        }

        
        # Initialize
        self.state = self.stateClass.zeros(**{
            component_name : self.components[component_name].state_diag for component_name in self.components.keys()
        })


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
        
    def genForwardFunc(
        self,
        first_time: bool,
    ):

        # Collect forward functions from components
        sub_forward_func = {
            component_name : self.components[component_name].genForwardFunc()
            for component_name in self.components.keys()
        }

        # For some reason I cannot obtain a valid PhysicsState at the first time step
        # So the flux and ocean model does not run until the second step
        if False and first_time:
            @jax.jit
            def forward_func(cpl):
                
                new_atm = sub_forward_func["atm"](cpl)
                new_cpl = cpl.copy(
                    atm = new_atm,
                )
            
                return new_cpl
        else:
            
            @jax.jit
            def forward_func(cpl):
                
                # Consider meta-programming to dynamically generate `forward_fun`
                # Call forward functions of each component
                new_atm = sub_forward_func["atm"](cpl)
                new_flx = sub_forward_func["flx"](cpl)
                new_ocn = sub_forward_func["ocn"](cpl)

                new_cpl = cpl.copy(
                    atm = new_atm,
                    flx = new_flx,
                    ocn = new_ocn,
                )
                
                return new_cpl
                
        return forward_func

    
    def run_without_using_scan(
        self,
        total_steps,
        begin_time : Timestamp,
        timestep   : Timedelta,
        save_interval_steps = 1,
    ):

        coupler = self
        cplstate = coupler.state.copy()

        cpl_forward_func = None
        
        start_time = time.time()

        time_now = begin_time
        for step in range(total_steps):
        
            _start_time = time.time()
            time_now_str = time_now.to_datetime64().astype('datetime64[us]').item().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Coupler Step: {step+1:d}/{total_steps:d}. DateTime = {time_now_str:s}. ", end="")
            
            cpl_forward_func = coupler.genForwardFunc(
                first_time = step == 0
            )
            cplstate = cpl_forward_func(cplstate)

            if step % save_interval_steps == 0:
                print("Save couple state. ", end="")
                coupler.record(cplstate)

            time_now += timestep
            _end_time = time.time()
            _elapsed_time = _end_time - _start_time
            print(f"Execution time: {_elapsed_time:.1f} seconds.")
      
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.1f} seconds.")


    def run_using_scan(
        self,
        total_steps,
        begin_time : Timestamp,
        timestep   : Timedelta,
        save_interval_steps = 1,
    ):

        coupler = self
        cplstate = coupler.state.copy()

        cpl_forward_func = None
        
        start_time = time.time()

        time_now = begin_time

        #final_state, traj = jax.lax.scan(f, init, xs, length=None)

        
        for step in range(total_steps):
        
            _start_time = time.time()
            time_now_str = time_now.to_datetime64().astype('datetime64[us]').item().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Coupler Step: {step+1:d}/{total_steps:d}. DateTime = {time_now_str:s}. ", end="")
            
            cpl_forward_func = coupler.genForwardFunc(
                begin_time = time_now,
                first_time=step==0
            )
            cplstate = cpl_forward_func(cplstate)

            if step % save_interval_steps == 0:
                print("Save couple state. ", end="")
                coupler.record(cplstate)

            time_now += timestep
            _end_time = time.time()
            _elapsed_time = _end_time - _start_time
            print(f"Execution time: {_elapsed_time:.1f} seconds.")
      
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.1f} seconds.")

        integrate_fn = jax.jit(dino_ti.trajectory_from_step(
            jax.checkpoint(step_fn),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            start_with_input=True,
            post_process_fn=lambda state: self._post_process(state, boundaries),
        ))


    
    def reportComponents(self):

        for i, name in enumerate(self.execution_order):
            component = self.components[name]
            component.report()




        
    