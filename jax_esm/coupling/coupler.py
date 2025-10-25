"""Main coupler class for Earth system model coupling."""

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_esm.components.base import AbstractComponentState, Component
from jax_esm.coupling.flux_exchange import FluxExchanger

from dataclasses import dataclass, make_dataclass
import tree_math

from jax_esm.utils.bulk_op import unwrap_leading_dims, stack_objects

# Python Equivalent. See https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
def adhoc_scan(f, init, xs=None, length=None):
    
    if xs is None:
        xs = [1] * length
    carry = init
    ys = []
    for i, x in enumerate(xs):
        
        print(f"The {i:d}-th iteration. ", end="")
        _start_time = time.time()
        
        carry, y = f(carry, x)
        ys.append(y)
        
        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        print(f"Execution time: {_elapsed_time:.1f} seconds.")
        
    return carry, stack_objects(ys)
    

@dataclass
class CouplerConfig:
    """Configuration for coupler."""
    timestep: float  # seconds

class CoupledState:
    ...

class CoupledForcing:
    ...


class Coupler:
    """Main coupler for Earth system components."""
    
    def __init__(
        self,
        config: CouplerConfig,
        components: Dict[str, Component],
        flux_exchangers: Optional[ List[ FluxExchanger ] ] = [],
    ):
        """Initialize the coupler.
        
        Args:
            components: Dictionary of components to couple
            flux_exchangers: A list of flux_exchangers.
            config: CouplerConfig object

        """
        self.components = components
        self.component_names = list(components.keys())
        self.config = config
        
        # Extract component timesteps
        component_timesteps = {
            name: comp.timestep for name, comp in components.items()
        }

        self.coupled_state_class = tree_math.struct(make_dataclass(
            cls_name = "JESMCoupledState",
            fields = [ (component_name, component.component_state_class) for component_name, component in components.items() ],
            bases = (CoupledState,),
        ))

        self.coupled_forcing_class = tree_math.struct(make_dataclass(
            cls_name = "JESMCoupledForcing",
            fields = [ (component_name, component.component_forcing_class) for component_name, component in components.items() ],
            bases = (CoupledForcing,),
        ))

        self.flux_exchangers = flux_exchangers

        
    def initialize(
        self,
    ) -> Dict[str, AbstractComponentState]:
        """Initialize all components.
        
        Args:
            rng_key: JAX random key for initialization
            
        Returns:
            Dictionary of initial states for all components
        """
        
        return self.coupled_state_class(**{
            name : component.initialize() 
            for name, component in self.components.items()
        })
   

    def gen_step_fn(
        self,
        jitted: bool = True,
    ) -> callable:

        """Advance coupled system by one coupling timestep.
        
        Args:
            
        Returns:
            New states after one coupling timestep
        """

        # Get step functions for the three components
        step_functions = { component_name : component.gen_step_fn(jitted = jitted) for component_name, component in self.components.items() }

        def step_fn(cpl_state, t):
            

            # Compute forcing
            forcing_group = { component_name : component.component_forcing_class.zeros() for component_name, component in self.components.items() }
            for flux_exchanger in self.flux_exchangers:
                
                # Only certain information is collected
                state_group = { component_name : getattr(cpl_state, component_name) for component_name, _ in flux_exchanger.components.items() }
                forcing_group = flux_exchanger.transformation(state_group, forcing_group)
            
            # Call forward functions and unpack results directly into dictionaries
            results = {
                component_name: step_fn(getattr(cpl_state, component_name), forcing_group[component_name], t) 
                for component_name, step_fn in step_functions.items()
            }
            
            new_cplstate = self.coupled_state_class(**{name: state for name, (state, _) in results.items()})
            cpl_predictions = {name: pred for name, (_, pred) in results.items()}

            return new_cplstate, cpl_predictions

        return jax.jit(step_fn) if jitted else step_fn


    def run(
        self,
        init_cplstate : Dict[str, AbstractComponentState],
        start_time    : float,
        end_time      : float,
        jax_scan: bool = True,
        save_interval_steps = 1,
    ):

        coupler = self
        
        _start_time = time.time()
        scan_func = jax.lax.scan if jax_scan else adhoc_scan

        timestep = self.config.timestep
        total_time = end_time - start_time
        total_steps = int(total_time / timestep)
        
        if total_steps * timestep != total_time:
            raise Exception("timestep has to exactly divide (end_time - start_time).")

        # The goal should be generate forward function once
        # and reuse it all the time.

        # Currently, atmosphere model output will have strange
        # shape if reuse the forward function. This causes the
        # error during post-processing. Therefore for now, I 
        # fall back to generate forward function every time.
        #
        cpl_step_fn = coupler.gen_step_fn(jitted=jax_scan)
        final_state, predictions = scan_func(
            cpl_step_fn,
            init_cplstate,
            xs=jnp.arange(total_steps),
        )

        predictions = unwrap_leading_dims(predictions, first_n_dim=2)

        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        print(f"Execution time: {_elapsed_time:.1f} seconds.")

        return final_state, predictions
        
    
    def add_component(
        self,
        name: str,
        component: Component,
    ) -> None:
        """Add a new component to the coupler.
        
        Args:
            name: Component name
            component: Component instance
            flux_mappings: Optional flux mappings from this component to others
        """

        if not hasattr(CoupledState, name):
            raise Exception("Unable to add {name:s}. It has to be one of the attribute names of CoupledState.".format(
                name = name,
            ))

        self.components[name] = component
        self.component_names = list(self.components.keys())
        
        # Recreate time integrator with new component
        component_timesteps = {
            n: c.timestep for n, c in self.components.items()
        }
        
        # Revalidate
        self._validate_components()
    
    def remove_component(self, name: str) -> None:
        """Remove a component from the coupler.
        
        Args:
            name: Component name to remove
        """
        if name in self.components:
            del self.components[name]
            self.component_names = list(self.components.keys())
            
            # Recreate time integrator without removed component
            component_timesteps = {
                n: c.timestep for n, c in self.components.items()
            }


    def predictions_to_xarray(
        self,
        predictions,
    ):
        
        """
        A tool function that converts a trajectory into an xarray Dataset.

        Args:
            predictions : The predictions returned from `forward_func`
            
        Returns:
            ds : The resulting xarray dataset.
        """
        d = dict()
        for component_name in self.components.keys():
            component = self.components[component_name]
            d[component_name] = component.predictions_to_xarray(predictions[component_name])

        return d
