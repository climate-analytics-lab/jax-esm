"""Main coupler class for Earth system model coupling."""

from functools import partial
import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_esm.components.base import ComponentState, Component
from jax_esm.coupling.forcing_mapper import ForcingMapper

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
    

class Coupler:
    """Main coupler for Earth system components."""

    def __init__(
        self,
        coupling_timestep: float,
        components: Dict[str, Component],
        forcing_mapper: Optional[ ForcingMapper ] = None,
    ):
        """Initialize the coupler.
        
        Args:
            components: Dictionary of components to couple
            flux_exchangers: A list of flux_exchangers.
            config: CouplerConfig object

        """
        self.components = components
        self.component_names = list(components.keys())
        self.coupling_timestep = coupling_timestep
        
        # Extract component timesteps
        component_timesteps = {
            name: comp.config.timestep for name, comp in components.items()
        }

        # Check the compatibility of timestep
        for component_name, component_timestep in component_timesteps.items():
            if self.coupling_timestep % component_timestep != 0.0:
                raise ValueError(f"Timestep of {component_name:s} ({component_timestep:f}) is not compatible with coupling timestep {self.coupling_timestep:f}.")
 
        self.forcing_mapper = forcing_mapper

        
    def initialize(
        self,
    ) -> Dict[str, ComponentState]:
        """Initialize all components.
        
        Args:
            rng_key: JAX random key for initialization
            
        Returns:
            Dictionary of initial states for all components
        """
        
        return {
            name : component.initialize() 
            for name, component in self.components.items()
        }
   

    def generate_step_function(
        self,
        jitted: bool = True,
    ) -> callable:

        """Advance coupled system by one coupling timestep.
        
        Args:
            
        Returns:
            New states after one coupling timestep
        """
        
        scan_func = jax.lax.scan if jitted else adhoc_scan

        # Get step functions of each component
        step_functions = {}
        for component_name, component in self.components.items():
            step_function = component.generate_step_function(jitted=jitted)
            # Closure in a loop is used. Using functools.partial to cache.
            def looped_step_function(state, forcing, t, step_function, component):
                ts = t + component.config.timestep * jnp.arange(int(self.coupling_timestep / component.config.timestep))
                def wrapped_step_function(bundle, t):
                    new_state, history = step_function(bundle["state"], bundle["forcing"], t)
                    return dict(state=new_state, forcing=bundle["forcing"]), history
                _carry, _predictions = scan_func(
                    wrapped_step_function,
                    dict(state=state, forcing=forcing),
                    xs = ts,
                )
                _predictions = unwrap_leading_dims(_predictions, first_n_dim=2)
                return _carry["state"], _predictions
            step_functions[component_name] = partial(looped_step_function, step_function=step_function, component=component)
        
        def step_function(coupled_state, t):
            
            # Compute forcing
            forcings = self.forcing_mapper.couple_components(coupled_state)
            
            # Call forward functions and unpack results directly into dictionaries
            results = {
                component_name: step_function(coupled_state[component_name], forcings[component_name], t) 
                for component_name, step_function in step_functions.items()
            }
            coupled_state = {name: state for name, (state, _) in results.items()}
            coupled_predictions = {name: prediction for name, (_, prediction) in results.items()}

            return coupled_state, coupled_predictions

        return jax.jit(step_function) if jitted else step_function


    def run(
        self,
        init_coupled_state : Dict[str, ComponentState],
        start_time    : float,
        end_time      : float,
        jax_scan: bool = True,
        save_interval_steps = 1,
    ):

        _start_time = time.time()
        scan_func = jax.lax.scan if jax_scan else adhoc_scan
        
        total_time = end_time - start_time
        total_steps = int(total_time / self.coupling_timestep)
        
        if total_steps * self.coupling_timestep != total_time:
            raise Exception("timestep has to exactly divide (end_time - start_time).")

        coupled_step_function = self.generate_step_function(jitted=jax_scan)
        final_coupled_state, predictions = scan_func(
            coupled_step_function,
            init_coupled_state,
            xs = start_time + self.coupling_timestep * jnp.arange(total_steps),
        )
        predictions = unwrap_leading_dims(predictions, first_n_dim=2)

        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        print(f"Execution time: {_elapsed_time:.1f} seconds.")

        return final_coupled_state, predictions
        
    
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
