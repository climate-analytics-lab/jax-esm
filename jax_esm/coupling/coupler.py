"""Main coupler class for Earth system model coupling."""

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_esm.components.base import CoupledComponent, AbstractComponentState
from dataclasses import dataclass, make_dataclass
import tree_math

from jax_esm.utils.bulk_op import unwrap_leading_dims

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
        
    return carry, ys
    

@dataclass
class CouplerConfig:
    """Configuration for coupler."""
    timestep: float  # seconds

class AbstractCoupledState:
    ...


class Coupler:
    """Main coupler for Earth system components."""
    
    def __init__(
        self,
        components: Dict[str, CoupledComponent],
        config: CouplerConfig,
    ):
        """Initialize the coupler.
        
        Args:
            components: Dictionary of components to couple
            coupling_timestep: Coupling time step in seconds
            flux_mappings: Optional custom flux mappings between components
            flux_transformations: Optional flux transformation functions
        """
        self.components = components
        self.component_names = list(components.keys())
        self.config = config
        
        # Extract component timesteps
        component_timesteps = {
            name: comp.timestep for name, comp in components.items()
        }

        self.coupled_state_class = tree_math.struct(make_dataclass(
            cls_name = "CoupledState",
            fields = [ (component_name, component.component_state_class) for component_name, component in components.items() ],
            bases = (AbstractCoupledState,),
        ))


        
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
    ) -> callable:

        """Advance coupled system by one coupling timestep.
        
        Args:
            
        Returns:
            New states after one coupling timestep
        """

        # Get step functions for the three components
        atm_step_fn = self.components["atm"].gen_step_fn()
        flx_step_fn = self.components["flx"].gen_step_fn()
        ocn_step_fn = self.components["ocn"].gen_step_fn()

        @jax.jit
        def step_fn(cplstate, t):
            
            # Call forward functions and unpack results directly into dictionaries
            results = {
                name: step_fn(cplstate, t) 
                for name, step_fn in [
                    ("atm", atm_step_fn),
                    ("flx", flx_step_fn), 
                    ("ocn", ocn_step_fn)
                ]
            }
            
            new_cplstate = {name: state for name, (state, _) in results.items()}
            cpl_predictions = {name: pred for name, (_, pred) in results.items()}

            return new_cplstate, cpl_predictions

        return step_fn


    def run(
        self,
        init_cplstate : Dict[str, AbstractComponentState],
        start_time    : float,
        end_time      : float,
        timestep      : float,
        jax_scan: bool = True,
        save_interval_steps = 1,
    ):

        coupler = self
        
        _start_time = time.time()
        scan_func = jax.lax.scan if jax_scan else adhoc_scan

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
        cpl_step_fn = coupler.gen_step_fn()
        final_state, predictions = scan_func(
            cpl_step_fn,
            init_cplstate,
            length=total_steps,
        )

        predictions = unwrap_leading_dims(predictions, first_n_dim=2)

        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        print(f"Execution time: {_elapsed_time:.1f} seconds.")

        return final_state, predictions
        
    
    def add_component(
        self,
        name: str,
        component: CoupledComponent,
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
