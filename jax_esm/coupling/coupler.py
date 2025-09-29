"""Main coupler class for Earth system model coupling."""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_esm.components.base import ComponentState, CoupledComponent
from jax_esm.coupling.flux_exchange import FluxExchanger
from jax_esm.coupling.time_integration import IntegrationState, TimeIntegrator
from dataclasses import dataclass


@dataclass
class CouplerConfig:
    """Configuration for coupler."""
    timestep: float  # seconds

class Coupler:
    """Main coupler for Earth system components."""
    
    def __init__(
        self,
        components: Dict[str, CoupledComponent],
        config: CouplerConfig,
        flux_mappings: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
        flux_transformations: Optional[Dict[Tuple[str, str, str], callable]] = None,
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
        
        # Initialize flux exchanger
        self.flux_exchanger = FluxExchanger(
            component_names=self.component_names,
            flux_mappings=flux_mappings,
            transformations=flux_transformations,
        )
        
        # Initialize time integrator
        self.time_integrator = TimeIntegrator(
            coupling_timestep=config.timestep,
            component_timesteps=component_timesteps,
        )
        
        # Validate component compatibility
        self._validate_components()
    
    def _validate_components(self) -> None:
        """Validate that components can be coupled."""
        # Check that all required fluxes are provided
        for name, component in self.components.items():
            required = set(component.get_required_fluxes())
            
            # Check if required fluxes can be provided by other components
            provided = set()
            for other_name, other_comp in self.components.items():
                if other_name != name:
                    provided.update(other_comp.get_provided_fluxes())
            
            missing = required - provided
            if missing and len(self.components) > 1:
                print(f"Warning: Component {name} requires fluxes {missing} "
                      f"that are not provided by other components")
    
    def initialize(
        self,
        rng_key: jax.random.PRNGKey,
    ) -> Dict[str, ComponentState]:
        """Initialize all components.
        
        Args:
            rng_key: JAX random key for initialization
            
        Returns:
            Dictionary of initial states for all components
        """
        states = {}
        
        # Split random key for each component
        keys = jax.random.split(rng_key, len(self.components))
        
        for (name, component), key in zip(self.components.items(), keys):
            states[name] = component.initialize(key)
        
        return states
   

    def gen_step_fn(
        self,
    ) -> Callable:

        """Advance coupled system by one coupling timestep.
        
        Args:
            
        Returns:
            New states after one coupling timestep
        """

        # Collect forward functions from components
        sub_forward_func = {
            component_name : self.components[component_name].gen_step_fn()
            for component_name in self.components.keys()
        }

        @jax.jit
        def step_fn(cplstate, t):
            
            # Consider meta-programming to dynamically generate `stepforward_fun`
            # Call forward functions of each component

            new_atmstate, atm_predictions = sub_step_fn["atm"](cplstate, t)
            new_flxstate, flx_predictions = sub_step_fn["flx"](cplstate, t)
            new_ocnstate, ocn_predictions = sub_step_fn["ocn"](cplstate, t)

       
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

        return step_fn
        
    def run(
        self,
        initial_states: Dict[str, ComponentState],
        start_time: float,
        end_time: float,
        save_frequency: Optional[int] = None,
    ) -> Tuple[Dict[str, ComponentState], List[IntegrationState]]:
        """Run coupled simulation from start to end time.
        
        Args:
            initial_states: Initial states for all components
            start_time: Start time in seconds
            end_time: End time in seconds
            save_frequency: Save state every N coupling steps (None = save all)
            
        Returns:
            Tuple of (final_states, history)
        """
        final_states, history = self.time_integrator.integrate(
            components=self.components,
            initial_states=initial_states,
            flux_exchanger=self.flux_exchanger.couple_components,
            start_time=start_time,
            end_time=end_time,
        )
        
        # Filter history based on save frequency
        if save_frequency is not None and save_frequency > 1:
            history = [h for i, h in enumerate(history) if i % save_frequency == 0]
        
        return final_states, history
    
    def add_component(
        self,
        name: str,
        component: CoupledComponent,
        flux_mappings: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """Add a new component to the coupler.
        
        Args:
            name: Component name
            component: Component instance
            flux_mappings: Optional flux mappings from this component to others
        """
        self.components[name] = component
        self.component_names = list(self.components.keys())
        
        # Update flux mappings if provided
        if flux_mappings:
            for target, mapping in flux_mappings.items():
                self.flux_exchanger.add_flux_mapping(name, target, mapping)
        
        # Recreate time integrator with new component
        component_timesteps = {
            n: c.timestep for n, c in self.components.items()
        }
        self.time_integrator = TimeIntegrator(
            coupling_timestep=self.config.timestep,
            component_timesteps=component_timesteps,
        )
        
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
            self.time_integrator = TimeIntegrator(
                coupling_timestep=self.config.timestep,
                component_timesteps=component_timesteps,
            )
