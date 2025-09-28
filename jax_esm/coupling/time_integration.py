"""Time integration utilities using jax.lax.scan."""

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jax_esm.components.base import BoundaryFluxes, ComponentState, CoupledComponent


class IntegrationState(NamedTuple):
    """State for time integration."""
    component_states: Dict[str, ComponentState]
    time: float
    step: int


class TimeStepInfo(NamedTuple):
    """Information about time steps for each component."""
    name: str
    timestep: float
    subcycles: int  # Number of subcycles within coupling interval
    
    
class TimeIntegrator:
    """Flexible time integration for coupled Earth system components."""
    
    def __init__(
        self,
        coupling_timestep: float,
        component_timesteps: Dict[str, float],
    ):
        """Initialize time integrator.
        
        Args:
            coupling_timestep: Main coupling time step (seconds)
            component_timesteps: Time steps for each component (seconds)
        """
        self.coupling_timestep = coupling_timestep
        self.component_timesteps = component_timesteps
        
        # Calculate subcycling for each component
        self.timestep_info = self._calculate_subcycles()
    
    def _calculate_subcycles(self) -> Dict[str, TimeStepInfo]:
        """Calculate number of subcycles for each component."""
        timestep_info = {}
        
        for name, dt in self.component_timesteps.items():
            # Ensure component timestep divides coupling timestep evenly
            subcycles = int(self.coupling_timestep / dt)
            if subcycles * dt != self.coupling_timestep:
                raise ValueError(
                    f"Component {name} timestep {dt} does not divide "
                    f"coupling timestep {self.coupling_timestep} evenly"
                )
            
            timestep_info[name] = TimeStepInfo(
                name=name,
                timestep=dt,
                subcycles=subcycles,
            )
        
        return timestep_info
     
    def integrate_step(
        self,
        components: Dict[str, CoupledComponent],
        states: Dict[str, ComponentState],
        flux_exchanger: Callable,
        time: float,
    ) -> Tuple[Dict[str, ComponentState], Dict[str, BoundaryFluxes]]:
        """Integrate all components for one coupling timestep.
        
        Args:
            components: Dictionary of components
            states: Current states of all components
            flux_exchanger: Function to exchange fluxes between components
            time: Current simulation time
            
        Returns:
            Tuple of (new_states, final_fluxes)
        """
        # Initialize fluxes
        current_fluxes = {}
        for name, component in components.items():
            # Get initial boundary conditions
            boundary_fields = component.get_boundary_fields(states[name])
            # Convert to fluxes (simplified - real implementation would compute fluxes)
            current_fluxes[name] = self._fields_to_fluxes(boundary_fields)
        
        # Exchange fluxes between components
        input_fluxes = flux_exchanger(states, current_fluxes)
        
        # Integrate each component with subcycling
        new_states = {}
        output_fluxes = {}
        
        for name, component in components.items():
            info = self.timestep_info[name]
            state = states[name]
            forcing = input_fluxes.get(name)
            
            # Subcycle this component
            state, fluxes = self._subcycle_component(
                component, state, forcing, info.timestep, info.subcycles
            )
            
            new_states[name] = state
            output_fluxes[name] = fluxes
        
        return new_states, output_fluxes
    
    def _subcycle_component(
        self,
        component: CoupledComponent,
        initial_state: ComponentState,
        forcing: BoundaryFluxes,
        dt: float,
        n_steps: int,
    ) -> Tuple[ComponentState, BoundaryFluxes]:
        """Subcycle a component for multiple timesteps.
        
        Args:
            component: Component to integrate
            initial_state: Initial component state
            forcing: Boundary forcing
            dt: Component timestep
            n_steps: Number of steps to take
            
        Returns:
            Tuple of (final_state, average_fluxes)
        """
        def step_fn(carry, _):
            state = carry
            new_state, fluxes = component.step(state, forcing, dt)
            return new_state, fluxes
        
        # Use lax.scan for efficient subcycling
        final_state, flux_history = jax.lax.scan(
            step_fn, initial_state, None, length=n_steps
        )
        
        # Average fluxes over subcycles
        avg_fluxes = jax.tree.map(lambda x: jnp.mean(x, axis=0), flux_history)
        
        return final_state, avg_fluxes
    
    def integrate(
        self,
        components: Dict[str, CoupledComponent],
        initial_states: Dict[str, ComponentState],
        flux_exchanger: Callable,
        start_time: float,
        end_time: float,
    ) -> Tuple[Dict[str, ComponentState], List[IntegrationState]]:
        """Integrate coupled system from start to end time.
        
        Args:
            components: Dictionary of components
            initial_states: Initial states for all components
            flux_exchanger: Function to exchange fluxes
            start_time: Start time (seconds)
            end_time: End time (seconds)
            
        Returns:
            Tuple of (final_states, history)
        """
        n_steps = int((end_time - start_time) / self.coupling_timestep)
        
        def scan_fn(carry, step_idx):
            states, time = carry
            
            # Integrate for one coupling timestep
            new_states, _ = self.integrate_step(
                components, states, flux_exchanger, time
            )
            
            new_time = time + self.coupling_timestep
            
            # Save state snapshot
            snapshot = IntegrationState(
                component_states=new_states,
                time=new_time,
                step=step_idx,
            )
            
            return (new_states, new_time), snapshot
        
        # Run integration using lax.scan
        (final_states, final_time), history = jax.lax.scan(
            scan_fn,
            (initial_states, start_time),
            jnp.arange(n_steps),
        )
        
        return final_states, history
    
    def _fields_to_fluxes(self, fields: Dict[str, Array]) -> BoundaryFluxes:
        """Convert boundary fields to fluxes (placeholder implementation)."""
        # In a real implementation, this would compute actual fluxes
        # from boundary fields (e.g., using bulk formulas)
        return BoundaryFluxes(
            heat=fields.get("surface_temperature", jnp.zeros((1, 1))),
            moisture=fields.get("surface_humidity", jnp.zeros((1, 1))),
            momentum_u=fields.get("surface_wind_u", jnp.zeros((1, 1))),
            momentum_v=fields.get("surface_wind_v", jnp.zeros((1, 1))),
            tracers={},
        )
