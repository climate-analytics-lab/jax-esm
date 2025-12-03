"""Main coupler class for Earth system model coupling."""

from functools import partial
import time
from typing import Dict, Optional, Callable

import jax
import jax.numpy as jnp

from jax_esm.components.base import ComponentState, Component
from jax_esm.coupling.forcing_mapper import ForcingMapper


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
        components: Optional[ Dict[str, Component] ] = None,
        forcing_mapper: Optional[ForcingMapper] = None,
    ):
        """Initialize the coupler.

        Args:
            components: Dictionary of components to couple
            flux_exchangers: A list of flux_exchangers.
            config: CouplerConfig object

        """
        
        self.coupling_timestep = coupling_timestep
        self.components = components or {}
        for name, component in self.components.items():
            self.add_component(name, component)


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
            name: component.initialize() for name, component in self.components.items()
        }

    def generate_step_function(
        self,
        jitted: bool = True,
    ) -> Callable:
        """Advance coupled system by one coupling timestep.

        Args:

        Returns:
            New states after one coupling timestep
        """

        scan_func = jax.lax.scan if jitted else adhoc_scan

        # Get step functions of each component
        step_functions = {}
        for component_name, component in self.components.items():
            _step_function = component.generate_step_function(jitted=jitted)

            # Closure in a loop is used. Using functools.partial to cache.
            def looped_step_function(state, forcing, t, step_function, component):
                ts = t + component.config.timestep * jnp.arange(
                    int(self.coupling_timestep / component.config.timestep)
                )

                def wrapped_step_function(bundle, t):
                    new_state, history = step_function(
                        bundle["state"], bundle["forcing"], t
                    )
                    return dict(state=new_state, forcing=bundle["forcing"]), history

                _carry, _predictions = scan_func(
                    wrapped_step_function,
                    dict(state=state, forcing=forcing),
                    xs=ts,
                )
                _predictions = unwrap_leading_dims(_predictions, first_n_dim=2)
                return _carry["state"], _predictions

            step_functions[component_name] = partial(
                looped_step_function, step_function=_step_function, component=component
            )

        def step_function(coupled_state, t):
            # Compute forcing
            forcings = self.forcing_mapper.couple_components(coupled_state)

            # Call forward functions and unpack results directly into dictionaries
            results = {
                component_name: step_function(
                    coupled_state[component_name], forcings[component_name], t
                )
                for component_name, step_function in step_functions.items()
            }
            coupled_state = {name: state for name, (state, _) in results.items()}
            coupled_predictions = {
                name: prediction for name, (_, prediction) in results.items()
            }

            return coupled_state, coupled_predictions

        return jax.jit(step_function) if jitted else step_function

    def run(
        self,
        init_coupled_state: Dict[str, ComponentState],
        start_time: float,
        end_time: float,
        jax_scan: bool = True,
        save_interval_steps=1,
    ):
        _start_time = time.time()
        total_time = end_time - start_time
        total_steps = int(total_time / self.coupling_timestep)

        if total_steps * self.coupling_timestep != total_time:
            raise Exception("timestep has to exactly divide (end_time - start_time).")

        scan_func = jax.lax.scan if jax_scan else adhoc_scan
        coupled_step_function = self.generate_step_function(jitted=jax_scan)
        final_coupled_state, predictions = scan_func(
            coupled_step_function,
            init_coupled_state,
            xs=start_time + self.coupling_timestep * jnp.arange(total_steps),
        )
        predictions = unwrap_leading_dims(predictions, first_n_dim=2)

        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        print(f"Execution time: {_elapsed_time:.1f} seconds.")

        return final_coupled_state, predictions

    def add_component(
        self,
        name: Optional[str] = None,
        component: Optional[Component] = None,
    ) -> None:
        """Add a new component to the coupler.
           If name is not provided, then simply re-extract component names and timesteps
        Args:
            name: Component name
            component: Component instance
            flux_mappings: Optional flux mappings from this component to others
        """

        if component is not None:
            if name is None:
                raise ValueError("When component is provided, the name must be given.")
        
            self.components[name] = component

        self.component_names = list(self.components.keys())
        self.component_timesteps = {name: component.config.timestep for name, component in self.components.items()}
        
        self._validate_components()

    def _validate_components(self):

        # Check the compatibility of timestep
        for component_name, component_timestep in self.component_timesteps.items():

            if component_timestep <= 0:
                raise ValueError(
                    f"Timestep of {component_name:s} ({component_timestep:f}) must be a positive number."
                )

            if self.coupling_timestep % component_timestep != 0.0:
                raise ValueError(
                    f"Timestep of {component_name:s} ({component_timestep:f}) is not compatible with coupling timestep {self.coupling_timestep:f}."
                )

        for component_name, component in self.components.items():
            component.validate()

    def remove_component(self, name: str) -> None:
        """Remove a component from the coupler.

        Args:
            name: Component name to remove
        """
        if name in self.components:
            del self.components[name]

            # update information
            self.add_component()

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
            d[component_name] = component.predictions_to_xarray(
                predictions[component_name]
            )

        return d
