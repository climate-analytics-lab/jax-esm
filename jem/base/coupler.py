"""Main coupler class for Earth system model coupling."""

from functools import partial
import time
from typing import Any, Dict, Optional, Callable

import jem.base.mixin as mixin
from jem.base.typing import (
    JEMComponent,
    JEMForcingMapper,
    Workflow,
)

import jax
import jax.numpy as jnp

from jem.utils.bulk_op import unwrap_leading_dims, stack_objects

from jax_tqdm import scan_tqdm
from tqdm import tqdm

# Python Equivalent. See https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
def adhoc_scan(f, init, xs):
    carry = init
    ys = []
    for x in xs:
        _start_time = time.time()

        carry, y = f(carry, x)
        ys.append(y)

        _end_time = time.time()
        _elapsed_time = _end_time - _start_time

    return carry, stack_objects(ys)


def generate_scan_function(jitted: bool):
    return jax.lax.scan if jitted else adhoc_scan


class Coupler:
    """Main coupler for Earth system components.
   
    The engine of JEM. 
    
    Attributes:
        components:
            A dict whose key is the name of the component and the value is the
            instantiated component.
        forcing_mappers: 
            A dict whose key is the name of :code:`JEMForcingMapper`-compatible
            object that is responsible for sending information between 
            components.
    """
   
    timestep: float
    components: Dict[str, JEMComponent]
    forcing_mappers: Dict[str, JEMForcingMapper]
    
    def __init__(
        self,
        timestep: float,
        components: Optional[Dict[str, Any]] = None,
        forcing_mappers: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the coupler.

        Each forcing mapper needs to come with a coupling timestep. Currently,
        all coupling timesteps must be an integer multiple of the minimum one.
        
        Args:
            components: Dictionary of components to couple
            flux_exchangers: A list of flux_exchangers.

        """

        self.timestep = timestep
        self.components = components or {}
        for name, component in self.components.items():
            self.add_component(name, component)

        self.forcing_mappers = forcing_mappers or {}
        for name, forcing_mapper in self.forcing_mappers.items():
            self.add_forcing_mapper(name, forcing_mapper)

    def initialize(
        self,
    ) -> Dict[str, tuple[type, type]]:
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
        workflow: Workflow,
        jitted: bool = True,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> Callable:
        """Advance coupled system by one coupling timestep.

        Args:

        Returns:
            New states after one coupling timestep
        """

        self._verify_name_uniqueness()
        self._verify_workflow(workflow)
        flattened_workflow, _ = jax.tree.flatten(workflow)

        if verbose:
            print("Workflow: ", ", ".join(flattened_workflow))
        
        scan_func = generate_scan_function(jitted=jitted)

        # Get step functions of each component
        component_step_functions = {
            component_name: component.generate_step_function(jitted=jitted)
            for component_name, component in self.components.items()
        }

        def step_function(carry, step_time):
            _, t = step_time

            states = carry["states"]
            forcings = carry["forcings"]
        
            unstacked_predictions = { component_name : [] for component_name in self.components.keys() }
        
            for name in flattened_workflow:
                if name in self.components: 
                    states[name], _predictions = component_step_functions[name](states[name], forcings[name], t)
                    unstacked_predictions[name].append(_predictions)
                elif name in self.forcing_mappers:
                    forcing_mapper = self.forcing_mappers[name]
                    sub_states = { component_name : states[component_name] for component_name in forcing_mapper.involved_component_names }
                    sub_forcings = { component_name : forcings[component_name] for component_name in forcing_mapper.involved_component_names }
                    sub_forcings = forcing_mapper.map_forcings(sub_states, sub_forcings)
                    for name in sub_forcings.keys():
                        forcings[name] = sub_forcings[name]
                else:
                    raise Exception(f"Unknown error: Cannot find `{name}` in components or forcing_mappers.")        

            predictions = {
                name : unwrap_leading_dims(stack_objects(unstacked_predictions[name]), first_n_dim=2)
                for name in unstacked_predictions.keys()
            }
            
            return dict(states=states, forcings=forcings), predictions

        if jitted:
            step_function = jax.jit(step_function)

        return step_function

    def generate_trajectory_function(
        self,
        workflow: Workflow,
        start_time: float,
        end_time: float,
        save_interval_steps=1,
        jitted: bool = True,
        show_progress: bool = False,
        tqdm_kwargs: Dict[str, Any] = dict(),
    ):
        total_time = end_time - start_time
        total_steps = int(total_time / self.timestep)

        if total_steps * self.timestep != total_time:
            raise Exception("timestep has to exactly divide (end_time - start_time).")

        scan_func = generate_scan_function(jitted=jitted)

        coupled_step_function = self.generate_step_function(
            workflow=workflow,
            show_progress=show_progress,
            jitted=jitted,
        )

        steps = jnp.arange(total_steps)
        times = start_time + self.timestep * steps
        if jitted:
            step_times = (steps, times)  # type: ignore
        else:
            step_times = list(zip(steps, times))  # type: ignore

        if show_progress:
            if jitted:
                coupled_step_function = scan_tqdm(n=total_steps, **tqdm_kwargs)(
                    coupled_step_function
                )
            else:
                step_times = tqdm(step_times, **tqdm_kwargs)

        def trajectory_function(
            initial_coupled_state_forcing: Dict[str, tuple[type, type]],
        ):
            final_coupled_state, predictions = scan_func(
                coupled_step_function,
                dict(
                    states={
                        component_name: _state
                        for component_name, (
                            _state,
                            _,
                        ) in initial_coupled_state_forcing.items()
                    },
                    forcings={
                        component_name: _forcing
                        for component_name, (
                            _,
                            _forcing,
                        ) in initial_coupled_state_forcing.items()
                    },
                ),
                xs=step_times,
            )
            predictions = unwrap_leading_dims(predictions, first_n_dim=2)
            return final_coupled_state, predictions

        return trajectory_function

    def add_component(
        self,
        name: str,
        component: Any,
    ) -> None:
        """Add a new component to the coupler.
           If name is not provided, then simply re-extract component names and timesteps
        Args:
            name: Component name
            component: Component instance
            flux_mappings: Optional flux mappings from this component to others
        """

        print(f"Validate mixin of component {name:s}.")
        self.components[name] = JEMComponent(
            raw_component = component,
            name = name,
            **mixin.verify(component, reference_class=JEMComponent, skip=["name", "raw_component"], verbose=True)
        )

        self._validate_components()

    def remove_component(self, name: str) -> None:
        """Remove a component from the coupler.

        Args:
            name: Component name to remove
        """
        if name in self.components:
            del self.components[name]

            # update information
            self.add_component()

    def add_forcing_mapper(
        self,
        name: str,
        forcing_mapper: Any,
    ) -> None:
        """Add a new forcing mapper to the coupler.
        Args:
            name: Forcing mapper name
            forcing_mapper: Forcing mapper
        """
        self.forcing_mappers[name] = JEMForcingMapper(
            raw_forcing_mapper = forcing_mapper,
            name = name,
            **mixin.verify(forcing_mapper, reference_class=JEMForcingMapper, skip=["name", "raw_forcing_mapper"], verbose=True)
        )
        

    def remove_forcing_mapper(self, name: str) -> None:
        """Remove a forcing mapper from the coupler.

        Args:
            name: Forcing mapper name to remove
        """
        if name in self.forcing_mappers:
            del self.forcing_mappers[name]


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
        return {
            component_name : component.predictions_to_xarray(predictions[component_name])
            for component_name, component in self.components.items()
        }

    def get_info(self):
        return {
            "component_info" : {
                component_name : component.get_info() for component_name, component in self.components.items() if hasattr(component, "get_info")
            },
            "forcing_mappers" : "None" if self.forcing_mappers is None else { name: forcing_mapper.get_info() for name, forcing_mapper in self.forcing_mappers.items() } ,
        }
        return info

    def _validate_components(self):
        pass

    def _verify_name_uniqueness(self) -> None:
        
        all_names = list(self.components.keys()) + list(self.forcing_mappers.keys())
        counts = { name : 0 for name in all_names }
        for name in all_names:
            counts[name] += 1

        for name, count in counts.items():
            if count != 1:
                raise Exception(f"The name `{name}` is not unique. There are {count:d} of the same name.")

    def _verify_workflow(
        self,
        workflow: Workflow,
    ) -> None:

        flattened_workflow, _ = jax.tree.flatten(workflow)
        for action in flattened_workflow:
            if not isinstance(action, str):
                raise ValueError("Actions in the workflow have to be strings.")
            if action not in self.components and action not in self.forcing_mappers:
                raise ValueError(f"Action `{action:s}` does not map to any component or forcing mapper.")
        

