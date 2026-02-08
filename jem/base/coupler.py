"""Main coupler class for Earth system model coupling."""

import time
from typing import Any, Dict, Optional, Callable

from jem.base.interface import resolve_interface
from jem.base.typing import (
    JEMComponent,
    JEMForcingMapper,
    Workflow,
    Pytree,
    History,
    TrajectoryFunction,
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

    if len(ys) > 0:
        ys = stack_objects(ys)

    return carry, ys


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
   
    components: Dict[str, JEMComponent]
    forcing_mappers: Dict[str, JEMForcingMapper]
    
    def __init__(
        self,
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
        initial_value = {}
        for component_name, component in self.components.items():
            _state_derived_forcing = component.initialize()
            if len(_state_derived_forcing) != 3:
                raise ValueError(f"Component `{component_name:s}`'s initialize function needs 3 return values (state, derived and forcing).")
            initial_value[component_name] = _state_derived_forcing
        return initial_value


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
            print("Flattened workflow: ", ", ".join(flattened_workflow))
        
        # Get step functions of each component
        component_step_functions = {
            component_name: component.generate_step_function(jitted=jitted)  # type: ignore
            for component_name, component in self.components.items()
        }

        def step_function(carry, step):

            states = carry["states"]
            forcings = carry["forcings"]
            deriveds = carry["deriveds"]
        
            unstacked_predictions = { component_name : [] for component_name in self.components.keys() }
        
            for name in flattened_workflow:
                if name in self.components: 
                    states[name], deriveds[name], _predictions = component_step_functions[name](states[name], forcings[name], step)
                    unstacked_predictions[name].append(_predictions)
                elif name in self.forcing_mappers:
                    forcing_mapper = self.forcing_mappers[name]
                    sub_states = { component_name : states[component_name] for component_name in forcing_mapper.involved_component_names }
                    sub_deriveds = { component_name : deriveds[component_name] for component_name in forcing_mapper.involved_component_names }
                    sub_forcings = { component_name : forcings[component_name] for component_name in forcing_mapper.involved_component_names }
                    sub_forcings = forcing_mapper.map_forcings(sub_states, sub_deriveds, sub_forcings)
                    for name in sub_forcings.keys():
                        forcings[name] = sub_forcings[name]
                else:
                    raise Exception(f"Unknown error: Cannot find `{name}` in components or forcing_mappers.")        

            predictions = {
                name : unwrap_leading_dims(stack_objects(unstacked_predictions[name]), first_n_dim=2)
                for name in unstacked_predictions.keys()
            }
            
            return dict(states=states, deriveds=deriveds, forcings=forcings), predictions

        if jitted:
            step_function = jax.jit(step_function)

        return step_function

    def generate_trajectory_function(
        self,
        workflow: Workflow,
        iterations: int,
        jitted: bool = True,
        show_progress: bool = True,
        tqdm_kwargs: Dict[str, Any] = dict(desc="Simulation"),
    ) -> TrajectoryFunction:

        scan_func = generate_scan_function(jitted=jitted)
        coupled_step_function = self.generate_step_function(
            workflow=workflow,
            show_progress=show_progress,
            jitted=jitted,
        )
        steps = jnp.arange(iterations)
        
        if not jitted:
            steps = list(steps) # type: ignore

        if show_progress:
            if jitted:
                coupled_step_function = scan_tqdm(n=iterations, **tqdm_kwargs)(
                    coupled_step_function
                )
            else:
                steps = tqdm(steps, **tqdm_kwargs)

        def trajectory_function(
            initial_coupled_state_derived_forcing: Dict[str, tuple[type, type, type]],
        ) -> tuple[Pytree, History]:

            initial_carry_value = dict(states={}, deriveds={}, forcings={})
            for component_name, (_state, _derived, _forcing) in initial_coupled_state_derived_forcing.items():
                initial_carry_value["states"][component_name] = _state
                initial_carry_value["deriveds"][component_name] = _derived
                initial_carry_value["forcings"][component_name] = _forcing
            
            final_coupled_state, predictions = scan_func(
                coupled_step_function,
                initial_carry_value,
                xs=steps,
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

        self.components[name] = JEMComponent(
            raw_component = component,
            name = name,
            **resolve_interface(component, reference_class=JEMComponent, skip=["name", "raw_component", "get_info", "predictions_to_xarray"], verbose=True)
        )

        self._validate_components()

    def remove_component(self, name: str) -> None:
        """Remove a component from the coupler.

        Args:
            name: Component name to remove
        """
        if name in self.components:
            del self.components[name]
        
        self._validate_components()

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
            **resolve_interface(forcing_mapper, reference_class=JEMForcingMapper, skip=["name", "raw_forcing_mapper", "get_info"], verbose=True)
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

        component_info = {}
        forcing_mapper_info = {}

        for component_name, component in self.components.items():
            if hasattr(component, "get_info"):
                component_info[component_name] = component.get_info()
            else:
                component_info[component_name] = { "message" : "get_info not provided." }

        for name, forcing_mapper in self.forcing_mappers.items():
            if hasattr(forcing_mapper, "get_info"):
                forcing_mapper_info[name] = forcing_mapper.get_info()
            else:
                forcing_mapper_info[name] = { "message" : "get_info not provided." }

        return {
            "component_info" : component_info,
            "forcing_mappers" : forcing_mapper_info
        }

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
        

