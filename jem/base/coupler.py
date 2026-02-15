"""Main coupler class for Earth system model coupling."""

import time
from typing import Any, Dict, Optional, Callable

from jem.base.interface import resolve_interface
from jem.base.typing import (
    JEMComponent,
    JEMMapper,
    Workflow,
    Pytree,
    History,
    TrajectoryFunction,
    CoupledCarry,
)

import jax
import jax.numpy as jnp
from jax.tree_util import tree_structure
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
        mappers: 
            A dict whose key is the name of :code:`JEMMapper`-compatible
            object that is responsible for sending information between 
            components.
    """
   
    components: Dict[str, JEMComponent]
    mappers: Dict[str, JEMMapper]
    
    def __init__(
        self,
        components: Optional[Dict[str, Any]] = None,
        mappers: Optional[Dict[str, Any]] = None,
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

        self.mappers = mappers or {}
        for name, mapper in self.mappers.items():
            self.add_mapper(name, mapper)

    def initialize(
        self,
    ) -> CoupledCarry:
        """Initialize all components.

        Args:
            rng_key: JAX random key for initialization

        Returns:
            Dictionary of initial states for all components
        """
        return {
            component_name : component.initialize()
            for component_name, component in self.components.items()
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
            print("Flattened workflow: ", ", ".join(flattened_workflow))
        
        # Get step functions of each component
        component_step_functions = {}
        for component_name, component in self.components.items():
            _component_step_function = component.generate_step_function()
            if jitted:
                _component_step_function = jax.jit(_component_step_function)
            component_step_functions[component_name] = _component_step_function

        def step_function(carry: CoupledCarry, step):
            unstacked_predictions = { component_name : [] for component_name in self.components.keys() }
            input_carry_structure = tree_structure(carry)
            for name in flattened_workflow:
                if name in self.components:
                    carry[name], _predictions = component_step_functions[name](carry[name], step)
                    unstacked_predictions[name].append(_predictions)
                elif name in self.mappers:
                    carry = self.mappers[name](carry)
                else:
                    raise Exception(f"Unknown error: Cannot find `{name}` in components or mappers.")        
                if input_carry_structure != tree_structure(carry):
                    print(f"Warning: carry value structure changed after workflow element `{name:s}` is used.")

            predictions = {
                name : unwrap_leading_dims(stack_objects(unstacked_predictions[name]), first_n_dim=2)
                for name in unstacked_predictions.keys()
            }

            return carry, predictions

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
            initial_coupled_carry: CoupledCarry,
        ) -> tuple[Pytree, History]:

            final_coupled_state, predictions = scan_func(
                coupled_step_function,
                initial_coupled_carry,
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
            **resolve_interface(component, reference_class=JEMComponent, skip=["name", "raw_component"], verbose=True)
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

    def add_mapper(
        self,
        name: str,
        mapper: Any,
    ) -> None:
        """Add a new forcing mapper to the coupler.
        Args:
            name: Forcing mapper name
            mapper: Forcing mapper
        """
        self.mappers[name] = JEMMapper(
            raw_mapper = mapper,
            name = name,
            **resolve_interface(mapper, reference_class=JEMMapper, skip=["name", "raw_mapper"], verbose=True)
        )
        

    def remove_mapper(self, name: str) -> None:
        """Remove a forcing mapper from the coupler.

        Args:
            name: Forcing mapper name to remove
        """
        if name in self.mappers:
            del self.mappers[name]


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
            "mappers" : "None" if self.mappers is None else { name: mapper.get_info() for name, mapper in self.mappers.items() } ,
        }

    def _validate_components(self):
        pass

    def _verify_name_uniqueness(self) -> None:
        
        all_names = list(self.components.keys()) + list(self.mappers.keys())
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
            if action not in self.components and action not in self.mappers:
                raise ValueError(f"Action `{action:s}` does not map to any component or forcing mapper.")
        

