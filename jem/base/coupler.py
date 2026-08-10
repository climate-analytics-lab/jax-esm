"""Main coupler class for Earth system model coupling."""

import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import tree_structure
from jax_tqdm import scan_tqdm
from tqdm import tqdm

from jem.base.interface import resolve_interface
from jem.base.typing import (
    CoupledCarry,
    JEMComponent,
    MapperFunction,
    Predictions,
    Pytree,
    TrajectoryFunction,
    Workflow,
)
from jem.utils.bulk_op import stack_objects, unwrap_leading_dims


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
            A dict whose key is the name of :code:`MapperFunction`-compatible
            object that is responsible for sending information between 
            components.
    """
   
    components: dict[str, JEMComponent]
    mappers: dict[str, MapperFunction]
    tracjectory_holder: TrajectoryFunction | None
     
    def __init__(
        self,
        components: dict[str, Any] | None = None,
        mappers: dict[str, Any] | None = None,
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

        self.trajectory_holder = None

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
            _unstacked_predictions = { component_name : [] for component_name in self.components } # type: ignore
            input_carry_structure = tree_structure(carry)
            for name in flattened_workflow:
                if name in self.components:
                    carry[name], _predictions = component_step_functions[name](carry[name], step)
                    _unstacked_predictions[name].append(_predictions)
                elif name in self.mappers:
                    carry = self.mappers[name](carry)
                else:
                    raise RuntimeError(f"Unknown error: Cannot find `{name}` in components or mappers.")
                if input_carry_structure != tree_structure(carry): # type: ignore
                    print(f"Warning: carry value structure changed after workflow element `{name:s}` is used.")

            predictions = {
                name : unwrap_leading_dims(stack_objects(_unstacked_predictions[name]), first_n_dim=2)
                for name in _unstacked_predictions if len(_unstacked_predictions[name]) != 0
            }

            return carry, predictions

        return jax.jit(step_function) if jitted else step_function

    def run(
        self,
        workflow: Workflow,
        iterations: int,
        initial_carry: CoupledCarry | None = None,
        jitted: bool = True,
        show_progress: bool = True,
        tqdm_kwargs: dict[str, Any] | None = None,
        reuse_last_available_trajectory: bool = False,
        verbose: bool=True,
        checkpoint: bool = False,
    ) -> tuple[CoupledCarry, CoupledCarry, TrajectoryFunction]:
        if tqdm_kwargs is None:
            tqdm_kwargs = {"desc": "Simulation"}
        initial_carry = initial_carry or self.initialize()

        if reuse_last_available_trajectory and self.trajectory_holder is not None:
            verbose and print("Reuse last available trajectory.")
            trajectory = self.trajectory_holder
        else:
            _start_time = time.time()
            trajectory = self.generate_trajectory_function(
                workflow=workflow,
                iterations=iterations,
                jitted=jitted,
                show_progress=show_progress,
                tqdm_kwargs=tqdm_kwargs,
                checkpoint=checkpoint,
            )
            _elapsed_time = time.time() - _start_time
            print(f"generate_trajectory_function took {_elapsed_time:.2f} seconds.")

        self.trajectory_holder = trajectory  # type: ignore
        
        return initial_carry, *trajectory(initial_carry)

    def generate_trajectory_function(
        self,
        workflow: Workflow,
        iterations: int,
        jitted: bool = True,
        show_progress: bool = True,
        checkpoint: bool = False,
        tqdm_kwargs: dict[str, Any] | None = None,
    ) -> TrajectoryFunction:

        if tqdm_kwargs is None:
            tqdm_kwargs = {"desc": "Simulation"}
        scan_func = generate_scan_function(jitted=jitted)
        coupled_step_function = self.generate_step_function(
            workflow=workflow,
            show_progress=show_progress,
            jitted=jitted,
        )

        if checkpoint:
            coupled_step_function = jax.checkpoint(coupled_step_function)
        
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
        ) -> tuple[Pytree, Predictions]:

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
            **resolve_interface(
                component,
                reference_class=JEMComponent,
                skip=["name", "raw_component"],
                optional=["predictions_to_xarray", "get_info"],
                verbose=True
            )
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
        #try:
        #    typeguard.check_type(mapper, MapperFunction)
        #except typeguard.TypeCheckError as e:
        #    raise typeguard.TypeCheckError(f"The mapper {name} is not a valid MapperFunction.")
        
        self.mappers[name] = mapper

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
            for component_name, component in self.components.items() if component_name in predictions
            if component.predictions_to_xarray is not None
        }

    def get_info(self):
        component_info = {}
        mapper_info = {}

        for component_name, component in self.components.items():
            if getattr(component, "get_info", None) is not None:
                component_info[component_name] = component.get_info()
            else:
                component_info[component_name] = { "message" : "get_info not provided." }

        for name, mapper in self.mappers.items():
            if getattr(mapper, "get_info", None) is not None:
                mapper_info[name] = mapper.get_info()
            else:
                mapper_info[name] = { "message" : "get_info not provided." }
        return {
            "component_info" : component_info,
            "mappers" : mapper_info,
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
                raise ValueError(f"The name `{name}` is not unique. There are {count:d} of the same name.")

    def _verify_workflow(
        self,
        workflow: Workflow,
    ) -> None:

        flattened_workflow, _ = jax.tree.flatten(workflow)
        for action in flattened_workflow:
            if not isinstance(action, str):
                raise TypeError("Actions in the workflow have to be strings.")
            if action not in self.components and action not in self.mappers:
                raise ValueError(f"Action `{action:s}` does not map to any component or forcing mapper.")
        

