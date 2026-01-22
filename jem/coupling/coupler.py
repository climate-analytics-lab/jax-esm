"""Main coupler class for Earth system model coupling."""

from functools import partial
import time
from typing import Any, Dict, Optional, Callable
from jem.base.typing import (
    InitializeFunction,
    StepFunctionGenerator,
    HistoryToXarray,
)
import jem.base.mixin as mixin

import jax
import jax.numpy as jnp

from jem.components.base import CoupledComponent
from jem.coupling.forcing_mapper import ForcingMapper
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
        coupling_timestep: The time interval (seconds) between each time components exchanging 
            information (heat fluxes, surface temperature, and so on).
        components:
            A dict whose key is the name of the component and the value is the instantiated component.
        forcing_mapper: 
            A :code:`ForcingMapper` that is responsible for sending information between components.
    
    """
    
    coupling_timestep: float
    components: Dict[str, CoupledComponent]
    forcing_mapper: ForcingMapper
    
    def __init__(
        self,
        coupling_timestep: float,
        components: Optional[Dict[str, CoupledComponent]] = None,
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

        self.forcing_mapper = forcing_mapper or ForcingMapper(self.components)

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
        jitted: bool = True,
        show_progress: bool = True,
    ) -> Callable:
        """Advance coupled system by one coupling timestep.

        Args:

        Returns:
            New states after one coupling timestep
        """

        scan_func = generate_scan_function(jitted=jitted)

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

        def step_function(bundle, step_time):
            _, t = step_time

            coupled_state = bundle["state"]
            coupled_forcings = bundle["forcings"]

            # Forcing should have the flexibility that sometimes we want forcing to be
            # mapped (coupled), and sometimes being left unchanged (true forcing)
            coupled_forcings = self.forcing_mapper.couple_components(
                coupled_forcings, coupled_state
            )

            # Call forward functions and unpack results directly into dictionaries
            results = {
                component_name: step_function(
                    coupled_state[component_name],
                    coupled_forcings[component_name],
                    t,
                )
                for component_name, step_function in step_functions.items()
            }
            coupled_state = {name: state for name, (state, _) in results.items()}
            coupled_predictions = {
                name: prediction for name, (_, prediction) in results.items()
            }

            return dict(
                state=coupled_state, forcings=coupled_forcings
            ), coupled_predictions

        if jitted:
            step_function = jax.jit(step_function)

        return step_function

    def generate_trajectory_function(
        self,
        start_time: float,
        end_time: float,
        save_interval_steps=1,
        jitted: bool = True,
        show_progress: bool = False,
        tqdm_kwargs: Dict[str, Any] = dict(),
    ):
        total_time = end_time - start_time
        total_steps = int(total_time / self.coupling_timestep)

        if total_steps * self.coupling_timestep != total_time:
            raise Exception("timestep has to exactly divide (end_time - start_time).")

        scan_func = generate_scan_function(jitted=jitted)

        coupled_step_function = self.generate_step_function(
            show_progress=show_progress,
            jitted=jitted,
        )

        steps = jnp.arange(total_steps)
        times = start_time + self.coupling_timestep * steps
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
                    state={
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
        name: Optional[str] = None,
        component: Optional[CoupledComponent] = None,
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
            self._validate_components_mixin(component)

        self.component_names = list(self.components.keys())
        self.component_timesteps = {
            name: component.config.timestep
            for name, component in self.components.items()
        }

        self.component_state_variable_registries = {
            name: component.state_variable_registry
            for name, component in self.components.items()
        }

        self.component_forcing_variable_registries = {
            name: component.forcing_variable_registry
            for name, component in self.components.items()
        }

        self._validate_components()


    def _validate_components_mixin(self, component) -> None:

        mixin.verify_members(
            component,
            {
                "timestep" : float,
            },
            verbose=True,
        )

        mixin.verify_functions(
            component,
            {
                "initialize" : InitializeFunction,
                "generate_step_function" : StepFunctionGenerator,
                "predictions_to_xarray" : HistoryToXarray,
            },
            verbose = True,
        )
     
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

    def get_info(self):
        return {
            "coupling_timestep" : self.coupling_timestep,
            "component_info" : {
                component_name : component.get_info() for component_name, component in self.components.items()
            },
            "forcing_mapper" : "None" if self.forcing_mapper is None else self.forcing_mapper.get_info(),
        }
        return info
