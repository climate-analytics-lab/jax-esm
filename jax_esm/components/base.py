"""Base component interface for Earth system models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Sequence, Callable, Dict

import jax.numpy as jnp
import xarray as xr
import tree_math
from dataclasses import make_dataclass

from jax_esm.domain import Domain
from jax_esm.variable_registry import VariableRegistry


class AbstractFieldGroup:
    pass


class ComponentForcing(ABC):
    @classmethod
    @abstractmethod
    def zeros(cls):
        """Create a forcing instance with all fields set to zeros."""
        pass

    @classmethod
    @abstractmethod
    def ones(cls):
        """Create a forcing instance with all fields set to ones."""
        pass

    @abstractmethod
    def copy(self, **kwargs):
        """Create a copy of this forcing instance, optionally replacing fields."""
        pass


class ComponentState:
    prog: AbstractFieldGroup | Any
    phydata: AbstractFieldGroup | Any



@dataclass
class CoupledComponentConfig:
    """Configuration for a component."""

    name: str
    timestep: float  # seconds


class CoupledComponent(ABC):
    """Abstract base class for Earth system components."""

    component_forcing_class: ComponentForcing
    component_state_class: ComponentState
    state_variable_registry: VariableRegistry
    forcing_variable_registry: VariableRegistry
    domain: Domain

    def __init__(self, config: CoupledComponentConfig):
        """Initialize component with configuration."""
        self.config = config

    @abstractmethod
    def initialize(self) -> tuple[ComponentState, ComponentForcing]:
        """Initialize component state and forcing.

        Returns:
            Initial component state and forcing
        """
        pass

    @abstractmethod
    def generate_step_function(
        self,
        jitted: bool = False,
    ) -> Callable:
        """Generate a step function that advances the component state by one timestep.

        Args:
            jitted: If True, the returned function will be JIT-compiled.

        Returns:
            A function with signature:
                step_fn(state: ComponentState, forcing: ComponentForcing, t: float)
                    -> Tuple[ComponentState, Dict]

            Where:
                - state: Current component state
                - forcing: External forcing from other components
                - t: Current simulation time in seconds
                - Returns: (new_state, predictions) tuple
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate component configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        pass

    @abstractmethod
    def predictions_to_xarray(self, predictions: Dict) -> xr.Dataset:
        """Convert predictions from step function to xarray Dataset.

        Args:
            predictions: Dictionary of predictions accumulated from step function calls.

        Returns:
            xarray Dataset containing the predictions with appropriate coordinates
            and metadata.
        """
        pass

