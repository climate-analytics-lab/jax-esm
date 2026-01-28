"""Base component interface for Earth system models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict

from jem.base.exceptions import ValidationError
from jem.mapping.grid import Grid

import xarray as xr
from typeguard import check_type

@dataclass
class CoupledComponentConfig:
    """Configuration for a component."""

    name: str
    timestep: float  # seconds


class CoupledComponent(ABC):
    """Abstract base class for Earth system components."""

    component_forcing_class: type
    component_state_class: type
    state_variable_registry: Any
    forcing_variable_registry: Any
    horizontal_grids: Dict[str, Grid]

    def __init__(self, config: CoupledComponentConfig):
        """Initialize component with configuration."""
        self.config = config

    @abstractmethod
    def initialize(self) -> tuple[type, type]:
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
                step_fn(state, forcing, t: float)
                    -> Tuple[type, Dict]

            Where:
                - state: Current component state
                - forcing: External forcing from other components
                - t: Current simulation time in seconds
                - Returns: (new_state, predictions) tuple
        """
        pass

    def _validate(self) -> None:
        """Validate component configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        print(f"Validating {self.config.name}")
        check_type(self.horizontal_grids, Dict[str, Grid])

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

    def get_info(self) -> Dict:
        return {'name': self.config.name}
