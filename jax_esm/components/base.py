"""Base component interface for Earth system models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Sequence, Callable, Dict

import jax.numpy as jnp
import xarray as xr
import tree_math
from dataclasses import make_dataclass

from jax_esm.components.domain import Domain


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


def create_field_group_class(
    cls_name: str,
    fields: Sequence[Tuple[str, type, Sequence[int]]],
    base_cls=AbstractFieldGroup,
):
    """
    A tool function that creates a state class dynamically with given dimension.
    The created class will have the following methods: `zeros`, `ones`, and `copy`.

    Args:

        cls_name : Name of the state class
        fields   : A list of (varname, data type, shape) tuples.

    Returns:

        class : The resulting state class.

    """
    dataclass_fields = [(varname, dtype) for varname, dtype, _ in fields]

    cls = make_dataclass(
        cls_name=cls_name,
        fields=dataclass_fields,
        bases=(base_cls,),
    )

    @classmethod  # type: ignore
    def zeros(_cls, **kwargs):
        init_args = dict()
        for varname, dtype, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = jnp.zeros(shape, dtype=dtype)

        return _cls(**init_args)

    @classmethod  # type: ignore
    def ones(_cls, **kwargs):
        init_args = dict()
        for varname, dtype, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = jnp.ones(shape, dtype=dtype)

        return _cls(**init_args)

    def copy(self, **kwargs):
        init_args = dict()
        for varname, dtype, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = getattr(self, varname)

        return type(self)(**init_args)

    cls.zeros = zeros  # type: ignore
    cls.ones = ones  # type: ignore
    cls.copy = copy  # type: ignore

    return tree_math.struct(cls)


def create_component_state_class(
    prog_cls: type,
    phydata_cls: type,
    cls_name: str = "",
):
    """
    A tool function that creates a ComponentState class dynamically with given dimension.
    The created class will have the following methods: `zeros`, `ones`, and `copy`.

    Args:

        prog      : The prog class.
        phydata   : The phydata class.
        cls_name  : Name of the class.

    Returns:

        The resulting class.

    """

    new_cls = make_dataclass(
        f"{cls_name:s}",
        [
            ("prog", prog_cls),
            ("phydata", phydata_cls),
        ],
        bases=(ComponentState,),
    )

    @classmethod  # type: ignore
    def zeros(_cls):
        return _cls(
            prog=prog_cls.zeros(),
            phydata=phydata_cls.zeros(),
        )

    @classmethod  # type: ignore
    def ones(_cls):
        return _cls(
            prog=prog_cls.ones(),
            phydata=phydata_cls.ones(),
        )

    def copy(self, prog_kwargs=None, phydata_kwargs=None):
        if prog_kwargs is None:
            prog_kwargs = {}
        if phydata_kwargs is None:
            phydata_kwargs = {}
        o = new_cls(
            prog=self.prog.copy(**prog_kwargs),
            phydata=self.phydata.copy(**phydata_kwargs),
        )

        return o

    new_cls.zeros = zeros  # type: ignore
    new_cls.ones = ones  # type: ignore
    new_cls.copy = copy  # type: ignore

    new_cls = tree_math.struct(new_cls)

    return new_cls


def create_component_forcing_class(
    flux_cls: type,
    scalar_cls: type,
    cls_name: str = "",
):
    """
    A tool function that creates a ComponentState class dynamically with given dimension.
    The created class will have the following methods: `zeros`, `ones`, and `copy`.

    Args:

        flux      : The flux class.
        scalar    : The scalar class.
        cls_name  : Name of the class.

    Returns:

        The resulting class.

    """

    new_cls = make_dataclass(
        f"{cls_name:s}",
        [
            ("flux", flux_cls),
            ("scalar", scalar_cls),
        ],
        bases=(ComponentForcing,),
    )

    @classmethod  # type: ignore
    def zeros(_cls):
        return _cls(
            flux=flux_cls.zeros(),
            scalar=scalar_cls.zeros(),
        )

    @classmethod  # type: ignore
    def ones(_cls):
        return _cls(
            flux=flux_cls.ones(),
            scalar=scalar_cls.ones(),
        )

    def copy(self, flux_kwargs=None, scalar_kwargs=None):
        if flux_kwargs is None:
            flux_kwargs = {}
        if scalar_kwargs is None:
            scalar_kwargs = {}
        o = new_cls(
            flux=self.flux.copy(**flux_kwargs),
            scalar=self.scalar.copy(**scalar_kwargs),
        )

        return o

    new_cls.zeros = zeros  # type: ignore
    new_cls.ones = ones  # type: ignore
    new_cls.copy = copy  # type: ignore

    new_cls = tree_math.struct(new_cls)

    return new_cls


@dataclass
class CoupledComponentConfig:
    """Configuration for a component."""

    name: str
    timestep: float  # seconds


class Component(ABC):
    """Abstract base class for Earth system components."""

    component_forcing_class: ComponentForcing
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
