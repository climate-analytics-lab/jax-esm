"""Base component interface for Earth system models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable

import jax.numpy as jnp
from jax import Array
import tree_math
from dataclasses import make_dataclass


class AbstractFieldGroup:
    pass



class ComponentForcing(ABC):
    
    @abstractmethod
    def zeros(cls):
        pass

    @abstractmethod
    def ones(cls):
        pass

    @abstractmethod
    def copy(cls):
        pass




class ComponentState:
    prog: AbstractFieldGroup | Any
    phydata: AbstractFieldGroup | Any


def create_field_group_class(
    cls_name: str,
    fields: Tuple,
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

    @classmethod # type: ignore
    def zeros(_cls, **kwargs):
        init_args = dict()
        for varname, dtype, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = jnp.zeros(shape, dtype=dtype)

        return _cls(**init_args)

    @classmethod # type: ignore
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
    cls.ones = ones    # type: ignore
    cls.copy = copy    # type: ignore

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

    @classmethod # type: ignore
    def zeros(_cls):
        return _cls(
            prog=prog_cls.zeros(),
            phydata=phydata_cls.zeros(),
        )

    @classmethod # type: ignore
    def ones(_cls):
        return _cls(
            prog=prog_cls.ones(),
            phydata=phydata_cls.ones(),
        )

    def copy(self, prog_kwargs={}, phydata_kwargs={}):
        o = new_cls(
            prog=self.prog.copy(**prog_kwargs),
            phydata=self.phydata.copy(**phydata_kwargs),
        )

        return o

    new_cls.zeros = zeros  # type: ignore
    new_cls.ones = ones    # type: ignore
    new_cls.copy = copy    # type: ignore

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

    @classmethod # type: ignore
    def zeros(_cls):
        return _cls(
            flux=flux_cls.zeros(),
            scalar=scalar_cls.zeros(),
        )

    @classmethod # type: ignore
    def ones(_cls):
        return _cls(
            flux=flux_cls.ones(),
            scalar=scalar_cls.ones(),
        )

    def copy(self, flux_kwargs={}, scalar_kwargs={}):
        o = new_cls(
            flux=self.flux.copy(**flux_kwargs),
            scalar=self.scalar.copy(**scalar_kwargs),
        )

        return o

    new_cls.zeros = zeros # type: ignore
    new_cls.ones = ones   # type: ignore
    new_cls.copy = copy   # type: ignore

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

    def __init__(self, config: CoupledComponentConfig):
        """Initialize component with configuration."""
        self.config = config

    @abstractmethod
    def initialize(self) -> ComponentState:
        """Initialize component state.

        Args:
            rng_key: JAX random key for initialization

        Returns:
            Initial component state
        """
        pass

    @abstractmethod
    def generate_step_function(
        self,
        jitted: bool = False,
    ) -> Callable:
        """Advance component state by one timestep.

        Args:

        Returns:
            A function that accepts (init_state, time) and returns (final_state, predictions)
        """
        pass
