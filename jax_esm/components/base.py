"""Base component interface for Earth system models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import tree_math
from dataclasses import make_dataclass

from datetime import datetime

class AbstractFieldGroup:
    pass

class AbstractComponentForcing:
    pass

class AbstractComponentState:
    prog     : AbstractFieldGroup | Any
    phydata  : AbstractFieldGroup | Any

def create_field_group_class(
    cls_name: str,
    fields: Tuple,
    base_cls = AbstractFieldGroup,
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
    dataclass_fields = [ (varname, dtype) for varname, dtype, _ in fields ]
    
    cls = make_dataclass(
        cls_name = cls_name,
        fields = dataclass_fields,
        bases = (base_cls,),
    )

    
    @classmethod
    def zeros(_cls, **kwargs):
        
        init_args = dict()
        for varname, dtype, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = jnp.zeros(shape, dtype=dtype)

        return _cls(**init_args)

    @classmethod
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

    cls.zeros = zeros
    cls.ones = ones
    cls.copy = copy
    
    return tree_math.struct(cls)

def create_component_state_class(
    prog_cls       : type,
    phydata_cls    : type,
    cls_name       : str = "",
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
            ("prog",     prog_cls),
            ("phydata",  phydata_cls),
        ],
        bases=(AbstractComponentState,),
    )

    @classmethod
    def zeros(_cls):
        return _cls(
            prog = prog_cls.zeros(),
            phydata = phydata_cls.zeros(),
        )

    @classmethod
    def ones(_cls):
        return _cls(
            prog = prog_cls.ones(),
            phydata = phydata_cls.ones(),
        )


    def copy(self, prog_kwargs = {}, phydata_kwargs = {}):
        o = new_cls(
            prog = self.prog.copy(**prog_kwargs),
            phydata = self.phydata.copy(**phydata_kwargs),
        )

        return o

    new_cls.zeros = zeros
    new_cls.ones = ones
    new_cls.copy = copy

    new_cls = tree_math.struct(new_cls)

    return new_cls



def create_component_forcing_class(
    flux_cls       : type,
    scalar_cls     : type,
    cls_name       : str = "",
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
            ("flux",   flux_cls),
            ("scalar", scalar_cls),
        ],
        bases=(AbstractComponentForcing,),
    )

    @classmethod
    def zeros(_cls):
        return _cls(
            flux = flux_cls.zeros(),
            scalar = scalar_cls.zeros(),
        )

    @classmethod
    def ones(_cls):
        return _cls(
            flux = flux_cls.ones(),
            scalar = scalar_cls.ones(),
        )


    def copy(self, flux_kwargs = {}, scalar_kwargs = {}):
        o = new_cls(
            flux   = self.flux.copy(**flux_kwargs),
            scalar = self.scalar.copy(**scalar_kwargs),
        )

        return o

    new_cls.zeros = zeros
    new_cls.ones = ones
    new_cls.copy = copy

    new_cls = tree_math.struct(new_cls)

    return new_cls



@dataclass
class ComponentConfig:
    """Configuration for a component."""
    name: str
    start_dt: datetime
    timestep: float  # seconds
    substeps: int           # count
    save_interval: float    # seconds
    coords: Any               # Coordinate specification
    params: Dict[str, Any]  # Component-specific parameters



class Component(ABC):
    """Abstract base class for Earth system components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize component with configuration."""
        self.config = config
        self.name = config.name
        self.timestep = config.timestep
        
    @abstractmethod
    def initialize(self) -> AbstractComponentState:
        """Initialize component state.
        
        Args:
            rng_key: JAX random key for initialization
            
        Returns:
            Initial component state
        """
        pass
    
    @abstractmethod
    def gen_step_fn(
        self,
    ) -> callable:
        """Advance component state by one timestep.
        
        Args:
           
        Returns:
            A function that accepts (init_state, time) and returns (final_state, predictions)
        """
        pass
    
    def get_boundary_fields(self, state: AbstractComponentState) -> Dict[str, Array]:
        """Extract boundary fields needed by other components.
        
        Args:
            state: Current component state
            
        Returns:
            Dictionary of boundary fields
        """
        return state.boundary
    
    def get_required_fluxes(self) -> List[str]:
        """Return list of required flux names from other components."""
        return ["heat", "moisture", "momentum_u", "momentum_v"]
    
    def get_provided_fluxes(self) -> List[str]:
        """Return list of flux names provided to other components."""
        return ["heat", "moisture", "momentum_u", "momentum_v"]
