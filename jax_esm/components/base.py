"""Base component interface for Earth system models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import tree_math

import pandas as pd

@dataclass
class Axis:
    
    """Axis
    
    Attributes:
        name      : name of the coordinate
        long_name : long name of the coordinate
        unit      : unit of the coordinate
    """
    values    : Array
    name      : str = ""
    long_name : str = ""
    unit      : str = ""


    def __len__(self):
        return len(Array)


@dataclass
class CoordinateSystem(dict):
    pass
 
@dataclass
class ComponentStateShape:
    
    """Description of shape of `ComponentState` for Earth system components.
    
    Attributes:
        prognostic : Prognostic variables that evolve with time
        diagnostic : Diagnostic variables computed from prognostic state
        boundary   : Boundary conditions and surface properties
        forcing    : External forcing fields

    """
    coord_sys  : CoordinateSystem
    prognostic : Dict[str, List[str]] = field(default_factory=lambda: {})
    diagnostic : Dict[str, List[str]] = field(default_factory=lambda: {})
    boundary   : Dict[str, List[str]] = field(default_factory=lambda: {})
    forcing    : Dict[str, List[str]] = field(default_factory=lambda: {})

    def get_shape(var_group, varname):
        axis_names = getattr(self, var_group)[varname]
        return [ len(self.coord_sys[axis_name]) for axis_name in axis_names ]
        

@tree_math.struct
@dataclass
class ComponentState:
    
    """State container for Earth system components.
    
    Attributes:
        prognostic : Prognostic variables that evolve with time
        diagnostic : Diagnostic variables computed from prognostic state
        boundary   : Boundary conditions and surface properties
        forcing    : External forcing fields
        metadata   : Additional metadata

    """
    prognostic: Dict[str, Array] = field(default_factory=lambda: {})
    diagnostic: Dict[str, Array] = field(default_factory=lambda: {})
    boundary:   Dict[str, Array] = field(default_factory=lambda: {})
    forcing:    Dict[str, Array] = field(default_factory=lambda: {})
    metadata:   Dict[str, Array] = field(default_factory=lambda: {})

    @classmethod
    def zeros(
        cls,
        comp_state_shp: ComponentStateShape,
    ):

        comp_state = ComponentState()
        
        for var_group, field_info in comp_state.__dataclass_fields__.items():
            
            d = getattr(comp_state, var_group)
            varnames = getattr(comp_state_shp, var_group)

            for varname in varnames:
                
                d[varname] = jnp.zeros(
                    comp_state_shp.get_shape(var_group, varname)
                )

        return comp_state 

        


class BoundaryFluxes(NamedTuple):
    """Container for boundary fluxes between components.
    
    Attributes:
        heat: Heat flux (W/m²)
        moisture: Moisture flux (kg/m²/s)
        momentum_u: Zonal momentum flux (N/m²)
        momentum_v: Meridional momentum flux (N/m²)
        tracers: Dictionary of tracer fluxes
    """
    heat: Array
    moisture: Array
    momentum_u: Array
    momentum_v: Array
    tracers: Dict[str, Array]


@dataclass
class ComponentConfig:
    """Configuration for a component."""
    name: str
    start_dt: pd.Timestamp
    timestep: float  # seconds
    substeps: int           # count
    save_interval: float    # seconds
    grid: Dict[str, Any]  # Grid specification
    params: Dict[str, Any]  # Component-specific parameters



class Component(ABC):
    """Abstract base class for Earth system components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize component with configuration."""
        self.config = config
        self.name = config.name
        self.timestep = config.timestep
        
    @abstractmethod
    def initialize(self, rng_key: jax.random.PRNGKey) -> ComponentState:
        """Initialize component state.
        
        Args:
            rng_key: JAX random key for initialization
            
        Returns:
            Initial component state
        """
        pass
    
    @abstractmethod
    def step(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
        dt: float,
    ) -> Tuple[ComponentState, BoundaryFluxes]:
        """Advance component state by one timestep.
        
        Args:
            state: Current component state
            forcing: Boundary fluxes from other components
            dt: Time step size (seconds)
            
        Returns:
            Tuple of (new_state, output_fluxes)
        """
        pass
    
    @abstractmethod
    def compute_tendencies(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
    ) -> Dict[str, Array]:
        """Compute tendencies for prognostic variables.
        
        Args:
            state: Current component state
            forcing: Boundary fluxes from other components
            
        Returns:
            Dictionary of tendencies for each prognostic variable
        """
        pass
    
    def get_boundary_fields(self, state: ComponentState) -> Dict[str, Array]:
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


class CoupledComponent(Protocol):
    """Protocol for components that can be coupled."""
    
    name: str
    timestep: float
    
    def initialize(self, rng_key: jax.random.PRNGKey) -> ComponentState:
        ...
    
    def step(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
        dt: float,
    ) -> Tuple[ComponentState, BoundaryFluxes]:
        ...
    
    def get_boundary_fields(self, state: ComponentState) -> Dict[str, Array]:
        ...
    
    def get_required_fluxes(self) -> List[str]:
        ...
    
    def get_provided_fluxes(self) -> List[str]:
        ...
