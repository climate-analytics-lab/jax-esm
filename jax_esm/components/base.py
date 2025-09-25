"""Base component interface for Earth system models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Tuple, Type, Union

import jax
import jax.numpy as jnp
from jax import Array

import pandas as pd

@dataclass
class ComponentConfig:
    """Configuration for a component."""
    name: str
    start_dt: pd.Timestamp
    timestep: float         # seconds
    substeps: int           # count
    save_interval: float    # seconds
    grid: Dict[str, Any]    # Grid specification
    params: Dict[str, Any]  # Component-specific parameters


class Component(ABC):
    """Abstract base class for Earth system components."""

    stateDiagClass : Type
    state_diag : Any
    trajectory : List
    
    def __init__(self, config: ComponentConfig):
        """Initialize component with configuration."""
        self.config = config
        self.name = config.name
        self.timestep = config.timestep

    @abstractmethod
    def initialize(
        self,
    ):
        pass
        
    @abstractmethod
    def genForwardFunc(
        self,
        begin_time,
    ):
        pass

    @abstractmethod
    def record(
        self,
        state_diag,
    ):
        pass