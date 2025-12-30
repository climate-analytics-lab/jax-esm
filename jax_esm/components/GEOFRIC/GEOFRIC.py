"""GEOFRIC Wrapper Class"""

from dataclasses import dataclass
import jax_datetime as jdt
from GEOFRIC.model import Model as RawModel
from jax_esm.base.grid import Grid, GridSpecification
import coordax as cx
import jax
import jax.numpy as jnp
from jax import Array
from jax_esm import constants as constants
from jax_esm.components.base import (
    CoupledComponent,
    CoupledComponentConfig,
)

from jax_esm.utils.bulk_op import mean_leaf
from jax_esm.base.domain import Domain
from jax_esm.base.variable import VariableMetadata, VariableRegistry

import tree_math
from typing import Any, Dict

class GEOFRIC(CoupledComponent):
    """
    This is a class wrapping GEOFRIC.
    """

    def __init__(
        self,
        model: RawModel,
        coupling_timestep: float = 86400.0,
        save_interval: float = 86400.0,
    ):
        """
        Configuration of GEOFRIC.
        """

        self.model = model
        super().__init__(
            CoupledComponentConfig(
                name="GEOFRIC",
                timestep=coupling_timestep,
            )
        )
        
        shapes = self.model.LLC_grid.shapes
        self.domain = Domain(
            horizontal_grids = dict(
                T = Grid(
                    coordinate = cx.coords.compose(
                        cx.LabeledAxis('tile', jnp.arange(shapes["T"][0])),
                        cx.LabeledAxis('j', jnp.arange(shapes["T"][1])),
                        cx.LabeledAxis('i', jnp.arange(shapes["T"][2])),
                    ),
                    grid_specification = GridSpecification.parse_grid_specification(self.model.grid_specification),
                ),
            ),
        )

        self.save_interval = save_interval

        if save_interval > coupling_timestep:
            raise ValueError("Error: `save_interval` is larger than model timestep. ")

        self.component_state_class = model.component_state_class
        self.component_forcing_class = model.component_forcing_class

        self.state_variable_registry = VariableRegistry([
            VariableMetadata(name="prog.sea_surface_temperature", shape=shapes["T"], dimensions=("tile", "j", "i")),
        ])

        self.forcing_variable_registry = VariableRegistry([
            VariableMetadata(name="flux.total_heat_flux", shape=shapes["T"], dimensions=("tile", "j", "i")),
        ])
    
    def initialize(
        self,
        start_date: jdt.Datetime = jdt.to_datetime("2000-01-01"),
    ):
        return self.model.initialize()

    def generate_step_function(self, jitted: bool = True):
        return self.model.generate_step_function(jitted=jitted)


    def validate(self):
        pass

    def predictions_to_xarray(
        self,
        predictions,
    ):
        return self.model.predictions_to_xarray(predictions)
    
    def report(self):
        pass
