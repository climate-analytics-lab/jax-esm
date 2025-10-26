"""Tests for component interface.

DEPRECATED: These tests use the old API and are kept for reference only.
All tests are skipped. See test_coupler.py, test_slab_ocean.py, and test_flux_model.py
for current API tests.
"""

import jax
import jax.numpy as jnp
import unittest
import pandas as pd

from jax_esm.components.base import (
    Component,
    ComponentConfig,
    create_component_state_class,
    create_field_group_class,
)

from jax_esm.components.domain import Domain
from jax_esm.utils.bulk_op import stack_objects

class MockComponent(Component):

    """
    Mock ocean model
    """
        
    def __init__(
        self,
        config: ComponentConfig,
    ):
        """Initialize slab ocean model."""
        
        super().__init__(config)

        self.subtimestep = config.timestep / config.substeps
       
        D3_nodal_shape = config.domain.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]

        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("T", float, D2_nodal_shape),
                    ("mld", float, D2_nodal_shape),
                ],
            ),

            phydata_cls =  create_field_group_class(
                cls_name = "phydata",
                fields = [
                    ("heatflx", float, D2_nodal_shape),
                ],
            ),
        )

    def initialize(self):

        init_state = self.component_state_class.zeros()


        return init_state
 
    def generate_step_function(
        self,
    ):

        @jax.jit
        def step_function(state, t):


            T = state.prog.T
            sim_time = state.prog.sim_time
           
            hist = [] 
            for step in range(self.config.substeps):
            
                T = T + self.subtimestep * state.phydata.heatflx / self.config.params["heat_capacity"]
                sim_time += self.subtimestep

            state = state.copy(
                prog_kwargs = dict(
                    sim_time = sim_time,
                    T = T,
                ),
            )

            hist.append(dict(state=state))
            
            return state, stack_objects( hist )
            
        return step_function

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
        st = predictions["state"]
        ds = xr.Dataset(
            data_vars = dict(
                T   = (["time", "lon", "lat"], st.T),
                mld = (["time", "lon", "lat"], st.mld),
            ), 
            coords = dict(
                time = (["time",], st.sim_time),
            ),
        )
        
        return ds
        



class TestComponent(unittest.TestCase):
    
    """Test component interface."""
    
    def generate_test_component(self, horizontal_resolution:int = 31):

        domain = Domain.from_resolution_all_ocean(horizontal_resolution)

        config = ComponentConfig(
            name = "test",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            domain = domain,
            params = {"heat_capacity": 1000.0},
        )
        
        return MockComponent(config) 

    def test_component_initialization(self):
        """Test component initialization."""
 
        component = self.generate_test_component()
        
        assert component.name == "test"
        assert component.timestep == 1800.0
        assert component.config.params["heat_capacity"] == 1000.0
    
    def test_component_initialize_state(self):

        """Test state initialization."""
        
        component = self.generate_test_component()

        state = component.initialize()

        state = state.copy(
            prog_kwargs = dict(T = state.prog.T + 300.0),
            phydata_kwargs = dict(heatflx = state.phydata.heatflx + 500.0),
        )


        assert hasattr(state.prog, "T")
        assert state.prog.T.shape == (96, 48)
        assert jnp.allclose(state.prog.T, 300.0)
        assert jnp.allclose(state.phydata.heatflx, 500.0)
        


    def test_component_step(self):
        """Test component stepping."""

        component = self.generate_test_component()

        state = component.initialize()

        state = state.copy(
            prog_kwargs = dict(T = state.prog.T.at[:].set(300.0)),
            phydata_kwargs = dict(heatflx = state.phydata.heatflx.at[:].set(500.0)),
        )
        
        step_function = component.generate_step_function()

        new_state, hist = step_function(state, 0)

        # Check state update
        assert new_state.prog.sim_time == 1800.0

        expected_temp = 300 + 500 * 1800.0 / 1000.0
        assert jnp.allclose(new_state.prog.T, expected_temp)
        

