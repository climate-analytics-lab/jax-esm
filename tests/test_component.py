"""Tests for component interface."""

import jax
import jax.numpy as jnp
import unittest
import pandas as pd

from jax_esm.components.base import (
    Component, BoundaryFluxes,
    ComponentConfig,
    create_component_state_class,
    create_field_group_class,
)

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
        
        D3_nodal_shape = (config.grid["z"], config.grid["lon"], config.grid["lat"])
        D2_nodal_shape = D3_nodal_shape[1:]

        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("T", float, D3_nodal_shape),
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
 
    def gen_step_fn(
        self,
    ):

        @jax.jit
        def step_fn(state, t):


            T = state.prog.T
            sim_time = state.prog.sim_time
           
            hist = [] 
            for step in range(self.config.substeps):
            
                T = T.at[0, :, :].set(T[0, :, :] + self.subtimestep * state.phydata.heatflx / self.config.params["heat_capacity"])
                sim_time += self.subtimestep

            state = state.copy(
                prog_kwargs = dict(
                    sim_time = sim_time,
                    T = T,
                ),
            )

            hist.append(dict(state=state))
            
            return state, stack_objects( hist )
            
        return step_fn

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
    
    def gen_test_component(self):

        grid = {"lat": 5, "lon": 10, "z": 8}
        config = ComponentConfig(
            name = "test",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            grid = grid,
            params = {"heat_capacity": 1000.0},
        )
        
        return MockComponent(config) 

    def test_component_initialization(self):
        """Test component initialization."""
 
        component = self.gen_test_component()
        
        assert component.name == "test"
        assert component.timestep == 1800.0
        assert component.config.params["heat_capacity"] == 1000.0
    
    def test_component_initialize_state(self):

        """Test state initialization."""
        
        component = self.gen_test_component()

        state = component.initialize()

        state = state.copy(
            prog_kwargs = dict(T = state.prog.T + 300.0),
            phydata_kwargs = dict(heatflx = state.phydata.heatflx + 500.0),
        )


        assert hasattr(state.prog, "T")
        assert state.prog.T.shape == (8, 10, 5)
        assert jnp.allclose(state.prog.T, 300.0)
        assert jnp.allclose(state.phydata.heatflx, 500.0)
        


    def test_component_step(self):
        """Test component stepping."""

        component = self.gen_test_component()

        state = component.initialize()

        state = state.copy(
            prog_kwargs = dict(T = state.prog.T.at[:].set(300.0)),
            phydata_kwargs = dict(heatflx = state.phydata.heatflx.at[:].set(500.0)),
        )
        
        step_fn = component.gen_step_fn()

        new_state, hist = step_fn(state, 0)

        # Check state update
        assert new_state.prog.sim_time == 1800.0

        expected_temp = 300 + 500 * 1800.0 / 1000.0
        assert jnp.allclose(new_state.prog.T[0, :, :], expected_temp)
        

        assert jnp.allclose(new_state.prog.T[1:, :, :], 300)
        
    """    
    def test_boundary_fields(self):


        def mk_center_grid(bnd_l, bnd_r, n):
            tmp = jnp.linspace(bnd_l, bnd_r, n+1)
            return (tmp[1:] + tmp[:-1])/2 
 
        grid = {"nlon": 10, "nlat": 5, "z": 8}
        config = ComponentConfig(
            name = "test",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            grid = grid,
            params = {"test_param": 42},
            comp_state_shp = ComponentStateShape(
                coord_sys = CoordinateSystem(
                    lat  = Axis(values=mk_center_grid(-jnp.pi, jnp.pi, grid["nlat"])), 
                    lon  = Axis(values=mk_center_grid(0, 2*jnp.pi, grid["nlon"])),
                    z    = Axis(values=mk_center_grid(0, 1000, 8)),
                    time = Axis(values=[0]),
                ),
                prognostic = dict(
                    temperature = ["lon", "lat", "z"],
                ),
                boundary = dict(
                    surface_temp = ["lon", "lat"],
                ),
                metadata = dict(
                    time = ["time"],
                ),

            ),
        )
        
        component = MockComponent(config)
        rng_key = jax.random.PRNGKey(42)
        state = component.initialize(rng_key)

        state.boundary["surface_temp"] = state.boundary["surface_temp"].at[:].set(288.0)
 
        boundary_fields = component.get_boundary_fields(state)
        assert "surface_temp" in boundary_fields
        assert jnp.allclose(boundary_fields["surface_temp"], 288.0)
    """

    """
    def test_flux_requirements(self):


        def mk_center_grid(bnd_l, bnd_r, n):
            tmp = jnp.linspace(bnd_l, bnd_r, n+1)
            return (tmp[1:] + tmp[:-1])/2 
 
        grid = {"nlon": 10, "nlat": 5, "z": 8}
        config = ComponentConfig(
            name = "test",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            grid = grid,
            params = {"test_param": 42},
            comp_state_shp = ComponentStateShape(
                coord_sys = CoordinateSystem(
                    lat  = Axis(values=mk_center_grid(-jnp.pi, jnp.pi, grid["nlat"])), 
                    lon  = Axis(values=mk_center_grid(0, 2*jnp.pi, grid["nlon"])),
                    z    = Axis(values=mk_center_grid(0, 1000, 8)),
                    time = Axis(values=[0]),
                ),
                prognostic = dict(
                    temperature = ["lon", "lat", "z"],
                ),
                boundary = dict(
                    surface_temp = ["lon", "lat"],
                ),
                metadata = dict(
                    time = ["time"],
                ),

            ),
        )
        
        component = MockComponent(config)
 
        required = component.get_required_fluxes()
        provided = component.get_provided_fluxes()
        
        assert "heat" in required
        assert "moisture" in required
        assert "heat" in provided
        assert "moisture" in provided

    """
