"""Tests for the coupler."""

import jax
import jax.numpy as jnp
import unittest

from jax_esm.coupling.coupler import Coupler, CouplerConfig

from jax_esm.components.base import (
    Component, BoundaryFluxes,
    ComponentConfig,
    create_component_state_class,
    create_field_group_class,
)

from jax_esm.utils.bulk_op import stack_objects

import pandas as pd

class MockAtmComponent(Component):

    """
    Mock atmosphere model
    """
        
    def __init__(
        self,
        config: ComponentConfig,
    ):
        """Initialize atmosphere model."""
        
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
        coupled: bool = True
    ):

        @jax.jit
        def step_fn(cplstate, t):

            state = cplstate.atm
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

class MockOcnComponent(Component):

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
        def step_fn(cplstate, t):


            state = cplstate.ocn
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


class MockFlxComponent(Component):

    """
    Mock flux model
    """
    
    def __init__(
        self,
        config: ComponentConfig,
    ):
        
        super().__init__(config)

        self.subtimestep = config.timestep / config.substeps
        
        D3_nodal_shape = (config.grid["z"], config.grid["lon"], config.grid["lat"])
        D2_nodal_shape = D3_nodal_shape[1:]

        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
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
        def step_fn(cplstate, t):


            state = cplstate.flx
            sim_time = state.prog.sim_time
           
            sim_time += self.config.timestep
            hist = [] 

            state = state.copy(
                prog_kwargs = dict(
                    sim_time = sim_time,
                ),
            )

            hist.append(dict(state=state))
            
            return state, stack_objects( hist )
            
        return step_fn




class TestCoupler(unittest.TestCase):
    """Test coupler functionality."""

    def gen_test_atm_component(self):

        grid = {"lat": 5, "lon": 10, "z": 1}
        config = ComponentConfig(
            name = "test",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 86400.0,
            substeps = 24,
            save_interval = 3600.0,
            grid = grid,
            params = {"heat_capacity": 1000.0},
        )
        
        return MockAtmComponent(config) 


    def gen_test_ocn_component(self):

        grid = {"lat": 5, "lon": 10, "z": 1}
        config = ComponentConfig(
            name = "test",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 86400.0,
            substeps = 4,
            save_interval = 12 * 3600.0,
            grid = grid,
            params = {"heat_capacity": 4e6},
        )
        
        return MockOcnComponent(config) 

    def gen_test_flx_component(self):

        grid = {"lat": 5, "lon": 10, "z":1}
        config = ComponentConfig(
            name = "test",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 86400.0,
            substeps = 4,
            save_interval = 12 * 3600.0,
            grid = grid,
            params = {},
        )
        
        return MockFlxComponent(config) 

    def gen_test_cpl(self, **kwargs):

        model_atm = self.gen_test_atm_component()
        model_ocn = self.gen_test_ocn_component()
        model_flx = self.gen_test_flx_component()


        d = dict(
            timestep = 1800.0,
        )

        d.update(kwargs)

        cpl = Coupler(
            components = dict(
                atm = model_atm,
                ocn = model_ocn,
                flx = model_flx,
            ),
            
            config = CouplerConfig(**d)
        )

        return cpl
    
    def test_coupler_initialization(self):

        """Test coupler initialization."""

        cpl = self.gen_test_cpl()
        
        assert len(cpl.components) == 3
        assert "atm" in cpl.components
        assert "ocn" in cpl.components
        assert "flx" in cpl.components
    
    def test_coupler_initialize_states(self):
        """Test state initialization through coupler."""
        
        cpl = self.gen_test_cpl()
        state = cpl.initialize()
        
        assert state.atm is not None
        assert state.ocn is not None
        assert state.flx is not None
        assert jnp.allclose(state.atm.prog.T, 0.0)
        assert jnp.allclose(state.ocn.prog.T, 0.0)
        assert jnp.allclose(state.flx.phydata.heatflx, 0.0)
    
    def test_coupler_step(self):
        """Test single coupling step."""
        
        cpl = self.gen_test_cpl()
 
        state = cpl.initialize()
        # Take one coupling step
        step_fn = cpl.gen_step_fn()
        new_state = step_fn(state, 0.0)
       
         
        # Check that coupling occurred (temperatures should have changed)
        #atm_temp_changed = not jnp.allclose(
        #    states["atmosphere"].prognostic["temperature"],
        #    new_states["atmosphere"].prognostic["temperature"]
        #)
        #ocean_temp_changed = not jnp.allclose(
        #    states["ocean"].prognostic["temperature"],
        #    new_states["ocean"].prognostic["temperature"]
        #)
        
        #assert True or atm_temp_changed or ocean_temp_changed
    
    def test_coupler_subcycling(self):
        
        """Test subcycling with different component timesteps."""
        # Atmosphere runs at 15 minutes, ocean at 30 minutes
        # Coupling at 30 minutes means atmosphere takes 2 steps per coupling
        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 900.0,
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params = {},
        )
        ocean_config = ComponentConfig(
            name="ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep=1800.0,  # 30 minutes
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
        
        atmosphere = MockAtmosphere(atm_config)
        ocean = MockOcean(ocean_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere, "ocean": ocean},
            config = CouplerConfig(
                timestep=1800.0,
            )
        )
        
        # Check that subcycles were calculated correctly
        assert coupler.time_integrator.timestep_info["atmosphere"].subcycles == 2
        assert coupler.time_integrator.timestep_info["ocean"].subcycles == 1
    
    def test_coupler_run(self):
        """Test running full simulation."""

        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 900.0,
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params = {},
        )
        ocean_config = ComponentConfig(
            name="ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep=1800.0,  # 30 minutes
            substeps = 2,
            save_interval = 1800.0,
            grid={"nlat": 5, "nlon": 10},
            params={},
        )
 
        
        atmosphere = MockAtmosphere(atm_config)
        ocean = MockOcean(ocean_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere, "ocean": ocean},
            config = CouplerConfig(
                timestep=1800.0,
            )

        )
        
        rng_key = jax.random.PRNGKey(42)
        initial_states = coupler.initialize(rng_key)
        
        # Run for 2 hours
        component_results = coupler.run(
            initial_states=initial_states,
            start_time=0.0,
            end_time=7200.0,  # 2 hours
        )
        
        # Should have 4 coupling steps
        assert len(component_results[atmosphere][1].time) == 4
        assert component_results["atmosphere"][1].metadata["time"] == 7200.0
        assert component_results["ocean"][1].metadata["time"] == 7200.0

        # Check history
        assert component_results[atmosphere][1].history.time[0] == 1800.0
        assert component_results[atmosphere][1].history.time[1] == 3600.0
        assert component_results[atmosphere][1].history.time[2] == 5400.0
        assert component_results[atmosphere][1].history.time[3] == 7200.0

    def test_add_remove_component(self):
        
        """Test adding and removing components."""
        atm_config = ComponentConfig(
            name = "atmosphere",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            grid = {},
            params = {},
        )

        atmosphere = MockAtmosphere(atm_config)
        
        coupler = Coupler(
            components={"atmosphere": atmosphere},
            config = CouplerConfig(
                timestep=1800.0,
            )
        )
        
        assert len(coupler.components) == 1
        
        # Add ocean
        ocean_config = ComponentConfig(
            name = "ocean",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            grid = {},
            params = {},
        )

        ocean = MockOcean(ocean_config)
        coupler.add_component("ocean", ocean)
        
        assert len(coupler.components) == 2
        assert "ocean" in coupler.components
        
        # Remove atmosphere
        coupler.remove_component("atmosphere")
        
        assert len(coupler.components) == 1
        assert "atmosphere" not in coupler.components
        assert "ocean" in coupler.components
