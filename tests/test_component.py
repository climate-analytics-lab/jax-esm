"""Tests for component interface."""

import jax
import jax.numpy as jnp
import unittest
import pandas as pd
from jax_esm import (
    Component,
    ComponentConfig,
    BoundaryFluxes,
)

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

        self.coords = config.params["coords"]
        self.start_dt = config.start_dt
        self.timestep = config.timestep
        self.substeps = config.substeps
        self.subtimestep = self.timestep / self.substeps
        
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]

        self.component_state_class = create_component_state_class(
            prog_cls = createFieldsClass(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("T", float, D2_nodal_shape),
                ],
            ),

            phydata_cls =  createFieldsClass(
                cls_name = "phydata",
                fields = [
                    ("heatflx", float, D2_nodal_shape),
                ],
            ),
        )

    def initialize(self):

        init_state = self.component_state_class.zeros()

        init_state = init_state.copy(
            prog_kwargs = dict(T = init_state.prog.T + 300.0),
            phydata_kwargs = dict(heatflx = init_state.phydata.heatflx + 500.0),
        )


        return init_state
 
    def gen_step_fn(
        self,
    ):

        # Find day of the year to locate climatology
        ref_dt = pd.Timestamp(year=self.start_dt.year, month=self.start_dt.month, day=1)
        start_dt_offset = jnp.int_(jnp.floor( ( self.start_dt - ref_dt ) / pd.Timedelta(days=1) ))
        
        @jax.jit
        def step_fn(cpl, t):

            somstate = cpl["ocn"]["state"]
            fmstate  = cpl["flx"]["state"]

            days_after_start = jnp.floor( somstate.sim_time / 86400.0 ).astype(jnp.int32)
            
            # This snapshot SST will be frozen
            snapshot_SST_clim = self.SST_clim[:, :, start_dt_offset + days_after_start]
            snapshot_SST_clim = jnp.where(self.fmask_ocn != 0, snapshot_SST_clim, 273.15+15)

            new_Tanom = somstate.T - snapshot_SST_clim
            
            for step in range(self.substeps):
                new_Tanom = self.time_factor * ( new_Tanom + self.cd_factor * ( - (
                    fmstate.hfluxn[:, :, 0]
                )))

                somstate = somstate.copy(
                    sim_time = somstate.sim_time + self.subtimestep
                )
            
            new_T = new_Tanom + snapshot_SST_clim

            new_state = somstate.copy(
                T = new_T,
            )
            
            new_phydata = cpl["ocn"]["phydata"].copy()
            
            return dict(state=new_state, phydata=new_phydata), stack_objects( [ dict(state=new_state, phydata=new_phydata) , ] )
            
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
    
    def test_component_initialization(self):
        """Test component initialization."""
 

        
        def mk_center_grid(bnd_l, bnd_r, n):
            tmp = jnp.linspace(bnd_l, bnd_r, n+1)
            return (tmp[1:] + tmp[:-1])/2 
        
        grid = {"nlat": 5, "nlon": 10, "z": 8, "time": 0}
        config = ComponentConfig(
            name = "test",
            start_dt = pd.Timestamp("2001-01-01"),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            grid = grid,
            params = {"test_param": 42},
        )
        
        component = MockComponent(config)
        
        assert component.name == "test"
        assert component.timestep == 1800.0
        assert component.config.params["test_param"] == 42
    
    def test_component_initialize_state(self):
        """Test state initialization."""

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
                metadata = dict(
                    time = ["time"],
                ),

            ),
        )
 
        component = MockComponent(config)
        

        rng_key = jax.random.PRNGKey(42)
        state = component.initialize(rng_key)
        state.prognostic["temperature"] = state.prognostic["temperature"].at[:].set(280.0) 
        state.metadata["time"] = state.metadata["time"].at[:].set(155.0) 

        assert "temperature" in state.prognostic
        assert state.prognostic["temperature"].shape == (10, 5, 8)
        assert jnp.allclose(state.prognostic["temperature"], 280.0)
        assert state.metadata["time"] == jnp.array([155.0,])
        


    def test_component_step(self):
        """Test component stepping."""

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
                metadata = dict(
                    time = ["time"],
                ),

            ),
        )
        
        component = MockComponent(config)
        rng_key = jax.random.PRNGKey(42)
        state = component.initialize(rng_key)
        
        state.prognostic["temperature"] = state.prognostic["temperature"].at[:].set(280.0) 
        
        forcing = BoundaryFluxes(
            heat=jnp.ones((5, 10)) * 100.0,
            moisture=jnp.ones((5, 10)) * 0.001,
            momentum_u=jnp.zeros((5, 10)),
            momentum_v=jnp.zeros((5, 10)),
            tracers=jnp.ones((5, 10)) * 100.0,
        )
        
        new_state, output_fluxes = component.step(state, forcing, 1800.0)
        
        # Check state update
        assert new_state.metadata["time"] == 1800.0
        expected_temp = 280.0 + 0.1 * 1800.0
        assert jnp.allclose(new_state.prognostic["temperature"], expected_temp)
        
        # Check output fluxes
        assert jnp.allclose(output_fluxes.heat, 100.0)
        assert jnp.allclose(output_fluxes.moisture, 0.001)
    
    def test_component_tendencies(self):
        """Test tendency computation."""


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
                metadata = dict(
                    time = ["time"],
                ),

            ),
        )
        
        component = MockComponent(config)
        rng_key = jax.random.PRNGKey(42)
        state = component.initialize(rng_key)
        
        forcing = BoundaryFluxes(
            heat=jnp.zeros((10, 20)),
            moisture=jnp.zeros((10, 20)),
            momentum_u=jnp.zeros((10, 20)),
            momentum_v=jnp.zeros((10, 20)),
            tracers={},
        )
        
        tendencies = component.compute_tendencies(state, forcing)
        
        assert "temperature" in tendencies
        assert jnp.allclose(tendencies["temperature"], 0.1)
    
    def test_boundary_fields(self):
        """Test boundary field extraction."""


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
    
    def test_flux_requirements(self):
        """Test flux requirement declarations."""


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
