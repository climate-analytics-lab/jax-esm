"""Tests for flux exchange."""
import unittest
import jax.numpy as jnp
import dinosaur
from datetime import datetime
from jax_esm.coupling.flux_exchange import FluxExchanger

from jax_esm.components.base import (
    Component,
    ComponentConfig,
    create_field_group_class,
    create_component_state_class,
    create_component_forcing_class,
)

def get_coords(horizontal_resolution=31) -> dinosaur.coordinate_systems.CoordinateSystem:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425).
    """
    try:
        horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, f'T{horizontal_resolution}')
    except AttributeError:
        raise ValueError(f"Invalid horizontal resolution: {horizontal_resolution}. Must be one of: 21, 31, 42, 85, 106, 119, 170, 213, 340, or 425.")
    
    return dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=horizontal_grid(radius=1.0),#PHYSICS_SPECS.radius),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates([0.0, 1.0])
    )


exchange_coefficient_of_heat = 1e-3
air_density = 1.2 # kg/m^3
air_heat_capacity_under_constant_pressure = 1004.0 # J / K / kg


class MockAtmosphere(Component):

    """
    Mock atmosphere model
    """
        
    def __init__(
        self,
        config: ComponentConfig,
    ):
        """Initialize mock atmosphere model."""
        
        super().__init__(config)

        self.subtimestep = config.timestep / config.substeps
       
        D3_nodal_shape = config.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]

        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("air_temperature", float, D2_nodal_shape),
                    ("wind_speed", float, D2_nodal_shape),
                ],
            ),

            phydata_cls =  create_field_group_class(
                cls_name = "phydata",
                fields = [
                    ("heat_flux", float, D2_nodal_shape),
                ],
            ),
        )


        self.component_forcing_class = create_component_forcing_class(
            flux_cls =  create_field_group_class(
                cls_name = "flux",
                fields = [
                    ("heat_flux", float, D2_nodal_shape),
                ],
            ),
            scalar_cls =  create_field_group_class(
                cls_name = "scalar",
                fields = [
                    ("sea_surface_temperature", float, D2_nodal_shape),
                ],
            ),

        )


    def initialize(self):

        init_state = self.component_state_class.zeros()


        return init_state
 
    def gen_step_fn(
        self,
    ):

        heat_capacity = self.config.params["heat_capacity"]

        @jax.jit
        def step_fn(state, t):


            T = state.prog.air_temperature
            sim_time = state.prog.sim_time
           
            hist = [] 
            for step in range(self.config.substeps):
            
                T = T + self.subtimestep * state.phydata.heatflx / heat_capacity
                sim_time += self.subtimestep

            state = state.copy(
                prog_kwargs = dict(
                    sim_time = sim_time,
                    air_temperature = T,
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
        

class MockOcean(Component):

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
       
        D3_nodal_shape = config.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]

        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("sea_surface_temperature", float, D2_nodal_shape),
                    ("mld", float, D2_nodal_shape),
                ],
            ),

            phydata_cls =  create_field_group_class(
                cls_name = "phydata",
                fields = [
                ],
            ),
        )


        self.component_forcing_class = create_component_forcing_class(
            flux_cls =  create_field_group_class(
                cls_name = "flux",
                fields = [
                    ("heat_flx", float, D2_nodal_shape),
                ],
            ),
            scalar_cls =  create_field_group_class(
                cls_name = "scalar",
                fields = [
                ],
            ),

        )


    def initialize(self):

        init_state = self.component_state_class.zeros()


        return init_state
 
    def gen_step_fn(
        self,
    ):
        
        heat_capacity = self.config.params["heat_capacity"]

        @jax.jit
        def step_fn(state, t):


            T = state.prog.sea_surface_temperature
            sim_time = state.prog.sim_time
           
            hist = [] 
            for step in range(self.config.substeps):
            
                T = T + self.subtimestep * state.phydata.heatflx / heat_capacity
                sim_time += self.subtimestep

            state = state.copy(
                prog_kwargs = dict(
                    sim_time = sim_time,
                    sea_surface_temperature = T,
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
        
class TestFluxExchanger(unittest.TestCase):
    """Test flux exchange functionality."""

    def generate_components(
        self,
        atm_horizontal_resolution:int = 31,
        ocn_horizontal_resolution:int = 31,
    ):
 
        atm_config = ComponentConfig(
            name = "test",
            start_dt = datetime(year=2001, month=1, day=1),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            coords = get_coords(horizontal_resolution=atm_horizontal_resolution),
            params = {},
        )
 
        ocn_config = ComponentConfig(
            name = "test",
            start_dt = datetime(year=2001, month=1, day=1),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            coords = get_coords(horizontal_resolution=ocn_horizontal_resolution),
            params = {},
        )
       
        return dict(
            atm = MockAtmosphere(atm_config),
            ocn = MockOcean(ocn_config),
        )
  

    def generate_atm2ocn_flux_exchanger_noremap(
        self,
        components,
    ):
 
        return FluxExchanger(
            components = components,
            forcing_classes = dict( ocn = components["ocn"].component_forcing_class ),
            source_variables = dict( atm = [ ("phydata", "heat_flux") ] ),
            target_variables = dict( ocn = [ ("flux", "heat_flux") ] ),
        )

    def generate_ocn2atm_flux_exchanger_noremap_1(
        self,
        components,
    ):
 
        return FluxExchanger(
            components = components,
            forcing_classes = dict( atm = components["atm"].component_forcing_class ),
            source_variables = dict( ocn = [ ("prog", "sea_surface_temperature") ] ),
            target_variables = dict( atm = [ ("scalar", "sea_surface_temperature") ] ),
        )


    def generate_ocn2atm_flux_exchanger_noremap_2(
        self,
        components,
    ):

        def transformation(state_group, forcing_group):
            
            atm = state_group["atm"] 
            ocn = state_group["ocn"] 
            
            forcing_group["atm"].scalar.sea_surface_temperature = ocn.prog.sea_surface_temperature
            forcing_group["atm"].flux.heat_flux = air_density * air_heat_capacity_under_constant_pressure * exchange_coefficient_of_heat * ( ocn.prog.sea_surface_temperature - atm.prog.air_temperature ) * atm.prog.wind_speed


            return forcing_group


        return FluxExchanger(
            components = components,
            forcing_classes = dict( atm = components["atm"].component_forcing_class ),
            source_variables = dict(
                atm = [ ("prog", "air_temperature"), ("prog", "wind_speed") ], 
                ocn = [ ("prog", "sea_surface_temperature") ], 
            ),
            target_variables = dict( atm = [ ("flux", "heat_flux"), ("scalar", "sea_surface_temperature") ] ),
            transformation = transformation,
        )



 
    def test_flux_exchanger_initialization(self):
        
        """Test flux exchanger initialization."""
        components = self.generate_components()

        exchanger = self.generate_atm2ocn_flux_exchanger_noremap(components)
        assert exchanger.transformation
 
        exchanger = self.generate_ocn2atm_flux_exchanger_noremap_1(components)
        assert exchanger.transformation
 
        exchanger = self.generate_ocn2atm_flux_exchanger_noremap_2(components)
        assert exchanger.transformation
 


    def test_flux_exchanger_initialization_with_different_grids(self):
        
        """Test flux exchanger initialization failure due to grid mismatch."""
 
        components = self.generate_components(
            atm_horizontal_resolution = 21,
            ocn_horizontal_resolution = 31,
        )
        exchanger = None
        try:
            # Exchanger is suppose to fail due to grid mismatch
            exchanger = self.generate_atm2ocn_flux_exchanger_noremap(components)
        except Exception as e:
            pass
 
        assert exchanger is None
 
    def test_flux_exchanger_atm2ocn_noremap(self):
        
        """Test flux exchanger transformation."""
 
        components = self.generate_components()
        exchanger = self.generate_atm2ocn_flux_exchanger_noremap(components)

        forcing_group = exchanger.generate_empty_forcing_group()
        state_group = { component_name : component.initialize() for component_name, component in exchanger.components.items() }
       
        assert "atm" not in forcing_group 
        assert "ocn" in forcing_group 

        state_group["atm"].phydata.heat_flux = state_group["atm"].phydata.heat_flux * 0 + 680.0 

        forcing_group = exchanger.transformation(state_group, forcing_group)

        assert "atm" not in forcing_group 
        assert "ocn" in forcing_group 

        assert jnp.allclose( forcing_group["ocn"].flux.heat_flux , 680.0 )

    def test_flux_exchanger_ocn2atm_noremap_1(self):
        
        """Test flux exchanger transformation."""
 
        components = self.generate_components()
        exchanger = self.generate_ocn2atm_flux_exchanger_noremap_1(components)

        forcing_group = exchanger.generate_empty_forcing_group()
        state_group = { component_name : component.initialize() for component_name, component in exchanger.components.items() }
       
        assert "atm" in forcing_group 
        assert "ocn" not in forcing_group 

        sea_surface_temperature = 273.15+20
        
        state_group["ocn"].prog.sea_surface_temperature = state_group["ocn"].prog.sea_surface_temperature * 0 + sea_surface_temperature

        forcing_group = exchanger.transformation(state_group, forcing_group)

        assert "atm" in forcing_group 
        assert "ocn" not in forcing_group 

        assert jnp.allclose( forcing_group["atm"].scalar.sea_surface_temperature, sea_surface_temperature)

 
    def test_flux_exchanger_ocn2atm_noremap_2(self):
        
        """Test flux exchanger transformation."""
 
        components = self.generate_components()
        exchanger = self.generate_ocn2atm_flux_exchanger_noremap_2(components)

        forcing_group = exchanger.generate_empty_forcing_group()
        state_group = { component_name : component.initialize() for component_name, component in exchanger.components.items() }
       
        assert "atm" in forcing_group 
        assert "ocn" not in forcing_group 

        air_temperature = 273.15+20
        sea_surface_temperature = 273.15+25
        wind_speed = 8.0

        expected_flux = air_density * exchange_coefficient_of_heat * air_heat_capacity_under_constant_pressure * (sea_surface_temperature - air_temperature) * wind_speed


        state_group["ocn"].prog.sea_surface_temperature = state_group["ocn"].prog.sea_surface_temperature * 0 + sea_surface_temperature
        state_group["atm"].prog.air_temperature = state_group["atm"].prog.air_temperature * 0 + air_temperature
        state_group["atm"].prog.wind_speed = state_group["atm"].prog.wind_speed * 0 + wind_speed

        forcing_group = exchanger.transformation(state_group, forcing_group)

        assert "atm" in forcing_group 
        assert "ocn" not in forcing_group 
        
        assert jnp.allclose( forcing_group["atm"].scalar.sea_surface_temperature, sea_surface_temperature)
        assert jnp.allclose( forcing_group["atm"].flux.heat_flux, expected_flux)
                                             
