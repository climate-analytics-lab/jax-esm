"""Tests for coupler"""
import unittest
from typing import Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from datetime import datetime
from jax_esm.coupling.flux_exchange import FluxExchanger
from jax_esm.coupling.coupler import (
    Coupler,
    CouplerConfig,
)

from jax_esm.components.base import (
    Component,
    ComponentConfig,
    create_field_group_class,
    create_component_state_class,
    create_component_forcing_class,
)

from jax_esm.components.domain import Domain

from jax_esm.utils.bulk_op import stack_objects

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
       
        D3_nodal_shape = config.domain.coords.nodal_shape
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
 
    def generate_step_function(
        self,
        jitted : bool = True,
    ):

        heat_capacity = self.config.params["heat_capacity"]

        def step_function(state, forcing, t):


            T = state.prog.air_temperature
            sim_time = state.prog.sim_time
           
            hist = [] 
            for step in range(self.config.substeps):
                T = T + self.subtimestep * forcing.flux.heat_flux / heat_capacity
                sim_time += self.subtimestep

            state = state.copy(
                prog_kwargs = dict(
                    sim_time = sim_time,
                    air_temperature = T,
                ),
            )

            print("Air_temp = ", T[0, 0])

            hist.append(dict(
                sim_time = sim_time,
                air_temperature = T,
                heat_flux = forcing.flux.heat_flux,
                sea_surface_temperature = forcing.scalar.sea_surface_temperature,
            ))
            
            return state, stack_objects( hist )
            
        return jax.jit(step_function) if jitted else step_function

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
       
        D3_nodal_shape = config.domain.coords.nodal_shape
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
                    ("heat_flux", float, D2_nodal_shape),
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
 
    def generate_step_function(
        self,
        jitted : bool = True,
    ):
        
        heat_capacity = self.config.params["heat_capacity"]

        def step_function(state, forcing, t):

            T = state.prog.sea_surface_temperature
            sim_time = state.prog.sim_time
           
            hist = [] 
            for step in range(self.config.substeps):
            
                T = T + self.subtimestep * forcing.flux.heat_flux / heat_capacity
                sim_time += self.subtimestep

            state = state.copy(
                prog_kwargs = dict(
                    sim_time = sim_time,
                    sea_surface_temperature = T,
                    heat_flux = forcing.flux.heat_flux,
                ),
            )

            hist.append(dict(
                sim_time = sim_time,
                sea_surface_temperature = T,
                air_temperature = T,
                heat_flux = forcing.flux.heat_flux,
            ))
 
            return state, stack_objects( hist )
            
        return jax.jit(step_function) if jitted else step_function

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




class TestCoupler(unittest.TestCase):

    """Test coupler functionality."""


    def generate_atm_ocn_flux_exchanger(
        self,
        components,
    ):

        def transformation(state_group, forcing_group):
            
            atm = state_group["atm"] 
            ocn = state_group["ocn"] 
            
            heat_flux = air_density * air_heat_capacity_under_constant_pressure * exchange_coefficient_of_heat * ( ocn.prog.sea_surface_temperature - atm.prog.air_temperature ) * atm.prog.wind_speed
            
            forcing_group["atm"].scalar.sea_surface_temperature = ocn.prog.sea_surface_temperature
            forcing_group["atm"].flux.heat_flux = heat_flux
            forcing_group["ocn"].flux.heat_flux = heat_flux


            return forcing_group


        return FluxExchanger(
            components = components,
            forcing_classes = dict( atm = components["atm"].component_forcing_class ),
            source_variables = dict(
                atm = [ ("prog", "air_temperature"), ("prog", "wind_speed") ], 
                ocn = [ ("prog", "sea_surface_temperature") ], 
            ),
            target_variables = dict(
                 atm = [ ("flux", "heat_flux"), ("scalar", "sea_surface_temperature") ], 
                 ocn = [ ("flux", "heat_flux"), ], 
            ),
            transformation = transformation,
        )


    def generate_coupler(
        self,
        atm_horizontal_resolution:int = 31,
        ocn_horizontal_resolution:int = 31,
        flux_exchanger_names : List[ str ] = [ "atm_ocn_flux_exchanger" ],
    ):
 
        atm_config = ComponentConfig(
            name = "test",
            start_dt = datetime(year=2001, month=1, day=1),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            domain = Domain.from_resolution_all_ocean(horizontal_resolution=atm_horizontal_resolution),
            params = dict( heat_capacity = 1004.0 * 1e4 ), # Cp * total mass in air column ( 1e4 kg / m^2)
        )
 
        ocn_config = ComponentConfig(
            name = "test",
            start_dt = datetime(year=2001, month=1, day=1),
            timestep = 1800.0,
            substeps = 2,
            save_interval = 1800.0,
            domain = Domain.from_resolution_all_ocean(horizontal_resolution=atm_horizontal_resolution),
            params = dict( heat_capacity = 1029 * 4996 * 50 ), # 50 meter thick ocean water
        )
 
        components = dict(
            atm = MockAtmosphere(atm_config),
            ocn = MockOcean(ocn_config),
        )
     
        flux_exchangers = [ getattr(self, f"generate_{flux_exchanger_name:s}")(components) for flux_exchanger_name in flux_exchanger_names ]

 
        return Coupler(
            components = components,
            flux_exchangers = flux_exchangers,
            config = CouplerConfig(
                timestep = 86400.0,
            )
        )


 
    def test_coupler_initialization(self):
        
        """Test coupler initialization."""
        
        coupler = self.generate_coupler()
        
 
    def test_coupler_stepping(self):
        
        """Test flux exchanger transformation."""
 
        coupler = self.generate_coupler()
        
        init_state = coupler.initialize()
       
        air_temperature = 273.15+20
        sea_surface_temperature = 273.15+25
        wind_speed = 8.0

        expected_flux = air_density * exchange_coefficient_of_heat * air_heat_capacity_under_constant_pressure * (sea_surface_temperature - air_temperature) * wind_speed

        init_state.ocn.prog.sea_surface_temperature = init_state.ocn.prog.sea_surface_temperature * 0 + sea_surface_temperature
        init_state.atm.prog.air_temperature = init_state.atm.prog.air_temperature * 0 + air_temperature
        init_state.atm.prog.wind_speed = init_state.atm.prog.wind_speed * 0 + wind_speed

        final_state, predictions = coupler.run(
            init_cplstate = init_state,
            start_time = 0.0,
            end_time = 86400.0 * 5,
            jax_scan = False,
        )  
        
        assert jnp.allclose( predictions["atm"]["sea_surface_temperature"][0, :, :], sea_surface_temperature)
        assert jnp.allclose( predictions["atm"]["heat_flux"][0, :, :], expected_flux)
        assert jnp.allclose( predictions["ocn"]["heat_flux"][0, :, :], expected_flux)

        nrecord = predictions["ocn"]["heat_flux"].shape[0]


        # Test SST evolves 
        assert jnp.all( (predictions["ocn"]["sea_surface_temperature"][:-1, :, :] - predictions["ocn"]["sea_surface_temperature"][1:, :, :]) != 0.0)


        # Test SST is passed to atmosphere in the next time step 
        assert jnp.all( (predictions["ocn"]["sea_surface_temperature"][:-1, :, :] - predictions["atm"]["sea_surface_temperature"][1:, :, :]) == 0.0)



