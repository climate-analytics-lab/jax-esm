"""Tests for component interface."""

from typing import List 

import jax
import jax.numpy as jnp
import pytest
import unittest

from jax_esm.components.JCM import JCM
from jax_esm.components.SlabOceanModel import SlabOceanModel
from jax_esm.components.base import ComponentConfig
from jax_esm.coupling.flux_exchange import FluxExchanger
from jax_esm.coupling.coupler import (
    Coupler,
    CouplerConfig,
)


import jcm
from pathlib import Path


class TestCoupledJCMSlabOceanModel(unittest.TestCase):
    
    """Test component interface."""
 
    def generate_atm_ocn_flux_exchanger(
        self,
        components,
    ):

        def transformation(state_group, forcing_group):
            
            atm = state_group["atm"] 
            ocn = state_group["ocn"] 
           
            forcing_group["atm"].scalar.sea_surface_skin_temperature = ocn.prog.T
            forcing_group["ocn"].flux.heat_flux = atm.phydata.surface_flux.hfluxn.sum(axis=-1)

            return forcing_group

        return FluxExchanger(
            components = components,
            forcing_classes = dict(
                atm = components["atm"].component_forcing_class,
                ocn = components["ocn"].component_forcing_class,
            ),
            source_variables = dict(
                atm = [ ("phydata", "air_temperature"), ("prog", "wind_speed") ], 
                ocn = [ ("prog", "T") ], 
            ),
            target_variables = dict(
                 atm = [ ("scalar", "sea_surface_skin_temperature") ], 
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
       
         
        shared_topo_file = Path(jcm.__file__).parent / "data" / "bc" / "t30" / "clim" / "boundaries.nc"
        atm_model = JCM.generate_default_model(layers=8, topo_file=shared_topo_file)
        ocn_model = SlabOceanModel.generate_default_model(topo_file=shared_topo_file)
        
        components = dict(
            atm = atm_model,
            ocn = ocn_model,
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
        model = self.generate_coupler()



    def test_coupler_stepping(self):
        """Test coupler initialization."""
        model = self.generate_coupler()
        
        init_state = model.initialize()
       
        final_state, predictions = model.run(
            init_cplstate = init_state,
            start_time = 0.0,
            end_time = 86400.0 * 5,
            jax_scan = False,
        )  
        
