
from datetime import datetime
from typing import List 
from jax_esm.components.domain import Domain
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


class CoupledJCMSlabOceanModel(Coupler):


    def __init__(
        self,
        topo_file: Optional[str] = None,
        horizontal_resolution: int = 31,
        layers: int = 8,
        coupling_timestep: float = 86400.0, 
        JCM_substeps = 24,
        SlabOceanModel_substeps = 1,
        start_datetime = datetime(year=2025, month=1, day=1),
    ):
        
        if topo_file is None:
            domain = Domain.from_resolution_all_ocean(horizontal_resolution = horizontal_resolution)
        else:
            domain = Domain.from_file_and_resolution(filename=topo_file, horizontal_resolution = horizontal_resolution)

        domain.meta["layers"] = layers

        JCM_config_dict = dict(
            name="JCM_config",
            timestep = coupling_timestep,
            start_dt = start_datetime,
            substeps = JCM_substeps,
            save_interval = coupling_timestep / JCM_substeps,
            domain = domain,
            params=dict(
            ),
        )

        SlabOceanModel_config_dict = dict(
            name="SlabOceanModel_config",
            timestep = coupling_timestep,
            start_dt = start_datetime,
            substeps = SlabOceanModel_substeps,
            save_interval = coupling_timestep / SlabOceanModel_substeps,
            domain = domain,
            params=dict(
            ),
        )

        atm_model = JCM(**JCM_config_dict)
        ocn_model = SlabOceanModel(**SlabOceanModel_config_dict)
        
        components = dict(
            atm = atm_model,
            ocn = ocn_model,
        )
     
        flux_exchangers = [ self.generate_atm_ocn_flux_exchanger(components), ]
      
        super().__init__(
            components = components,
            flux_exchangers = flux_exchangers,
            config = CouplerConfig(
                timestep = coupling_time,
            ),
        )



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


