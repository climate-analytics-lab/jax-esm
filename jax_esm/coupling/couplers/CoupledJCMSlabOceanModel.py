
from datetime import datetime
from typing import List, Optional, Dict 
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
        JCM_grid_specification: str = "JCM::T31",
        SlabOceanModel_grid_specification: str = "JCM::T31",
        JCM_layers: int = 8,
        coupling_timestep: float = 86400.0, 
        JCM_substeps = 24,
        SlabOceanModel_substeps = 1,
        SlabOceanModel_relaxation_time = 60 * 86400.0,
        start_datetime = datetime(year=2025, month=1, day=1),
        JCM_topography_file: Optional[str] = None,
        JCM_mask_file: Optional[str] = None,
        SlabOceanModel_topography_file: Optional[str] = None,
        SlabOceanModel_mask_file: Optional[str] = None,
    ):
 
        domain = Domain.from_grid_specification(
            grid_specification,
            topography_file = topography_file,
            mask_file = mask_file,
        )
        domain.meta["layers"] = JCM_layers

        JCM_config = ComponentConfig(
            name="JCM_config",
            timestep = coupling_timestep,
            start_dt = start_datetime,
            substeps = JCM_substeps,
            save_interval = coupling_timestep / JCM_substeps,
            domain = domain,
            params=dict(
            ),
        )

        SlabOceanModel_config = ComponentConfig(
            name="SlabOceanModel_config",
            timestep = coupling_timestep,
            start_dt = start_datetime,
            substeps = SlabOceanModel_substeps,
            save_interval = coupling_timestep / SlabOceanModel_substeps,
            domain = domain,
            params=dict(
                relaxation_time = SlabOceanModel_relaxation_time,
                SST_clim_file = topography_file,
            ),
        )

        components = dict(
            atm = JCM(JCM_config),
            ocn = SlabOceanModel(SlabOceanModel_config),
        )
     
        flux_exchangers = [ self.generate_atm_ocn_flux_exchanger(components), ]
      
        super().__init__(
            components = components,
            flux_exchangers = flux_exchangers,
            config = CouplerConfig(
                timestep = coupling_timestep,
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
            forcing_group["ocn"].flux.total_heat_flux = atm.phydata.surface_flux.hfluxn.sum(axis=-1)

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


