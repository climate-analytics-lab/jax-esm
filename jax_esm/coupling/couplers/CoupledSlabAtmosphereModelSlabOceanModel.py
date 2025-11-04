
from datetime import datetime
from typing import List, Optional, Dict 
from jax_esm.components.domain import Domain
from jax_esm.components.JCM import JCM
from jax_esm.components.SlabOceanModel import SlabOceanModel
from jax_esm.components.SlabAtmosphereModel import SlabAtmosphereModel
from jax_esm.components.base import ComponentConfig
from jax_esm.coupling.flux_exchange import FluxExchanger
from jax_esm.coupling.coupler import (
    Coupler,
    CouplerConfig,
)

from jax_esm import constants as constants
from pathlib import Path


class CoupledSlabAtmosphereModelSlabOceanModel(Coupler):


    def __init__(
        self,
        start_datetime = datetime(year=2025, month=1, day=1),
        atmosphere_grid_specification: str = "JCM::T31",
        ocean_grid_specification: str = "JCM::T31",
        coupling_timestep: float = 86400.0, 
        atmosphere_substeps = 24,
        ocean_substeps = 1,
        ocean_relaxation_time = 60 * 86400.0,
        atmosphere_topography_file: Optional[str] = None,
        atmosphere_mask_file: Optional[str] = None,
        ocean_topography_file: Optional[str] = None,
        ocean_mask_file: Optional[str] = None,
    ):
 
        atmosphere_domain = Domain.from_grid_specification(
            atmosphere_grid_specification,
            topography_file = atmosphere_topography_file,
            mask_file = atmosphere_mask_file,
        )

        atmosphere_config = ComponentConfig(
            name="atmosphere_config",
            timestep = coupling_timestep,
            start_dt = start_datetime,
            substeps = atmosphere_substeps,
            save_interval = coupling_timestep / atmosphere_substeps,
            domain = atmosphere_domain,
            params=dict(
            ),
        )

        ocean_domain = Domain.from_grid_specification(
            ocean_grid_specification,
            topography_file = ocean_topography_file,
            mask_file = ocean_mask_file,
        )

        ocean_config = ComponentConfig(
            name="ocean_config",
            timestep = coupling_timestep,
            start_dt = start_datetime,
            substeps = ocean_substeps,
            save_interval = coupling_timestep / ocean_substeps,
            domain = ocean_domain,
            params=dict(
                relaxation_time = ocean_relaxation_time,
                SST_clim_file = ocean_topography_file,
            ),
        )

        components = dict(
            atm = SlabAtmosphereModel(atmosphere_config),
            ocn = SlabOceanModel(ocean_config),
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
           
            surface_air_density = 1.22 # kg / m^3
            drag_coefficient = 1e-3 # scalar

            # Simple bulk formula
            flux = surface_air_density * drag_coefficient * ((atm.prog.mean_zonal_wind_velocity ** 2 + atm.prog.mean_meridional_wind_velocity**2)**0.5) * constants.atmosphere_specific_heat_capacity_under_constant_pressure * (ocn.prog.T - atm.prog.mean_air_temperature)

            forcing_group["atm"].flux.total_heat_flux = flux
            forcing_group["ocn"].flux.total_heat_flux = flux

            return forcing_group

        return FluxExchanger(
            components = components,
            forcing_classes = dict(
                atm = components["atm"].component_forcing_class,
                ocn = components["ocn"].component_forcing_class,
            ),
            source_variables = dict(
                atm = [ ("prog", "mean_air_temperature"), ("prog", "mean_zonal_wind_velocity"), ("prog", "mean_meridional_wind_velocity") ], 
                ocn = [ ("prog", "T") ], 
            ),
            target_variables = dict(
                 atm = [ ("flux", "heat_flux") ], 
                 ocn = [ ("flux", "heat_flux"), ], 
            ),
            transformation = transformation,
        )


