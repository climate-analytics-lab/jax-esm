
from datetime import datetime
from typing import List, Optional, Dict 
from jax_esm.components.domain import Domain
from jax_esm.components.SlabOceanModel import SlabOceanModel
from jax_esm.components.SlabAtmosphereModel import SlabAtmosphereModel
from jax_esm.components.base import ComponentConfig
from jax_esm.coupling.forcing_mapper import ForcingMapper
from jax_esm.coupling.coupler import Coupler

import jcm
from pathlib import Path

from jax_esm.utils.bilinear_interp import BilinearInterpolator
import jax.numpy as jnp

def couple_SlabAtmosphereModel_and_SlabOceanModel(
    atm : SlabAtmosphereModel,
    ocn : SlabOceanModel,
    coupling_timestep: float = 86400.0, 
    start_datetime = datetime(year=2025, month=1, day=1),
):

    components = dict(
        atm = atm,
        ocn = ocn,
    )

    forcing_mapper = ForcingMapper(components=components)

    forcing_mapper.add_forcing_mapping("atm", "ocn", {
        "phydata.total_heat_flux" : "flux.total_heat_flux", 
    })

    forcing_mapper.add_forcing_mapping("ocn", "atm", {
        "prog.T" : "scalar.sea_surface_temperature",
    })





    return Coupler(
        components = components,
        forcing_mapper = forcing_mapper,
        coupling_timestep = 86400.0,
    )


def generate_atm_ocn_flux_exchanger(
    self,
    components,
):

    atm_model = components["atm"]
    ocn_model = components["ocn"]
    
    atm_domain = atm_model.config.domain
    ocn_domain = ocn_model.config.domain

    if atm_domain.grid_specification.grid_type in ["JCM", "Veros"] and ocn_domain.grid_specification.grid_type in ["JCM", "Veros"]:

        def find_latitude_longitude(domain: Domain):
            
            T_grid = domain.grids["T"] 

            lat_dim_idx = next( i for i, axis_name in enumerate(T_grid.axis_names) if axis_name == "latitude")
            lon_dim_idx = next( i for i, axis_name in enumerate(T_grid.axis_names) if axis_name == "longitude")

            return T_grid.axis_values[lat_dim_idx], T_grid.axis_values[lon_dim_idx] 
        
        atm_latitude, atm_longitude = find_latitude_longitude(atm_domain)
        ocn_latitude, ocn_longitude = find_latitude_longitude(ocn_domain)

        interpolator_atm_to_ocn = BilinearInterpolator(
            longitude_source_deg = atm_longitude,
            latitude_source_deg = atm_latitude,
            longitude_target_deg = ocn_longitude,
            latitude_target_deg = ocn_latitude,
            periodic_longitude = True,
            #target_mask: Optional[Array] = None,
            #source_mask: Optional[Array] = None,
            fill_value = 0.0,
        )

        interpolator_ocn_to_atm = BilinearInterpolator(
            longitude_target_deg = atm_longitude,
            latitude_target_deg = atm_latitude,
            longitude_source_deg = ocn_longitude,
            latitude_source_deg = ocn_latitude,
            periodic_longitude = True,
            #target_mask: Optional[Array] = None,
            #source_mask: Optional[Array] = None,
            fill_value = 0.0,
        )

        atm_latlon = atm_domain.grid_specification.grid_type == "Veros"
        ocn_latlon = ocn_domain.grid_specification.grid_type == "Veros"
       
    else:
        raise Exception("Currently only support grids in JCM and Veros.")


    def conditional_transpose(a, flag):
        return jnp.transpose(a) if flag else a

    def transformation(state_group, forcing_group):

        atm = state_group["atm"]
        ocn = state_group["ocn"]
       
        forcing_group["atm"].scalar.sea_surface_skin_temperature = conditional_transpose(
            interpolator_ocn_to_atm.apply_scalar( conditional_transpose( ocn.prog.T, not ocn_latlon ) ),
            not atm_latlon
        )
        
        forcing_group["ocn"].flux.total_heat_flux = conditional_transpose(
            interpolator_atm_to_ocn.apply_scalar( conditional_transpose( atm.phydata.surface_flux.hfluxn.sum(axis=-1), not atm_latlon) ),
            not ocn_latlon
        )

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


