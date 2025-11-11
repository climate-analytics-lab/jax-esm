import jax_datetime as jdt
from typing import List, Optional, Dict 
from jax_esm.components.domain import Domain
from jax_esm.components.base import Component
from jax_esm.coupling.forcing_mapper import ForcingMapper
from jax_esm.coupling.coupler import Coupler

from jax_esm.components.SlabAtmosphereModel import SlabAtmosphereModel
from jax_esm.components.JCM import JCM

import jcm
from pathlib import Path

from jax_esm.utils.bilinear_interp import BilinearInterpolator
import jax.numpy as jnp

def couple_atm_ocn(
    atm : Component,
    ocn : Component,
    coupling_timestep: float = 86400.0, 
    start_datetime : jdt.Datetime = jdt.to_datetime("2025-01-01"),
):

    components = dict(
        atm = atm,
        ocn = ocn,
    )

    surface_flux_name = None
    if isinstance(atm, JCM):
        surface_flux_name = "phydata.surface_flux.hfluxn"
    elif isinstance(atm, SlabAtmosphereModel):
        surface_flux_name = "phydata.hfluxn"
    else:
        raise ValueError("Currently only deal with JCM and SlabAtmosphereModel.")

    interpolators = generate_atm_ocn_interpolators(dict(atm=atm.domain, ocn=ocn.domain))
    forcing_mapper = ForcingMapper(components=components)
    forcing_mapper.add_forcing_mapping("atm", "ocn", {
        surface_flux_name : "flux.total_heat_flux", 
    })

    forcing_mapper.add_forcing_mapping("ocn", "atm", {
        "prog.T" : "scalar.sea_surface_temperature",
    })

    forcing_mapper.add_transformation("ocn", "atm", "prog.T", "scalar.sea_surface_temperature", interpolators["ocn_to_atm"])
    forcing_mapper.add_transformation("atm", "ocn", surface_flux_name, "flux.total_heat_flux", interpolators["atm_to_ocn_hfluxn"])

    return Coupler(
        components = components,
        forcing_mapper = forcing_mapper,
        coupling_timestep = 86400.0,
    )


def generate_atm_ocn_interpolators(
    domain : Dict[str, Domain],
):
    interpolators = {}

    atm_domain = domain["atm"]
    ocn_domain = domain["ocn"]

    if atm_domain.grid_specification.grid_type in ["JCM", "Veros"] and ocn_domain.grid_specification.grid_type in ["JCM", "Veros"]:

        def find_latitude_longitude(domain: Domain):
            
            T_grid = domain.grids["T"] 

            lat_dim_idx = next( i for i, axis_name in enumerate(T_grid.axis_names) if axis_name == "latitude")
            lon_dim_idx = next( i for i, axis_name in enumerate(T_grid.axis_names) if axis_name == "longitude")

            return T_grid.axis_values[lat_dim_idx], T_grid.axis_values[lon_dim_idx] 
        
        atm_latitude, atm_longitude = find_latitude_longitude(atm_domain)
        ocn_latitude, ocn_longitude = find_latitude_longitude(ocn_domain)

        atm_latlon = atm_domain.grid_specification.grid_type == "Veros"
        ocn_latlon = ocn_domain.grid_specification.grid_type == "Veros"

        interpolator_atm_to_ocn = BilinearInterpolator(
            longitude_source_deg = atm_longitude,
            latitude_source_deg = atm_latitude,
            longitude_target_deg = ocn_longitude,
            latitude_target_deg = ocn_latitude,
            periodic_longitude = True,
            target_mask = None,
            source_mask = None,
            fill_value = 0.0,
        )

        interpolator_ocn_to_atm = BilinearInterpolator(
            longitude_target_deg = atm_longitude,
            latitude_target_deg = atm_latitude,
            longitude_source_deg = ocn_longitude,
            latitude_source_deg = ocn_latitude,
            periodic_longitude = True,
            target_mask = None,
            source_mask = None,
            fill_value = 0.0,
        )
        
        def conditional_transpose(a, flag):
            return jnp.transpose(a) if flag else a
 
        def atm_to_ocn_hfluxn(in_array):
            in_array = in_array.sum(axis=-1)
            return conditional_transpose(
                interpolator_atm_to_ocn.apply_scalar(conditional_transpose(in_array, not atm_latlon)),
                not ocn_latlon
            )

        def ocn_to_atm(in_array):
            return conditional_transpose(
                interpolator_ocn_to_atm.apply_scalar(conditional_transpose(in_array, not ocn_latlon)),
                not atm_latlon
            )
        interpolators["ocn_to_atm"] = ocn_to_atm
        interpolators["atm_to_ocn_hfluxn"] = atm_to_ocn_hfluxn
    else:
        raise Exception("Currently only support grids in JCM and Veros.")

    return interpolators
