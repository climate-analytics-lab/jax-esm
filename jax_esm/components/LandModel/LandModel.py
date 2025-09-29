"""Slab ocean model component."""

from typing import Dict, Tuple, Any, List
from collections import namedtuple

import jax
import jax.numpy as jnp
import tree_math
from dataclasses import make_dataclass

from jax import Array

from jax_esm import constants as constants
from jax_esm.utils.meta_prog_class import createFieldsClass
from jax_esm.utils.bulk_op import stack_objects

from jcm.geometry import Geometry


from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

from jax_esm.components.base import (
    Component,
    ComponentConfig,
)



class LandModel(Component):


    """    
    Land model migrating from jcm

    """
        
    def __init__(
        self,
        config: ComponentConfig,
    ):
        
        """Initialize land model."""
        
        super().__init__(config)

        self.ocn_rho = constants.ocn_rho # Seawater density (kg / m^3)
        self.ocn_cp = constants.ocn_cp   # Seawater specific heat capacity (J/kg/K)

        self.coords = config.params["coords"]
        self.geometry = config.params["geometry"]
        self.relaxation_time = config.params["relaxation_time"]

        self.start_dt = config.start_dt
        self.timestep = config.timestep
        self.substeps = config.substeps
        self.subtimestep = self.timestep / self.substeps
         
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        
        self.stateClass = createFieldsClass(
            cls_name = "land_state",
            fields = [
                ("sim_time", float, ()),
                ("stl", float, D2_nodal_shape),
            ],
        )

        self.phydataClass =  createFieldsClass(
            cls_name = "land_diag",
            fields = [
            ],
        )

        self.init_state = self.stateClass.zeros()
        self.init_phydata = self.phydataClass.zeros()

        # =========================================================================
        # Initialize slab ocean model boundary conditions
        # =========================================================================

        llon_rad = jnp.repeat(
            jnp.expand_dims(
                self.coords.horizontal.longitudes,
                axis = 1,
            ),
            repeats = D2_nodal_shape[1],
            axis = 1,
        )

        llat_rad = jnp.repeat(
            jnp.expand_dims(
                self.coords.horizontal.latitudes,
                axis = 0,
            ),
            repeats = D2_nodal_shape[0],
            axis = 0,
        )

        
        # initialize mask
        self.fmask_ocn = jnp.ones_like(llat_rad)

        params = config.params
         
        if "boundaries" in params and params["boundaries"] is not None:

            print("Boundary exists. ")
            print("Boundary file: ", params["boundary_file"])
            
            boundaries = params["boundaries"]
            thrsh = 0.3


    
            fmask_l = boundaries.fmask
            bmask_l = jnp.where(fmask_l >= thrsh, 1.0, 0.0)

            # Update fmask_l based on the conditions
            fmask_l = jnp.where(fmask_l >= thrsh,
                            jnp.where(boundaries.fmask > (1.0 - thrsh), 1.0, fmask_l), 0.0)

            ##### Land-surface temperature #####

            # Note: `clim` stands for climatology.
            stl_clim = jnp.array(xr.open_dataset(params["boundary_file"])["stl"])
            
            # instead of using a check forchk, we check that 0.0 < stl12 < 400 and if it isn't we set it to 273
            # this might not work exactly because stl12 is ix,il,12 and bmask is ix,il
            stl_clim = jnp.where(
                (bmask_l[:,:,jnp.newaxis] > 0.0) & ((stl_clim < 0.0) | (stl_clim > 400.0)),
                273.0,
                stl_clim,
            )
            self.stl_clim = stl_clim

            ##### Snow depth #####
            snowd_clim = jnp.asarray(xr.open_dataset(params["boundary_file"])["snowd"])

            # simple sanity check - same method ras above for stl12
            snowd_clim = jnp.where(
                (bmask_l[:,:,jnp.newaxis] > 0.0) & ((snowd_clim < 0.0) | (snowd_clim > 20000.0)), 
                0.0, 
                snowd_clim,
            )
            self.snowd_clim = snowd_clim

            # Read soil moisture and compute soil water availability using vegetation fraction
            # Read vegetation fraction
            veg_high = jnp.asarray(xr.open_dataset(params["boundary_file"])["vegh"])
            veg_low  = jnp.asarray(xr.open_dataset(params["boundary_file"])["vegl"])
            
            # Combine high and low vegetation fractions
            veg = jnp.maximum(0.0, veg_high + 0.8*veg_low)

            # Read soil moisture
            sdep1 = 70.0
            idep2 = 3
            # sdep2 = idep2*sdep1

            swwil2 = idep2 * parameters.land_model.swwil
            rsw    = 1.0 / ( parameters.land_model.swcap + idep2*( parameters.land_model.swcap - parameters.land_model.swwil ))

            # Combine soil water content from two top layers
            swl1 = jnp.asarray(xr.open_dataset(surface_filename)["swl1"])
            swl2 = jnp.asarray(xr.open_dataset(surface_filename)["swl2"])
            
            swroot = idep2 * swl2
            # Compute the intermediate max term
            max_term = jnp.maximum(0.0, swroot - swwil2)
            # Compute the soil water content

            soilw_am = jnp.minimum(1.0, rsw * (swl1 + veg[:,:,jnp.newaxis] * max_term))

            # simple sanity check - same method ras above for stl12
            soilw_am = jnp.where((bmask_l[:,:,jnp.newaxis] > 0.0) & ((soilw_am < 0.0) | (soilw_am > 10.0)), 0.0, soilw_am)
            
            
            # =========================================================================
            # Set heat capacities and dissipation times for soil and ice-sheet layers
            # =========================================================================

            # 2. Compute constant fields
            # Set domain mask (blank out sea points)
            dmask = jnp.ones_like(fmask_l)
            dmask = jnp.where(fmask_l < parameters.land_model.flandmin, 0, dmask)

            # Set time_step/heat_capacity and dissipation fields
            cdland = dmask * parameters.land_model.tdland/(1.0+dmask*parameters.land_model.tdland)

            #######################################

        
            # Fractional and binary land masks
            fmask_lnd = boundaries.fmask
            #bmask_lnd = jnp.where(fmask_lnd >= thrsh, 1.0, 0.0)
    
            # Update fmask_lnd based on the conditions
            fmask_lnd = jnp.where(
                fmask_lnd >= thrsh,
                1.0,
                0.0,
            )

            if jnp.any( jnp.isnan(init_stl) == (fmask_lnd == 0) ):
                print("fmask_ocn and init_T do share the same mask.")
            else:
                raise Exception("Warning: fmask_ocn and sst_init do not share the same mask.")
            
            self.fmask_ocn = fmask_ocn
            
        else:
           
             
            init_T   = T_min + (T_max - T_min) * jnp.cos(llat_rad - 20 * jnp.pi / 180.0)**3 + 5.0 * jnp.cos(llon_rad)
        
        
        # Compute cd and time factor
        self.init_state = self.init_state.copy(
            mld = init_mld,
            T = init_T,
        )
        
        
                
    def getInitState(self):
        return dict(state=self.init_state, phydata=self.init_phydata)

    
    def initialize(self):
        ...
    
    def genForwardFunc(
        self,
    ):

        # Find day of the year to locate climatology
        ref_dt = pd.Timestamp(year=self.start_dt.year, month=self.start_dt.month, day=1)
        start_dt_offset = jnp.int_(jnp.floor( ( self.start_dt - ref_dt ) / pd.Timedelta(days=1) ))
        
        @jax.jit
        def forward_func(cpl, t):

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

                somstate.sim_time += self.subtimestep
            
            new_T = new_Tanom + snapshot_SST_clim

            new_state = somstate.copy(
                T = new_T,
            )
            
            new_phydata = cpl["ocn"]["phydata"].copy()
            
            return dict(state=new_state, phydata=new_phydata), stack_objects( [ dict(state=new_state, phydata=new_phydata) , ] )
            
        return forward_func

    def convertTrajectoryToXarray(
        self,
        trajectory,
    ):
        
        """
        A tool function that converts a trajectory into an xarray Dataset.

        Args:
            trajectory : The stacked prediction
            
        Returns:
            ds : The resulting xarray dataset.
        """
        st = trajectory["state"]
        ds = xr.Dataset(
            data_vars = dict(
                sim_time = (["time",], st.sim_time),
                T   = (["time", "lon", "lat"], st.T),
                mld = (["time", "lon", "lat"], st.mld),
            ),
        )
        
        return ds
        
    
    def report(self):
       print("Ocean temperature = ", self.state.T[0]) 


def land_model_init(surface_filename, parameters: Parameters, boundaries: BoundaryData):

    """
        surface_filename: filename storing boundary data
        parameters: initialized model parameters
        boundaries: partially initialized boundary data
        time_step: time step - model timestep in minutes
    """
    import xarray as xr
    # =========================================================================
    # Initialize land-surface boundary conditions
    # =========================================================================

    # Fractional and binary land masks
    fmask_l = boundaries.fmask
    bmask_l = jnp.where(fmask_l >= parameters.land_model.thrsh, 1.0, 0.0)

    # Update fmask_l based on the conditions
    fmask_l = jnp.where(fmask_l >= parameters.land_model.thrsh,
                        jnp.where(boundaries.fmask > (1.0 - parameters.land_model.thrsh), 1.0, fmask_l), 0.0)

    # Compute sea mask as complement of land mask 
    fmask_s = 1.0 - fmask_l

    # Land-surface temperature
    stlcl_ob = jnp.asarray(xr.open_dataset(surface_filename)["stl"])
    # instead of using a check forchk, we check that 0.0 < stl12 < 400 and if it isn't we set it to 273
    # this might not work exactly because stl12 is ix,il,12 and bmask is ix,il
    stlcl_ob = jnp.where((bmask_l[:,:,jnp.newaxis] > 0.0) & ((stlcl_ob < 0.0) | (stlcl_ob > 400.0)), 273.0, stlcl_ob)

    # Snow depth
    snowd_am = jnp.asarray(xr.open_dataset(surface_filename)["snowd"])
    # simple sanity check - same method ras above for stl12
    snowd_am = jnp.where((bmask_l[:,:,jnp.newaxis] > 0.0) & ((snowd_am < 0.0) | (snowd_am > 20000.0)), 0.0, snowd_am)
    # Read soil moisture and compute soil water availability using vegetation fraction
    # Read vegetation fraction
    veg_high = jnp.asarray(xr.open_dataset(surface_filename)["vegh"])
    veg_low  = jnp.asarray(xr.open_dataset(surface_filename)["vegl"])

    # Combine high and low vegetation fractions
    veg = jnp.maximum(0.0, veg_high + 0.8*veg_low)

    # Read soil moisture
    sdep1 = 70.0
    idep2 = 3
    # sdep2 = idep2*sdep1

    swwil2 = idep2*parameters.land_model.swwil
    rsw    = 1.0/(parameters.land_model.swcap + idep2*(parameters.land_model.swcap - parameters.land_model.swwil))

    # Combine soil water content from two top layers
    swl1 = jnp.asarray(xr.open_dataset(surface_filename)["swl1"])
    swl2 = jnp.asarray(xr.open_dataset(surface_filename)["swl2"])
    
    swroot = idep2 * swl2
    # Compute the intermediate max term
    max_term = jnp.maximum(0.0, swroot - swwil2)
    # Compute the soil water content

    soilw_am = jnp.minimum(1.0, rsw * (swl1 + veg[:,:,jnp.newaxis] * max_term))

    # simple sanity check - same method ras above for stl12
    soilw_am = jnp.where((bmask_l[:,:,jnp.newaxis] > 0.0) & ((soilw_am < 0.0) | (soilw_am > 10.0)), 0.0, soilw_am)

    # =========================================================================
    # Set heat capacities and dissipation times for soil and ice-sheet layers
    # =========================================================================

    # 2. Compute constant fields
    # Set domain mask (blank out sea points)
    dmask = jnp.ones_like(fmask_l)
    dmask = jnp.where(fmask_l < parameters.land_model.flandmin, 0, dmask)

    # Set time_step/heat_capacity and dissipation fields
    cdland = dmask*parameters.land_model.tdland/(1.0+dmask*parameters.land_model.tdland)

    return boundaries.copy(cdland=cdland, fmask_l=fmask_l, fmask_s=fmask_s, stlcl_ob=stlcl_ob, snowd_am=snowd_am, soilw_am=soilw_am)

# Exchanges fluxes between land and atmosphere.
def couple_land_atm(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[PhysicsTendency, PhysicsData]:

    day = physics_data.date.model_day()
    stl_lm=None
    # Run the land model if the land model flags is switched on
    if (boundaries.land_coupling_flag):
        # stl_lm need to persist from time step to time step? what does this get from the model?
        stl_lm = run_land_model(physics_data.surface_flux.hfluxn, physics_data.stlcl_lm, boundaries.stlcl_ob[:,:,day], boundaries.cdland, boundaries.rhcapl)
        stl_am = stl_lm
    # Otherwise get the land surface from climatology
    else:
        stl_am = boundaries.stlcl_ob[:,:,day]

    # update land physics data
    land_model_data = physics_data.land_model.copy(stl_am=stl_am, stl_lm=stl_lm)
    physics_data = physics_data.copy(land_model=land_model_data)
    physics_tendency = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendency, physics_data

# Integrates slab land-surface model for one day.
@jit
def run_land_model(hfluxn, stl_lm, stlcl_ob, cdland, rhcapl):
    # Land-surface (soil/ice-sheet) layer
    # Anomaly w.r.t. final-time climatological temperature
    tanom = stl_lm - stlcl_ob

    # Time evolution of temperature anomaly
    tanom = cdland*(tanom + rhcapl*hfluxn[:,:,0])

    # Full surface temperature at final time
    stl_lm = tanom + stlcl_ob

    return stl_lm
