"""Flux model component."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import tree_math
from dataclasses import make_dataclass

from jax_esm import constants as constants

from jax_esm.utils.meta_prog_class import createFieldsClass
from jax_esm.utils.bulk_op import stack_objects


from jax_esm.components.base import (
    Component,
    ComponentConfig,
)

import xarray as xr

class FluxModel(Component):
    """
    FSimple slab ocean model with prescribed mixed layer depth.
    
    This model integrates SST anomalies based on surface heat fluxes
    and relaxes towards a prescribed climatology.
    """

    def __init__(
        self,
        config: ComponentConfig,    
    ):
        """
        Initialize flux model.        
        """
        super().__init__(config)
        
        self.coords = config.params["coords"]

        self.timestep = config.timestep
        self.substeps = config.substeps
        
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        
        
        self.stateClass = createFieldsClass(
            cls_name = "FM_state",
            fields = [
                ("lhflx", float, D2_nodal_shape),
                ("swflx_toa", float, D2_nodal_shape),
                ("swflx_sfc", float, D2_nodal_shape),
                ("lwflx_toa", float, D2_nodal_shape),
                ("hfluxn",    float, D2_nodal_shape + (2,)),
            ],
        )

        self.phydataClass = createFieldsClass(
            cls_name = "FM_diag",
            fields = [
                ("heatflx", float, D2_nodal_shape),
            ],
        )
        
        self.init_state = self.stateClass.zeros()
        self.init_phydata = self.phydataClass.zeros()

        self.stephan_boltzmann_const = constants.stephan_boltzmann_const
        self.solar_const = constants.solar_const
        self.u10 = 5.0 # m/s
        self.C_H = 1e-3
        self.rho_cp = 1.2 * 1004
        self.beta = 0.7

        if "boundaries" in config.params and config.params["boundaries"] is not None:

            boundaries = config.params["boundaries"]
            thrsh = 0.3

            SST_clim = jnp.array(xr.open_dataset(config.params["boundary_file"])["sst"])
            
            # Fractional and binary land masks
            fmask_lnd = boundaries.fmask
            #bmask_lnd = jnp.where(fmask_lnd >= thrsh, 1.0, 0.0)
    
            # Update fmask_lnd based on the conditions
            fmask_lnd = jnp.where(
                fmask_lnd >= thrsh,
                1.0,
                0.0,
            )

            fmask_ocn = 1.0 - fmask_lnd


        self.first_call = True
        
    def getInitState(self):
        return dict(state=self.init_state, phydata=self.init_phydata)
            
    def initialize(self):
        ...
        
    def genForwardFunc(self):

        @jax.jit
        def forward_func(cplstate, t):

            atm_phydata = cplstate["atm"]["phydata"]

            
            # In this flux model, we simply get the flux from atmosphere's calculation
            new_hfluxn = atm_phydata.surface_flux.hfluxn * (-1)
            
            new_state = cplstate["flx"]["state"].copy(
                #swflx_toa = new_swflx_toa,
                #swflx_sfc = new_swflx_sfc,
                #lwflx_toa = new_lwflx_toa,
                #lhflx = new_lhflx,
                hfluxn = new_hfluxn,
            )
            
            new_phydata = cplstate["flx"]["phydata"].copy()

            return dict(state=new_state, phydata=new_phydata), stack_objects( [ dict(state=new_state, phydata=new_phydata) , ] )

        return forward_func
        
    def genForwardFunc_SLAB(self):
        
        @jax.jit
        def forward_func(cplstate):

            atmstate = cplstate.atm
            fmstate  = cplstate.flx
            ocnstate = cplstate.ocn
            
            ocn_T = ocnstate.T
            atm_T = atmstate.T

            new_lhflx = (self.u10 * self.C_H * self.rho_cp) * (ocn_T - atm_T)
    
            # shortwave radiation
            _tmp = - self.solar_const / 4
            
            new_swflx_toa = fmstate.swflx_toa * 0 + _tmp
            new_swflx_sfc = new_swflx_toa * self.beta
    
            new_lwflx_toa = fmstate.lwflx_toa * 0 + self.stephan_boltzmann_const * (atm_T ** 4.0)

            new_fmstate = fmstate.copy(
                swflx_toa = new_swflx_toa,
                swflx_sfc = new_swflx_sfc,
                lwflx_toa = new_lwflx_toa,
                lhflx = new_lhflx,
            )
            
            return new_fmstate

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
                hfluxn  = (["time", "lon", "lat", "layer"], st.hfluxn),
            ),
        )
        
        return ds

    def report(self):
       print("Latent heat flux = ", self.state.lhflx[0]) 
       print("Top-of-atmosphere shortwave rad flux = ", self.state.swflx_toa[0]) 
       print("Surface shortwave rad flux = ", self.state.swflx_sfc[0]) 
