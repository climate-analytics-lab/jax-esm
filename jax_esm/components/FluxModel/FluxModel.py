"""Slab ocean model component."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jax_esm import constants as constants
from jax_esm.components.util import createPhysicsStateClass, createStateDiagClass

from jax_esm.components.util import stack_objects

from jax_esm.components.base import (
    BoundaryFluxes,
    Component,
    ComponentConfig,
    ComponentState,
)

import xarray as xr

class FluxModel(Component):
    """Simple slab ocean model with prescribed mixed layer depth.
    
    This model integrates SST anomalies based on surface heat fluxes
    and relaxes towards a prescribed climatology.
    """

    @classmethod
    def createStateClass(
        cls,
        D2_nodal_shape,
        D3_nodal_shape,
    ):
        
        SOMStateClass = createPhysicsStateClass(
            cls_name = "FMState",
            fields = [
                ("lhflx", float, D2_nodal_shape),
                ("swflx_toa", float, D2_nodal_shape),
                ("swflx_sfc", float, D2_nodal_shape),
                ("lwflx_toa", float, D2_nodal_shape),
                ("hfluxn",    float, D2_nodal_shape + (2,)),
            ],
        )
    
        return SOMStateClass

    @classmethod
    def createDiagClass(
        cls,
        D2_nodal_shape,
        D3_nodal_shape,
        cls_name = "FMDiag",
    ):
        
        new_cls = createPhysicsStateClass(
            cls_name = cls_name,
            fields = [
                ("heatflx", float, D2_nodal_shape),
            ],
        )
    
        return new_cls
        
    def __init__(
        self,
        config: ComponentConfig,    
    ):
        """Initialize slab ocean model.
        
        Expected parameters in config:
        - mixed_layer_depth: Ocean mixed layer depth (m)
        - relaxation_time: Relaxation timescale to climatology (days)
        - sst_clim_file: Optional path to SST climatology
        """
        super().__init__(config)
        
        self.coords = config.params["coords"]

        self.timestep = config.timestep
        self.substeps = config.substeps
        
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        
        
        self.stateClass = self.__class__.createStateClass(
            D2_nodal_shape = D2_nodal_shape,
            D3_nodal_shape = D3_nodal_shape,
        )

        self.diagClass = self.__class__.createDiagClass(
            D2_nodal_shape = D2_nodal_shape,
            D3_nodal_shape = D3_nodal_shape,
        )
        self.stateDiagClass = createStateDiagClass(
            state_cls = self.stateClass,
            diag_cls = self.diagClass,
        )
        self.state_diag = self.stateDiagClass.zeros()

        self.trajectory = []
        
        
        self.stephan_boltzmann_const = constants.stephan_boltzmann_const
        self.solar_const = constants.solar_const
        self.u10 = 5.0 # m/s
        self.C_H = 1e-3
        self.rho_cp = 1.2 * 1004
        self.beta = 0.7
            
    def initialize(self):
        self.trajectory = []

    def record(self, state_diag):
        self.state_diag = state_diag
        self.trajectory.append(state_diag.copy())

    
    def genForwardFunc(self, begin_time):
        
        @jax.jit
        def forward_func(cplstate):

            atmstate_diag = cplstate.atm
            fmstate_diag = cplstate.flx
            fmstate = fmstate_diag.state

            new_hfluxn = - jnp.mean(atmstate_diag.physics.surface_flux.hfluxn, axis=0)
            new_fmstate_diag = fmstate_diag.copy(
                #swflx_toa = new_swflx_toa,
                #swflx_sfc = new_swflx_sfc,
                #lwflx_toa = new_lwflx_toa,
                #lhflx = new_lhflx,
                state_kwargs = dict(hfluxn = new_hfluxn),
            )

            return new_fmstate_diag

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
        trajectory = None,
    ):
        if trajectory is None:
            trajectory = self.trajectory
        
        stacked = stack_objects(trajectory)
        ds = xr.Dataset(
            data_vars = dict(
                hfluxn  = (["time", "lon", "lat", "layer"], stacked.state.hfluxn),
            ),
        )
        
        return ds

    
    def run(self, master=None):
        #print("Flux model run: compute the fluxes")

        #ocn_model = master.components["ocn"]
        #atm_model = master.components["atm"]

        dc = self.data_center
        ocn_T = dc.getVariable(component="ocn", varname="sea_surface_temperature", by_component="flx", is_universal_name = True)
        atm_T = dc.getVariable(component="atm", varname="surface_air_temperature", by_component="flx", is_universal_name = True)


        """
        u10 = 5.0 # m/s
        C_H = 1e-3
        rho_cp = 1.2 * 1004
        
        new_lhflx = (u10 * 1e-3 * rho_cp) * (ocn_T - atm_T)

        # shortwave radiation
        _tmp = - self.solar_const * jnp.sin( 2 * jnp.pi * master.time / 86400.0 )
        _tmp = jnp.where(_tmp > 0, 0, _tmp)

        new_swflx_toa = self.state.swflx_toa * 0 + _tmp
        new_swflx_sfc = new_swflx_toa * 0.80

        new_lwflx_toa = self.state.lwflx_toa * 0 + self.stephan_boltzmann_const * (atm_T ** 4.0)
        """

        

        
        self.state = self.state.copy(
            lhflx = new_lhflx,
            swflx_toa = new_swflx_toa,
            swflx_sfc = new_swflx_sfc,
            lwflx_toa = new_lwflx_toa,
        )

    def report(self):
       print("Latent heat flux = ", self.state.lhflx[0]) 
       print("Top-of-atmosphere shortwave rad flux = ", self.state.swflx_toa[0]) 
       print("Surface shortwave rad flux = ", self.state.swflx_sfc[0]) 