"""Flux model component."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import tree_math
from dataclasses import make_dataclass

from jax_esm import constants as constants

from jax_esm.utils.bulk_op import stack_objects

from jax_esm.components.base import (
    Component,
    ComponentConfig,
    create_component_state_class,
    create_field_group_class,
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

        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
 
        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                ],
            ),

            phydata_cls =  create_field_group_class(
                cls_name = "phydata",
                fields = [
                    ("heatflx", float, D2_nodal_shape),
                    ("lhflx", float, D2_nodal_shape),
                    ("swflx_toa", float, D2_nodal_shape),
                    ("swflx_sfc", float, D2_nodal_shape),
                    ("lwflx_toa", float, D2_nodal_shape),
                    ("hfluxn",    float, D2_nodal_shape + (2,)),
                ],
            ),
        )

       
        
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
        
 
    def initialize(self):
        return self.component_state_class.zeros()
            
    def gen_step_fn(self):

        @jax.jit
        def step_fn(cplstate, t):

            atm_phydata = cplstate.atm.phydata

            # In this flux model, we simply get the flux from atmosphere's calculation
            new_flx_state = cplstate.flx.copy(
                prog_kwargs = dict(
                    sim_time = cplstate.flx.prog.sim_time + self.config.timestep,
                ),
                phydata_kwargs = dict(
                    hfluxn = - atm_phydata.surface_flux.hfluxn,
                ),
            )
           

            return new_flx_state, stack_objects( [ dict(prog=new_flx_state.prog, phydata=new_flx_state.phydata) , ] )

        return step_fn
        
    def predictions_to_xarray(
        self,
        predictions,
    ):
        
        """
        A tool function that converts a trajectory into an xarray Dataset.

        Args:
            predictions : The predictions returned from `step_fn`
            
        Returns:
            ds : The resulting xarray dataset.
        """
        
        phydata = predictions["phydata"]
        prog    = predictions["prog"]
        ds = xr.Dataset(
            data_vars = dict(
                hfluxn  = (["time", "lon", "lat", "layer"], phydata.hfluxn),
            ),
             coords = dict(
                time = (["time",], prog.sim_time),
            ),
        )
        
        return ds

