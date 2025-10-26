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

    A flux model with the simpliest implementation.

    This flux model computes the total heat flux from
    the atmosphere model. The resulting heat fluxes are 
    positive upward.

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
                ],
            ),
        )

        if "boundaries" in config.params and config.params["boundaries"] is not None:

            boundaries = config.params["boundaries"]
            thrsh = 0.3

            SST_clim = jnp.array(xr.open_dataset(config.params["boundary_file"])["sst"])
            
            # Update fmask_lnd based on the conditions
            fmask_lnd = jnp.where(
                boundaries.fmask >= thrsh,
                1.0,
                0.0,
            )

            fmask_ocn = 1.0 - fmask_lnd


        self.first_call = True
        
 
    def initialize(self):
        return self.component_state_class.zeros()
            
    def generate_step_function(self):

        @jax.jit
        def step_function(cplstate, t):

            atm_phydata = cplstate.atm.phydata

            # In this flux model, we simply get the flux from atmosphere's calculation
            new_flx_state = cplstate.flx.copy(
                prog_kwargs = dict(
                    sim_time = cplstate.flx.prog.sim_time + self.config.timestep,
                ),
                phydata_kwargs = dict(
                    heatflx = - atm_phydata.surface_flux.hfluxn.sum(axis=-1),
                ),
            )
           

            return new_flx_state, stack_objects( [ dict(prog=new_flx_state.prog, phydata=new_flx_state.phydata) , ] )

        return step_function
        
    def predictions_to_xarray(
        self,
        predictions,
    ):
        
        """
        A tool function that converts a trajectory into an xarray Dataset.

        Args:
            predictions : The predictions returned from `step_function`
            
        Returns:
            ds : The resulting xarray dataset.
        """
        
        phydata = predictions["phydata"]
        prog    = predictions["prog"]
        ds = xr.Dataset(
            data_vars = dict(
                heatflx  = (["time", "lon", "lat",], phydata.heatflx),
            ),
             coords = dict(
                time = (["time",], prog.sim_time),
            ),
        )
        
        return ds

