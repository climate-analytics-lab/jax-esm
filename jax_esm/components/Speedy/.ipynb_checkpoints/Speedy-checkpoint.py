"""Speedy Wrapper Class"""

from typing import Dict, Tuple

import jcm
from jcm.model import Model, get_coords
from jcm.boundaries import initialize_boundaries

import jax
import jax.numpy as jnp
from jax import Array
from jax_esm import constants as constants
from jax_esm.components.base import (
    BoundaryFluxes,
    Component,
    ComponentConfig,
    ComponentState,
)
from jax_esm.components.PhysicsState import CreatePhysicsStateClass

   
class Speedy(Component):
    """Simple slab ocean model with prescribed mixed layer depth.
    
    This model integrates SST anomalies based on surface heat fluxes
    and relaxes towards a prescribed climatology.
    """

    @classmethod
    def createSpeedyStateClass(
        cls,
        D2_nodal_shape,
        D3_nodal_shape,
    ):
        
        SpeedyState = CreatePhysicsStateClass(
            cls_name = "SpeedyState",
            fields = [
                ("u_wind", float, D3_nodal_shape),
                ("v_wind", float, D3_nodal_shape),
                ("temperature", float, D3_nodal_shape),
                ("specific_humidity", float, D3_nodal_shape),
                ("geopotential", float, D3_nodal_shape),
                ("normalized_surface_pressure", float, D3_nodal_shape),
            ],
        )
    
        return SpeedyState
    
    def __init__(
        self,
        config_speedy: Dict,
        config: ComponentConfig,
    ):

        super().__init__(config)

        self.model = Model(**config_speedy)
        
        D3_nodal_shape = self.model.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        self.stateClass = Speedy.createSpeedyStateClass(
            D2_nodal_shape = D2_nodal_shape,
            D3_nodal_shape = D3_nodal_shape,
        )
        
        self.state = self.stateClass.zeros()


    def run(self, master=None):

        if False:
            #print("Slab atmosphere run.")
    
            dc = self.data_center
            lwflx_toa = dc.getVariable(component="flx", varname="lwflx_toa", by_component="atm")
            swflx_toa = dc.getVariable(component="flx", varname="swflx_toa", by_component="atm")
            swflx_sfc = dc.getVariable(component="flx", varname="swflx_sfc", by_component="atm")
            lhflx     = dc.getVariable(component="flx", varname="lhflx", by_component="atm")
            
            #flux_model = master.components["flx"]
            time_step = master.config["time_step"]
            
            new_T = self.state.T + time_step *( - (lwflx_toa + swflx_toa - swflx_sfc - lhflx) / ( self.column_mass * self.cp ) )       
            self.state = self.state.copy(
                T = new_T,
            )

    def report(self):
        pass
        #print("Atmoshpere temperature = ", self.state.T[0]) 
        
    def initialize(self, rng_key: jax.random.PRNGKey) -> ComponentState:
        """Initialize ocean state with climatological SST."""
        shape = (self.nlat, self.nlon)
        
        # Initialize with simple zonal climatology
        lat = jnp.linspace(-90, 90, self.nlat)
        sst_clim = 273.15 + 30.0 * jnp.cos(jnp.pi * lat / 180.0)
        sst_clim = sst_clim[:, None]  # Add longitude dimension
        sst_clim = jnp.broadcast_to(sst_clim, shape)
        
        # Start with no anomaly
        sst = sst_clim
        
        return ComponentState(
            prognostic={
                "sst": sst,  # Sea surface temperature (K)
                "sst_anomaly": jnp.zeros(shape),  # SST anomaly (K)
            },
            diagnostic={
                "mixed_layer_depth": jnp.full(shape, self.mixed_layer_depth),
                "heat_flux_anomaly": jnp.zeros(shape),  # W/m²
            },
            boundary={
                "surface_temperature": sst,
            },
            forcing={
                "sst_clim": sst_clim,  # Current climatology
                "sst_clim_next": sst_clim,  # Next timestep climatology
                "hflux_clim": jnp.zeros(shape),  # Climatological heat flux
            },
            metadata={
                "time": 0.0,
                "day_of_year": 0,
                "lat": jnp.linspace(-90, 90, self.nlat),
                "lon": jnp.linspace(0, 360, self.nlon),
            },
        )
    
    def step(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
        dt: float,
    ) -> Tuple[ComponentState, BoundaryFluxes]:
        """Advance slab ocean model by one timestep.
        
        Args:
            state: Current ocean state
            forcing: Surface fluxes from atmosphere
            dt: Time step in seconds
            
        Returns:
            Tuple of (new_state, output_fluxes)
        """
        # Extract current state
        sst = state.prognostic["sst"]
        sst_clim_current = state.forcing["sst_clim"]
        sst_clim_next = state.forcing["sst_clim_next"]
        hflux_clim = state.forcing["hflux_clim"]
        
        # Get heat flux from atmosphere (W/m²)
        # Positive flux means heat into ocean
        hflux = forcing.heat
        
        # Calculate anomalies
        sst_anom = sst - sst_clim_current
        hflux_anom = hflux - hflux_clim
        
        # Time integration factor for this timestep
        dt_days = dt / 86400.0  # Convert seconds to days
        time_factor = jnp.power(self.time_factor_per_day, dt_days)
        
        # Integrate SST anomaly
        # d(SST_anom)/dt = -SST_anom/tau + Q_anom/(rho*cp*h)
        # Using exponential time factor for relaxation
        new_sst_anom = time_factor * (sst_anom + hflux_anom * self.cd_factor * dt)
        
        # Add anomaly to next climatology
        new_sst = new_sst_anom + sst_clim_next
        
        # Update state
        new_state = ComponentState(
            prognostic={
                "sst": new_sst,
                "sst_anomaly": new_sst_anom,
            },
            diagnostic={
                "mixed_layer_depth": state.diagnostic["mixed_layer_depth"],
                "heat_flux_anomaly": hflux_anom,
            },
            boundary={
                "surface_temperature": new_sst,
            },
            forcing=state.forcing,
            metadata={
                **state.metadata,
                "time": state.metadata["time"] + dt,
                "day_of_year": (state.metadata["day_of_year"] + dt_days) % 365,
            },
        )
        
        # Ocean doesn't directly output fluxes, but provides SST through boundary
        output_fluxes = BoundaryFluxes(
            heat=jnp.zeros_like(new_sst),
            moisture=jnp.zeros_like(new_sst),
            momentum_u=jnp.zeros_like(new_sst),
            momentum_v=jnp.zeros_like(new_sst),
            tracers={},
        )
        
        return new_state, output_fluxes
    
    def compute_tendencies(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
    ) -> Dict[str, Array]:
        """Compute tendencies for prognostic variables."""
        # Extract state
        sst = state.prognostic["sst"]
        sst_clim = state.forcing["sst_clim"]
        hflux_clim = state.forcing["hflux_clim"]
        
        # Get heat flux
        hflux = forcing.heat
        
        # Calculate anomalies
        sst_anom = sst - sst_clim
        hflux_anom = hflux - hflux_clim
        
        # SST tendency from heat flux and relaxation
        # d(SST)/dt = -SST_anom/tau + Q/(rho*cp*h)
        relaxation_tendency = -sst_anom / (self.relaxation_time * 86400.0)
        heating_tendency = hflux / self.heat_capacity
        sst_tendency = relaxation_tendency + heating_tendency
        
        # Anomaly tendency
        anom_tendency = relaxation_tendency + hflux_anom / self.heat_capacity
        
        return {
            "sst": sst_tendency,
            "sst_anomaly": anom_tendency,
        }    