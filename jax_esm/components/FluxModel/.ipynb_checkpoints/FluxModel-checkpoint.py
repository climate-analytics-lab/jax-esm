"""Slab ocean model component."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jax_esm.components.base import (
    BoundaryFluxes,
    Component,
    ComponentConfig,
    ComponentState,
)

from jax_esm.components.PhysicsState import PhysicsState
import tree_math

@tree_math.struct
class FMState(PhysicsState):
    lhflx     : jnp.ndarray
    swflx_toa : jnp.ndarray
    swflx_sfc : jnp.ndarray
    lwflx_toa : jnp.ndarray


    @classmethod
    def zeros(
        self,
        lhflx = None,
        swflx_toa = None,
        swflx_sfc = None,
        lwflx_toa = None,
    ):
        return FMState(
            lhflx     = jnp.zeros((1),) if lhflx is None else self.lhflx,
            swflx_toa = jnp.zeros((1),) if swflx_toa is None else self.swflx_toa,
            swflx_sfc = jnp.zeros((1),) if swflx_sfc is None else self.swflx_sfc,
            lwflx_toa = jnp.zeros((1),) if lwflx_toa is None else self.lwflx_toa,
        )


    def copy(
        self,
        lhflx = None,
        swflx_toa = None,
        swflx_sfc = None,
        lwflx_toa = None,
    ):
        return FMState(
            self.lhflx if lhflx is None else lhflx,
            self.swflx_toa if swflx_toa is None else swflx_toa,
            self.swflx_sfc if swflx_sfc is None else swflx_sfc,
            self.lwflx_toa if lwflx_toa is None else lwflx_toa,
        )


class FluxModel(Component):
    """Simple slab ocean model with prescribed mixed layer depth.
    
    This model integrates SST anomalies based on surface heat fluxes
    and relaxes towards a prescribed climatology.
    """
    
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

        self.state = FMState.zeros()
        self.stephan_boltzmann_const = 5.67e-8
        self.solar_constant = 1367.0

    def run(self, master=None):
        #print("Flux model run: compute the fluxes")

        ocn_model = master.components["ocn"]
        atm_model = master.components["atm"]
        
        u10 = 5.0 # m/s
        C_H = 1e-3
        rho_cp = 1.2 * 1004
        new_lhflx = (u10 * 1e-3 * rho_cp) * (ocn_model.state.T - atm_model.state.T)

        # shortwave radiation
        _tmp = - self.solar_constant * jnp.sin( 2 * jnp.pi * master.time / 86400.0 )
        _tmp = jnp.where(_tmp > 0, 0, _tmp)

        new_swflx_toa = self.state.swflx_toa * 0 + _tmp
        new_swflx_sfc = new_swflx_toa * 0.80

        new_lwflx_toa = self.state.lwflx_toa * 0 + self.stephan_boltzmann_const * (atm_model.state.T ** 4.0)
        
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
    
    def update_climatology(
        self,
        state: ComponentState,
        sst_clim: Array,
        #hflux_clim: Optional[Array] = None,
    ) -> ComponentState:
        """Update climatological fields (e.g., for seasonal cycle).
        
        Args:
            state: Current ocean state
            sst_clim: New SST climatology
            hflux_clim: Optional new heat flux climatology
            
        Returns:
            Updated state with new climatology
        """
        new_forcing = dict(state.forcing)
        new_forcing["sst_clim"] = new_forcing["sst_clim_next"]
        new_forcing["sst_clim_next"] = sst_clim
        
        #if hflux_clim is not None:
        #    new_forcing["hflux_clim"] = hflux_clim
        
        return ComponentState(
            prognostic=state.prognostic,
            diagnostic=state.diagnostic,
            boundary=state.boundary,
            forcing=new_forcing,
            metadata=state.metadata,
        )
    
    def get_required_fluxes(self) -> list[str]:
        """Ocean requires heat flux from atmosphere."""
        return ["heat"]
    
    def get_provided_fluxes(self) -> list[str]:
        """Ocean provides SST through boundary conditions."""
        return []