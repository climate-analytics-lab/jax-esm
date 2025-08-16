"""Wrapper to integrate JAX-GCM with JAX-ESM coupling framework."""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jax_esm.components.base import (
    BoundaryFluxes,
    Component,
    ComponentConfig,
    ComponentState,
)


class JaxGcmWrapper(Component):
    """Wrapper for JAX-GCM atmospheric model.
    
    This wrapper adapts JAX-GCM to work with the JAX-ESM coupling framework,
    handling state conversions and flux calculations.
    """
    
    def __init__(self, config: ComponentConfig, gcm_model=None):
        """Initialize JAX-GCM wrapper.
        
        Args:
            config: Component configuration
            gcm_model: Pre-configured JAX-GCM model instance (optional)
        """
        super().__init__(config)
        
        # Store GCM model instance
        self.gcm = gcm_model
        
        # Grid information
        self.nlat = config.grid["nlat"]
        self.nlon = config.grid["nlon"]
        self.nlevels = config.params.get("nlevels", 20)
        
        # If no model provided, we'll need to initialize one
        # This is a placeholder - actual implementation would import and configure JAX-GCM
        if self.gcm is None:
            self._initialize_gcm(config)
    
    def _initialize_gcm(self, config: ComponentConfig):
        """Initialize JAX-GCM model (placeholder for actual implementation)."""
        # In a real implementation, this would:
        # 1. Import JAX-GCM modules
        # 2. Set up model configuration
        # 3. Initialize the GCM instance
        pass
    
    def initialize(self, rng_key: jax.random.PRNGKey) -> ComponentState:
        """Initialize atmospheric state from JAX-GCM.
        
        Args:
            rng_key: JAX random key for initialization
            
        Returns:
            Initial component state
        """
        # In a real implementation, this would call JAX-GCM's initialization
        # For now, create a reasonable initial state
        shape_3d = (self.nlevels, self.nlat, self.nlon)
        shape_2d = (self.nlat, self.nlon)
        
        # Initialize with typical atmospheric values
        # Temperature decreases with height
        temp_profile = jnp.linspace(288, 220, self.nlevels)[:, None, None]
        temperature = jnp.broadcast_to(temp_profile, shape_3d)
        
        # Add latitudinal variation
        lat = jnp.linspace(-90, 90, self.nlat)
        lat_factor = 0.7 + 0.3 * jnp.cos(jnp.pi * lat / 180.0)
        temperature = temperature * lat_factor[None, :, None]
        
        return ComponentState(
            prognostic={
                "temperature": temperature,  # K
                "specific_humidity": jnp.full(shape_3d, 0.005),  # kg/kg
                "u_wind": jnp.zeros(shape_3d),  # m/s
                "v_wind": jnp.zeros(shape_3d),  # m/s
                "surface_pressure": jnp.full(shape_2d, 101325.0),  # Pa
            },
            diagnostic={
                "precipitation": jnp.zeros(shape_2d),  # mm/day
                "surface_temperature": temperature[0],  # K
                "sensible_heat_flux": jnp.zeros(shape_2d),  # W/m²
                "latent_heat_flux": jnp.zeros(shape_2d),  # W/m²
                "longwave_down": jnp.zeros(shape_2d),  # W/m²
                "shortwave_down": jnp.zeros(shape_2d),  # W/m²
            },
            boundary={
                "surface_temperature": jnp.full(shape_2d, 288.0),  # SST from ocean
                "surface_wind_u": jnp.zeros(shape_2d),
                "surface_wind_v": jnp.zeros(shape_2d),
            },
            forcing={
                "solar_insolation": jnp.full(shape_2d, 340.0),  # W/m²
                "co2_concentration": 420.0,  # ppm
            },
            metadata={
                "time": 0.0,
                "lat": jnp.linspace(-90, 90, self.nlat),
                "lon": jnp.linspace(0, 360, self.nlon),
                "levels": jnp.arange(self.nlevels),
            },
        )
    
    def step(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
        dt: float,
    ) -> Tuple[ComponentState, BoundaryFluxes]:
        """Advance atmospheric model by one timestep.
        
        Args:
            state: Current atmospheric state
            forcing: Boundary fluxes from other components
            dt: Time step in seconds
            
        Returns:
            Tuple of (new_state, output_fluxes)
        """
        # Update boundary conditions from ocean/land
        # In JAX-GCM, SST would be provided through the ocean component
        if hasattr(forcing, "surface_temperature"):
            # This would come from ocean component
            sst = forcing.surface_temperature
        else:
            sst = state.boundary["surface_temperature"]
        
        # In a real implementation, this would:
        # 1. Convert ComponentState to JAX-GCM state format
        # 2. Call JAX-GCM step function
        # 3. Convert back to ComponentState
        # 4. Calculate surface fluxes
        
        # For demonstration, do simple physics
        # Surface fluxes based on bulk formulas
        t_atm = state.prognostic["temperature"][0]  # Surface air temperature
        
        # Sensible heat flux (simplified)
        ch = 0.001  # Heat transfer coefficient
        rho = 1.2   # Air density kg/m³
        cp = 1005.0  # Specific heat J/kg/K
        wind_speed = jnp.sqrt(
            state.prognostic["u_wind"][0]**2 + state.prognostic["v_wind"][0]**2
        ) + 2.0  # Add minimum wind
        
        sensible_heat = rho * cp * ch * wind_speed * (sst - t_atm)
        
        # Latent heat flux (simplified)
        ce = 0.001  # Moisture transfer coefficient
        lv = 2.5e6  # Latent heat of vaporization J/kg
        q_sat_sst = 0.98 * 0.622 * jnp.exp(17.67 * (sst - 273.15) / (sst - 29.65)) / 1013.25
        q_atm = state.prognostic["specific_humidity"][0]
        
        latent_heat = rho * lv * ce * wind_speed * (q_sat_sst - q_atm)
        
        # Net longwave (simplified Stefan-Boltzmann with atmosphere)
        sigma = 5.67e-8
        emissivity = 0.97
        net_longwave = emissivity * sigma * (sst**4 - t_atm**4)
        
        # Total heat flux into atmosphere (positive upward)
        total_heat_flux = sensible_heat + latent_heat + net_longwave
        
        # Update atmospheric temperature (very simplified)
        heating_rate = total_heat_flux / (rho * cp * 1000.0)  # K/s (1km layer)
        new_temp = state.prognostic["temperature"].at[0].add(heating_rate * dt)
        
        # Create new state
        new_state = ComponentState(
            prognostic={
                **state.prognostic,
                "temperature": new_temp,
            },
            diagnostic={
                **state.diagnostic,
                "sensible_heat_flux": sensible_heat,
                "latent_heat_flux": latent_heat,
                "surface_temperature": new_temp[0],
            },
            boundary={
                **state.boundary,
                "surface_temperature": sst,  # Update SST from ocean
            },
            forcing=state.forcing,
            metadata={
                **state.metadata,
                "time": state.metadata["time"] + dt,
            },
        )
        
        # Output fluxes to ocean/land
        # Note: Sign convention - positive flux is downward (into ocean)
        output_fluxes = BoundaryFluxes(
            heat=-total_heat_flux,  # Negative because ocean convention is opposite
            moisture=latent_heat / lv,  # Convert to moisture flux kg/m²/s
            momentum_u=0.1 * state.prognostic["u_wind"][0],  # Simplified wind stress
            momentum_v=0.1 * state.prognostic["v_wind"][0],
            tracers={},
        )
        
        return new_state, output_fluxes
    
    def compute_tendencies(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
    ) -> Dict[str, Array]:
        """Compute tendencies for prognostic variables.
        
        In a real implementation, this would call JAX-GCM's tendency calculations.
        """
        # Placeholder tendencies
        shape_3d = state.prognostic["temperature"].shape
        
        return {
            "temperature": jnp.zeros(shape_3d),
            "specific_humidity": jnp.zeros(shape_3d),
            "u_wind": jnp.zeros(shape_3d),
            "v_wind": jnp.zeros(shape_3d),
            "surface_pressure": jnp.zeros((self.nlat, self.nlon)),
        }
    
    def get_boundary_fields(self, state: ComponentState) -> Dict[str, Array]:
        """Extract boundary fields needed by other components."""
        # Atmosphere provides surface fields to ocean/land
        return {
            "surface_temperature": state.diagnostic["surface_temperature"],
            "surface_wind_u": state.prognostic["u_wind"][0],
            "surface_wind_v": state.prognostic["v_wind"][0],
            "surface_pressure": state.prognostic["surface_pressure"],
            "precipitation": state.diagnostic.get("precipitation", jnp.zeros((self.nlat, self.nlon))),
        }
    
    def get_required_fluxes(self) -> list[str]:
        """Atmosphere can use surface temperature from ocean/land."""
        # Note: atmosphere doesn't strictly require fluxes, but benefits from SST updates
        return []
    
    def get_provided_fluxes(self) -> list[str]:
        """Atmosphere provides heat, moisture, and momentum fluxes."""
        return ["heat", "moisture", "momentum_u", "momentum_v"]