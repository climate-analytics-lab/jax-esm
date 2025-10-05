"""Example of coupling JAX-GCM with a simple slab ocean model."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jax_esm import Component, ComponentConfig, ComponentState, Coupler, BoundaryFluxes


class SimplifiedAtmosphere(Component):
    """Simplified atmosphere component (placeholder for JAX-GCM)."""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.nlat = config.params["nlat"]
        self.nlon = config.params["nlon"]
        self.nlevels = config.params["nlevels"]
    
    def initialize(self, rng_key: jax.random.PRNGKey) -> ComponentState:
        """Initialize atmosphere state."""
        shape_3d = (self.nlevels, self.nlat, self.nlon)
        shape_2d = (self.nlat, self.nlon)
        
        # Initialize with some reasonable values
        return ComponentState(
            prognostic={
                "temperature": jnp.full(shape_3d, 280.0),  # K
                "humidity": jnp.full(shape_3d, 0.01),  # kg/kg
                "u_wind": jnp.zeros(shape_3d),  # m/s
                "v_wind": jnp.zeros(shape_3d),  # m/s
                "surface_pressure": jnp.full(shape_2d, 101325.0),  # Pa
            },
            diagnostic={
                "precipitation": jnp.zeros(shape_2d),  # kg/m²/s
                "cloud_fraction": jnp.zeros(shape_3d),
            },
            boundary={
                "surface_temperature": jnp.full(shape_2d, 288.0),  # K (SST)
                "surface_wind_u": jnp.zeros(shape_2d),
                "surface_wind_v": jnp.zeros(shape_2d),
            },
            forcing={
                "solar_radiation": jnp.full(shape_2d, 340.0),  # W/m²
            },
            metadata={
                "time": 0.0,
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
        """Simplified atmosphere step."""
        # Extract SST from ocean forcing
        sst = state.boundary["surface_temperature"]
        
        # Very simplified physics
        # Temperature tendency based on SST difference
        surf_temp = state.prognostic["temperature"][0]
        temp_tendency = 0.1 * (sst - surf_temp) / dt
        
        # Update surface temperature
        new_temp = state.prognostic["temperature"].at[0].add(temp_tendency * dt)
        
        # Compute output fluxes (simplified)
        heat_flux = -50.0 * (surf_temp - sst)  # W/m²
        evaporation = 1e-5 * jnp.maximum(0, surf_temp - 273.15)  # kg/m²/s
        
        # Wind stress (simplified)
        wind_u = state.prognostic["u_wind"][0]
        wind_v = state.prognostic["v_wind"][0]
        
        output_fluxes = BoundaryFluxes(
            heat=heat_flux,
            moisture=evaporation,
            momentum_u=0.1 * wind_u,
            momentum_v=0.1 * wind_v,
            tracers={},
        )
        
        # Update state
        new_state = ComponentState(
            prognostic={**state.prognostic, "temperature": new_temp},
            diagnostic=state.diagnostic,
            boundary=state.boundary,
            forcing=state.forcing,
            metadata={**state.metadata, "time": state.metadata["time"] + dt},
        )
        
        return new_state, output_fluxes
    
    def compute_tendencies(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
    ) -> Dict[str, Array]:
        """Compute tendencies (placeholder)."""
        return {
            "temperature": jnp.zeros_like(state.prognostic["temperature"]),
            "humidity": jnp.zeros_like(state.prognostic["humidity"]),
            "u_wind": jnp.zeros_like(state.prognostic["u_wind"]),
            "v_wind": jnp.zeros_like(state.prognostic["v_wind"]),
        }


class SlabOcean(Component):
    """Simple slab ocean model."""
    
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.nlat = config.params["nlat"]
        self.nlon = config.params["nlon"]
        self.mixed_layer_depth = config.params.get("mixed_layer_depth", 50.0)  # m
        self.heat_capacity = 4.186e6  # J/m³/K (water)
    
    def initialize(self, rng_key: jax.random.PRNGKey) -> ComponentState:
        """Initialize ocean state."""
        shape_2d = (self.nlat, self.nlon)
        
        # Initialize with zonal temperature gradient
        lat = jnp.linspace(-90, 90, self.nlat)
        sst = 273.15 + 30.0 * jnp.cos(jnp.pi * lat / 180.0)[:, None]
        sst = jnp.broadcast_to(sst, shape_2d)
        
        return ComponentState(
            prognostic={
                "temperature": sst,  # K
                "salinity": jnp.full(shape_2d, 35.0),  # PSU
            },
            diagnostic={
                "mixed_layer_depth": jnp.full(shape_2d, self.mixed_layer_depth),
            },
            boundary={
                "surface_temperature": sst,
            },
            forcing={},
            metadata={
                "time": 0.0,
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
        """Advance ocean state."""
        # Get heat flux from atmosphere
        heat_flux = forcing.heat  # W/m²
        
        # Convert to temperature tendency
        # Q = rho * c_p * h * dT/dt
        # dT/dt = Q / (rho * c_p * h)
        temp_tendency = heat_flux / (self.heat_capacity * self.mixed_layer_depth)
        
        # Update temperature
        new_temp = state.prognostic["temperature"] + temp_tendency * dt
        
        # Output fluxes (ocean provides updated SST to atmosphere)
        output_fluxes = BoundaryFluxes(
            heat=jnp.zeros_like(heat_flux),  # Ocean doesn't output heat flux
            moisture=jnp.zeros_like(heat_flux),
            momentum_u=jnp.zeros_like(heat_flux),
            momentum_v=jnp.zeros_like(heat_flux),
            tracers={},
        )
        
        # Update state
        new_state = ComponentState(
            prognostic={**state.prognostic, "temperature": new_temp},
            diagnostic=state.diagnostic,
            boundary={"surface_temperature": new_temp},
            forcing=state.forcing,
            metadata={**state.metadata, "time": state.metadata["time"] + dt},
        )
        
        return new_state, output_fluxes
    
    def compute_tendencies(
        self,
        state: ComponentState,
        forcing: BoundaryFluxes,
    ) -> Dict[str, Array]:
        """Compute ocean tendencies."""
        heat_flux = forcing.heat
        temp_tendency = heat_flux / (self.heat_capacity * self.mixed_layer_depth)
        
        return {
            "temperature": temp_tendency,
            "salinity": jnp.zeros_like(state.prognostic["salinity"]),
        }
    
    def get_required_fluxes(self) -> list[str]:
        """Ocean requires heat flux from atmosphere."""
        return ["heat", "moisture", "momentum_u", "momentum_v"]


def main():
    """Run example coupled atmosphere-ocean simulation."""
    # Configuration
    nlat, nlon = 32, 64
    
    # Create atmosphere component (placeholder for JAX-GCM)
    atm_config = ComponentConfig(
        name="atmosphere",
        timestep=1800.0,  # 30 minutes
        grid={"type": "latlon", "nlat": nlat, "nlon": nlon},
        params={"nlat": nlat, "nlon": nlon, "nlevels": 10},
    )
    atmosphere = SimplifiedAtmosphere(atm_config)
    
    # Create ocean component
    ocean_config = ComponentConfig(
        name="ocean",
        timestep=3600.0,  # 1 hour
        grid={"type": "latlon", "nlat": nlat, "nlon": nlon},
        params={"nlat": nlat, "nlon": nlon, "mixed_layer_depth": 50.0},
    )
    ocean = SlabOcean(ocean_config)
    
    # Create coupler
    components = {
        "atmosphere": atmosphere,
        "ocean": ocean,
    }
    
    # Define custom flux mappings
    # Ocean SST affects atmospheric boundary conditions
    flux_mappings = {
        ("ocean", "atmosphere"): {
            "surface_temperature": "surface_temperature",
        },
        ("atmosphere", "ocean"): {
            "heat": "heat",
            "moisture": "moisture",
            "momentum_u": "momentum_u",
            "momentum_v": "momentum_v",
        },
    }
    
    coupler = Coupler(
        components=components,
        coupling_timestep=3600.0,  # 1 hour
        flux_mappings=flux_mappings,
    )
    
    # Initialize
    rng_key = jax.random.PRNGKey(42)
    initial_states = coupler.initialize(rng_key)
    
    # Run for 10 days
    start_time = 0.0
    end_time = 10 * 24 * 3600.0  # 10 days in seconds
    
    print("Running coupled atmosphere-ocean simulation...")
    print(f"Simulation period: {end_time / 86400:.1f} days")
    print(f"Coupling timestep: {coupler.coupling_timestep / 3600:.1f} hours")
    
    # Run simulation
    component_results = coupler.run(
        initial_states=initial_states,
        start_time=start_time,
        end_time=end_time,
        save_frequency=24,  # Save every 24 hours
    )
    
    # Print results
    print(f"\nSimulation complete!")
    print(f"Number of saved states: {len(history)}")
    
    # Print final SST statistics
    final_sst = component_results["ocean"][0].prognostic["temperature"]
    print(f"\nFinal SST statistics:")
    print(f"  Mean: {jnp.mean(final_sst) - 273.15:.2f}°C")
    print(f"  Min:  {jnp.min(final_sst) - 273.15:.2f}°C")
    print(f"  Max:  {jnp.max(final_sst) - 273.15:.2f}°C")
    
    # Print final atmosphere surface temperature
    final_atm_temp = component_results["atmosphere"][0].prognostic["temperature"]
    print(f"\nFinal atmosphere surface temperature:")
    print(f"  Mean: {jnp.mean(final_atm_temp) - 273.15:.2f}°C")
    print(f"  Min:  {jnp.min(final_atm_temp) - 273.15:.2f}°C")
    print(f"  Max:  {jnp.max(final_atm_temp) - 273.15:.2f}°C")


if __name__ == "__main__":
    main()