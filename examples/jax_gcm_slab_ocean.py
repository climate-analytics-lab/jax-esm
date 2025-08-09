"""Example coupling JAX-GCM with slab ocean model.

This example demonstrates how to couple JAX-GCM (through a wrapper)
with the slab ocean model using the JAX-ESM framework.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_esm import (
    ComponentConfig,
    Coupler,
    SlabOceanModel,
)
from jax_esm.components.jax_gcm_wrapper import JaxGcmWrapper


def create_seasonal_sst_climatology(nlat: int, nlon: int, day: float) -> jax.Array:
    """Create a simple seasonal SST climatology.
    
    Args:
        nlat: Number of latitude points
        nlon: Number of longitude points
        day: Day of year (0-365)
        
    Returns:
        SST climatology in Kelvin
    """
    lat = jnp.linspace(-90, 90, nlat)
    
    # Base temperature with latitude dependence
    base_sst = 273.15 + 30.0 * jnp.cos(jnp.pi * lat / 180.0)
    
    # Add seasonal cycle (stronger in mid-latitudes)
    seasonal_amplitude = 5.0 * jnp.sin(jnp.pi * jnp.abs(lat) / 90.0)**2
    seasonal_phase = 2 * jnp.pi * (day - 80) / 365.0  # Peak in late March
    seasonal_cycle = seasonal_amplitude * jnp.cos(seasonal_phase)
    
    sst_clim = base_sst + seasonal_cycle
    sst_clim = sst_clim[:, None]  # Add longitude dimension
    sst_clim = jnp.broadcast_to(sst_clim, (nlat, nlon))
    
    return sst_clim


def main():
    """Run coupled JAX-GCM and slab ocean simulation."""
    
    # Configuration
    nlat, nlon = 64, 128
    nlevels = 20
    
    print("Setting up coupled JAX-GCM and slab ocean model...")
    print(f"Grid: {nlat} x {nlon} x {nlevels} levels")
    
    # Create atmosphere component (JAX-GCM wrapper)
    atm_config = ComponentConfig(
        name="atmosphere",
        timestep=1800.0,  # 30 minutes
        grid={
            "type": "gaussian",  # JAX-GCM typically uses Gaussian grids
            "nlat": nlat,
            "nlon": nlon,
        },
        params={
            "nlevels": nlevels,
            "physics_timestep": 900.0,  # 15 minutes for physics
            "dynamics_timestep": 300.0,  # 5 minutes for dynamics
        },
    )
    
    # In a real implementation, we would pass an actual JAX-GCM instance
    # For this example, we use the wrapper with simplified physics
    atmosphere = JaxGcmWrapper(atm_config, gcm_model=None)
    
    # Create ocean component
    ocean_config = ComponentConfig(
        name="ocean",
        timestep=3600.0,  # 1 hour
        grid={
            "type": "latlon",
            "nlat": nlat,
            "nlon": nlon,
        },
        params={
            "mixed_layer_depth": 50.0,  # 50m mixed layer
            "relaxation_time": 30.0,     # 30-day relaxation to climatology
        },
    )
    ocean = SlabOceanModel(ocean_config)
    
    # Set up flux mappings
    # Ocean surface temperature affects atmospheric boundary layer
    # Atmosphere provides heat/moisture/momentum fluxes to ocean
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
    
    # Custom transformation for ocean to atmosphere
    # The ocean component provides SST through boundary fields,
    # so we need a special handling in the flux exchanger
    def ocean_to_atm_sst(ocean_state):
        """Extract SST from ocean state for atmosphere."""
        return ocean_state.boundary["surface_temperature"]
    
    # Create coupler
    coupler = Coupler(
        components={
            "atmosphere": atmosphere,
            "ocean": ocean,
        },
        coupling_timestep=3600.0,  # 1 hour coupling
        flux_mappings=flux_mappings,
    )
    
    # Initialize with random key
    rng_key = jax.random.PRNGKey(42)
    initial_states = coupler.initialize(rng_key)
    
    # Set initial SST climatology
    initial_sst_clim = create_seasonal_sst_climatology(nlat, nlon, 0.0)
    initial_states["ocean"] = ocean.update_climatology(
        initial_states["ocean"],
        sst_clim=initial_sst_clim,
    )
    
    # Simulation parameters
    days_to_run = 30
    end_time = days_to_run * 86400.0  # Convert to seconds
    save_frequency = 24  # Save every day (24 hourly coupling steps)
    
    print(f"\nRunning {days_to_run}-day coupled simulation...")
    print(f"Coupling timestep: {coupler.coupling_timestep/3600:.1f} hours")
    print(f"Atmosphere timestep: {atmosphere.timestep/60:.1f} minutes")
    print(f"Ocean timestep: {ocean.timestep/3600:.1f} hours")
    
    # Run simulation
    final_states, history = coupler.run(
        initial_states=initial_states,
        start_time=0.0,
        end_time=end_time,
        save_frequency=save_frequency,
    )
    
    print(f"\nSimulation complete!")
    print(f"Saved {len(history)} daily snapshots")
    
    # Analysis
    print("\nFinal state analysis:")
    
    # Ocean SST
    final_sst = final_states["ocean"].prognostic["sst"]
    sst_anom = final_states["ocean"].prognostic["sst_anomaly"]
    
    print(f"\nOcean SST:")
    print(f"  Global mean: {jnp.mean(final_sst) - 273.15:.2f}°C")
    print(f"  Tropical mean (30S-30N): {jnp.mean(final_sst[nlat//3:2*nlat//3]) - 273.15:.2f}°C")
    print(f"  Range: [{jnp.min(final_sst) - 273.15:.2f}, {jnp.max(final_sst) - 273.15:.2f}]°C")
    print(f"  RMS anomaly: {jnp.sqrt(jnp.mean(sst_anom**2)):.3f}K")
    
    # Atmosphere surface temperature
    final_t2m = final_states["atmosphere"].prognostic["temperature"][0]
    
    print(f"\nAtmosphere surface temperature:")
    print(f"  Global mean: {jnp.mean(final_t2m) - 273.15:.2f}°C")
    print(f"  Range: [{jnp.min(final_t2m) - 273.15:.2f}, {jnp.max(final_t2m) - 273.15:.2f}]°C")
    
    # Heat fluxes
    sensible = final_states["atmosphere"].diagnostic["sensible_heat_flux"]
    latent = final_states["atmosphere"].diagnostic["latent_heat_flux"]
    
    print(f"\nSurface heat fluxes (upward positive):")
    print(f"  Sensible heat: {jnp.mean(sensible):.1f} W/m²")
    print(f"  Latent heat: {jnp.mean(latent):.1f} W/m²")
    print(f"  Total: {jnp.mean(sensible + latent):.1f} W/m²")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot final SST
    ax = axes[0, 0]
    im = ax.imshow(
        final_sst - 273.15,
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
        extent=[0, 360, -90, 90],
    )
    ax.set_title("Final SST (°C)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax)
    
    # Plot SST anomaly
    ax = axes[0, 1]
    vmax = max(abs(jnp.min(sst_anom)), abs(jnp.max(sst_anom)))
    im = ax.imshow(
        sst_anom,
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
        extent=[0, 360, -90, 90],
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title("SST Anomaly (K)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax)
    
    # Plot time series of global mean SST
    ax = axes[1, 0]
    times = jnp.array([h.time / 86400.0 for h in history])
    global_sst = jnp.array([
        jnp.mean(h.component_states["ocean"].prognostic["sst"]) - 273.15
        for h in history
    ])
    ax.plot(times, global_sst, "b-", linewidth=2)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Global Mean SST (°C)")
    ax.set_title("SST Evolution")
    ax.grid(True, alpha=0.3)
    
    # Plot heat flux
    ax = axes[1, 1]
    total_heat = sensible + latent
    im = ax.imshow(
        total_heat,
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
        extent=[0, 360, -90, 90],
        vmin=-200,
        vmax=200,
    )
    ax.set_title("Total Surface Heat Flux (W/m²)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig("jax_gcm_slab_ocean_results.png", dpi=150, bbox_inches="tight")
    print(f"\nResults saved to jax_gcm_slab_ocean_results.png")
    
    # Additional diagnostics
    print("\n--- Additional Diagnostics ---")
    
    # Check air-sea temperature difference
    air_sea_diff = final_t2m - final_sst
    print(f"\nAir-sea temperature difference:")
    print(f"  Global mean: {jnp.mean(air_sea_diff):.3f}K")
    print(f"  RMS: {jnp.sqrt(jnp.mean(air_sea_diff**2)):.3f}K")
    
    # Check coupling strength
    heat_flux_anom = final_states["ocean"].diagnostic["heat_flux_anomaly"]
    print(f"\nOcean heat flux anomaly:")
    print(f"  RMS: {jnp.sqrt(jnp.mean(heat_flux_anom**2)):.1f} W/m²")
    print(f"  Range: [{jnp.min(heat_flux_anom):.1f}, {jnp.max(heat_flux_anom):.1f}] W/m²")


if __name__ == "__main__":
    main()